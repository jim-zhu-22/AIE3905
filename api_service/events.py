"""Replayable server-sent job notifications; payloads never contain image pixels."""

import asyncio
import json
from time import monotonic

from fastapi import Request
from fastapi.responses import StreamingResponse

from .errors import fail


def message(event, data, cursor):
    return f"id: {cursor}\nevent: {event}\ndata: {json.dumps(data,ensure_ascii=False,separators=(',',':'))}\n\n"


def register_event_route(api):
    @api.get("/v1/events", tags=["连续图片输入"], response_class=StreamingResponse)
    async def events(request: Request, after: int | None = None, once: bool = False):
        engine = api.state.workspace
        if engine is None:
            fail(503,"workspace_unavailable","任务工作区不可用。")
        raw_cursor = request.headers.get("last-event-id")
        try:
            cursor = int(raw_cursor) if raw_cursor is not None else after
            if cursor is not None and cursor < 0:
                raise ValueError
        except ValueError:
            fail(422,"invalid_cursor","事件编号必须是非负整数。")

        async def stream():
            nonlocal cursor
            last_heartbeat = monotonic()
            yield "retry: 1500\n\n"
            if cursor is None:
                snapshot = await asyncio.to_thread(engine.store.event_snapshot)
                cursor = snapshot["cursor"]
                yield message("snapshot",snapshot,cursor)
            while True:
                if await request.is_disconnected():
                    return
                reset, batch = await asyncio.to_thread(engine.store.events_after,cursor)
                if reset:
                    snapshot = await asyncio.to_thread(engine.store.event_snapshot)
                    cursor = snapshot["cursor"]
                    yield message("snapshot",snapshot,cursor)
                else:
                    for item in batch:
                        cursor = item["id"]
                        yield message("job",item["job"],cursor)
                if once:
                    return
                if monotonic() - last_heartbeat >= 10:
                    yield ": heartbeat\n\n"
                    last_heartbeat = monotonic()
                await asyncio.sleep(.2)

        return StreamingResponse(stream(),media_type="text/event-stream",
            headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})
