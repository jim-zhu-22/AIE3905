"""Bound request bodies before parsing; return traceable, bounded errors."""

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

LOG = logging.getLogger("aie.api")


def fail(status: int, code: str, message: str, *, retry_after: int | None = None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    raise HTTPException(status, detail={"code": code, "message": message}, headers=headers)


def error_response(status, code, message, request_id, *, fields=None, headers=None):
    return JSONResponse(status_code=status, headers=headers,
        content={"error": {"code": code, "message": message, "request_id": request_id,
                           "fields": fields or []}})


def validation_fields(errors):
    # Never echo the uploaded image, request body, or arbitrary exception context.
    return [{"location": [str(part) for part in error["loc"]], "type": error["type"],
             "message": error["msg"][:300]} for error in errors[:20]]


def register_handlers(api):
    @api.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return error_response(exc.status_code, detail.get("code", f"http_{exc.status_code}"),
            detail.get("message", str(exc.detail)), request.state.request_id, headers=exc.headers)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return error_response(422, "invalid_request", "请求字段不符合接口约定。", request.state.request_id,
                              fields=validation_fields(exc.errors()))


class RequestGuard:
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = perf_counter()
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        response_started = False
        status = 500

        async def traced_send(message):
            nonlocal response_started, status
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode()),
                    (b"x-process-time-ms", f"{(perf_counter() - started) * 1000:.2f}".encode())]
            await send(message)

        try:
            if scope["method"] in {"POST", "PUT", "PATCH"}:
                headers = dict(scope.get("headers", []))
                try:
                    length = int(headers.get(b"content-length", b"0"))
                    if length < 0:
                        raise ValueError
                except ValueError:
                    response = error_response(400, "invalid_content_length", "请求长度无效。", request_id)
                    return await response(scope, receive, traced_send)
                if length > self.max_bytes:
                    response = error_response(413, "request_too_large", "请求体超过服务允许的大小。", request_id)
                    return await response(scope, receive, traced_send)
                # Count actual chunks as well: Content-Length may be missing or inaccurate.
                content = bytearray()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if len(content) + len(chunk) > self.max_bytes:
                        response = error_response(413, "request_too_large", "请求体超过服务允许的大小。", request_id)
                        return await response(scope, receive, traced_send)
                    content.extend(chunk)
                    if not message.get("more_body", False):
                        break
                delivered = False

                async def replay():
                    nonlocal delivered
                    if delivered:
                        return await receive()
                    delivered = True
                    return {"type": "http.request", "body": bytes(content), "more_body": False}

                await self.app(scope, replay, traced_send)
            else:
                await self.app(scope, receive, traced_send)
        except Exception:
            LOG.exception("Unhandled request failure request_id=%s", request_id)
            if response_started:
                raise
            response = error_response(500, "internal_error", "服务处理失败，请使用请求编号查询日志。", request_id)
            await response(scope, receive, traced_send)
        finally:
            LOG.info("request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f", request_id,
                     scope["method"], scope["path"], status, (perf_counter() - started) * 1000)
