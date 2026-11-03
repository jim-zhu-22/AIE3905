"""Serve one generated training-candidate directory for browser review."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()
    directory = args.directory.resolve()
    missing = [name for name in ("review.html", "preannotations.json") if not (directory / name).is_file()]
    if missing:
        raise SystemExit(
            f"候选目录不完整: {directory}\n"
            f"缺少: {', '.join(missing)}\n"
            "请先运行 scripts/build_training_dataset.py 生成预标注。"
        )
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Review page: http://{args.host}:{args.port}/review.html", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
