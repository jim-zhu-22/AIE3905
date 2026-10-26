"""Send files sequentially, or receive stable new images from a local folder."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.perception_client import ApiError, PerceptionClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images",type=Path,nargs="*")
    parser.add_argument("--watch",type=Path)
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--prompt",default="box. bag. bottle. carton. packet.")
    parser.add_argument("--modules",nargs="*",default=[])
    parser.add_argument("--stable-seconds",type=float,default=1.5)
    args = parser.parse_args()
    if bool(args.images) == bool(args.watch):
        parser.error("Choose a file list OR --watch directory")
    if args.stable_seconds < .5:
        parser.error("--stable-seconds must be at least 0.5")
    client = PerceptionClient(args.url)
    ready = client.ready()
    if not ready.get("new_inference_available"):
        raise SystemExit("Model is not ready. Start the real model environment first; no images were submitted.")

    def send(path):
        job = client.submit_image(path,prompt=args.prompt,modules=args.modules,stable=True)
        print(json.dumps({"file":str(path),"job_id":job["id"],"status":"accepted"},ensure_ascii=False),flush=True)
        finished = client.wait(job)
        output = {"file":str(path),"job_id":job["id"],"status":finished["status"],"result_ref":finished["result_ref"]}
        if finished["result_ref"]:
            output["instance_count"] = client.result(finished)["instance_count"]
        print(json.dumps(output,ensure_ascii=False),flush=True)

    if args.images:
        errors = 0
        for path in args.images:
            try:
                send(path)
            except (ApiError,OSError,ValueError,TimeoutError) as exc:
                errors += 1
                print(json.dumps({"file":str(path),"error":str(exc)},ensure_ascii=False),flush=True)
                if isinstance(exc,ApiError) and exc.status in {429,503}:
                    raise SystemExit("Receiver unavailable; remaining files were not submitted.")
        raise SystemExit(1 if errors else 0)
    directory = args.watch.resolve()
    if not directory.is_dir():
        parser.error("--watch must be an existing directory")
    observed, finished = {}, {}
    try:
        while True:
            paths = {p for p in directory.iterdir() if p.is_file() and not p.is_symlink() and p.suffix.lower() in {".png",".jpg",".jpeg"}}
            for missing in observed.keys() - paths:
                observed.pop(missing,None)
                finished.pop(missing,None)
            for path in sorted(paths):
                stamp = path.stat()
                signature = (stamp.st_size,stamp.st_mtime_ns)
                old = observed.get(path)
                if old is None or old[0] != signature:
                    observed[path] = (signature,time.monotonic())
                    continue
                if finished.get(path) == signature or time.monotonic() - old[1] < args.stable_seconds:
                    continue
                try:
                    send(path)
                    finished[path] = signature
                except (ApiError,OSError,ValueError,TimeoutError) as exc:
                    print(json.dumps({"file":str(path),"error":str(exc)},ensure_ascii=False),flush=True)
                    if isinstance(exc,ApiError) and exc.status in {429,503}:
                        raise SystemExit("Receiver unavailable. Restart the sender after fixing the receiver.")
                    finished[path] = signature
            time.sleep(.5)
    except KeyboardInterrupt:
        print("Folder sender stopped. Already submitted tasks continue in the app.",flush=True)


if __name__ == "__main__":
    main()
