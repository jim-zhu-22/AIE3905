"""Download and verify torchvision ResNet-18 weights with resumable ranges."""

import argparse
import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
SIZE = 46830571
SHA_PREFIX = "f37072fd"


def fetch_part(parts: Path, index: int, chunk_size: int) -> Path:
    start = index * chunk_size
    end = min(start + chunk_size, SIZE) - 1
    path = parts / f"{index:04d}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    for attempt in range(5):
        try:
            response = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise ValueError(f"Unexpected response for range {start}-{end}")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Download retries exhausted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=1)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "models/pretrained/hub/checkpoints/resnet18-f37072fd.pth"
    if target.exists():
        print(f"Already present: {target}")
        return
    parts = target.parent / "resnet18_parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_mib * 1024 * 1024
    count = math.ceil(SIZE / chunk_size)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_part, parts, index, chunk_size) for index in range(count)]
        for number, future in enumerate(as_completed(futures), 1):
            future.result()
            if number % 5 == 0 or number == count:
                print(f"Downloaded {number}/{count} chunks", flush=True)
    digest = hashlib.sha256()
    temporary = target.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for index in range(count):
            block = (parts / f"{index:04d}.part").read_bytes()
            digest.update(block)
            output.write(block)
    if temporary.stat().st_size != SIZE or not digest.hexdigest().startswith(SHA_PREFIX):
        temporary.unlink()
        raise ValueError(f"Weight size/hash mismatch: {digest.hexdigest()}")
    temporary.replace(target)
    for index in range(count):
        (parts / f"{index:04d}.part").unlink()
    parts.rmdir()
    print(f"Verified {target} sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
