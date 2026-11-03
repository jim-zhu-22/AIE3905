"""Build a self-contained handoff ZIP (models + cached images + bootstrap runtimes).

Does not copy virtual environments, credentials, Git history, or raw depth data.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUNTIMES = {
    "python-macos-arm64.tar.gz": (
        "https://github.com/astral-sh/python-build-standalone/releases/download/20261013/"
        "cpython-3.12.15%2B20261013-aarch64-apple-darwin-install_only.tar.gz",
        "ba32036ad8f4c434d97a1ee7feef8b7edc0fac16777639b97e39c74a108e4440"),
    "python-windows-x64.tar.gz": (
        "https://github.com/astral-sh/python-build-standalone/releases/download/20261013/"
        "cpython-3.12.15%2B20261013-x86_64-pc-windows-msvc-install_only.tar.gz",
        "bbc902d4d187ac4914915bf538196a064b49109a33a104965b5164ced5b80a1e"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtimes():
    folder = ROOT / "runtime"
    folder.mkdir(exist_ok=True)
    for name, (url, expected) in RUNTIMES.items():
        path = folder / name
        if not path.exists() or digest(path) != expected:
            print("Download runtime: " + name, flush=True)
            temp = path.with_suffix(".download")
            with urlopen(url, timeout=90) as response, temp.open("wb") as out:
                shutil.copyfileobj(response, out)
            if digest(temp) != expected:
                raise RuntimeError("Runtime checksum mismatch: " + name)
            temp.replace(path)
        (folder / (name + ".sha256")).write_text(expected + "  " + name + "\n", encoding="ascii")
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = runtimes()
    output = args.output or ROOT / "07-报告与交付" / ("自动标注校验包_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    prefix = "AIE3905-Annotation/"
    entries = []
    for name in ("start_annotation_demo.command", "start_annotation_demo.cmd",
                 "start_annotation_demo.ps1", "start_annotation_demo.py",
                 "install_dependencies.command", "install_dependencies.cmd",
                 "requirements-demo.txt", "先看这里.html"):
        entries.append((ROOT / name, name))
    for package in ("rgbd_perception", "material_recognition"):
        for path in (ROOT / "02-代码" / package).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".html", ".md") and "__pycache__" not in path.parts:
                entries.append((path, path.relative_to(ROOT).as_posix()))
    for path in (ROOT / "05-模型权重" / "grounding-dino-tiny").iterdir():
        if path.is_file() and (path.suffix in (".safetensors", ".json", ".txt", ".md") or path.name.startswith("LICENSE")):
            entries.append((path, path.relative_to(ROOT).as_posix()))
    for name in ("05-模型权重/sam2.1_t.pt", "06-实验结果/训练记录/week1_resnet18/best.pt"):
        entries.append((ROOT / name, name))
    for path in runtime.iterdir():
        if path.name in RUNTIMES or path.name in {n + ".sha256" for n in RUNTIMES}:
            entries.append((path, path.relative_to(ROOT).as_posix()))
    workspace = ROOT / "06-实验结果" / "自动标注工作区"
    with tempfile.TemporaryDirectory(prefix="aie3905-package-") as temp:
        snapshot = Path(temp) / "queue.sqlite3"
        # SQLite online backup includes committed WAL state, unlike copying the DB file.
        source = sqlite3.connect((workspace / "queue.sqlite3").as_uri() + "?mode=ro", uri=True)
        target = sqlite3.connect(snapshot)
        source.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")
        rows = target.execute("SELECT id,state FROM jobs ORDER BY created,id").fetchall()
        # A still-running job has no guaranteed result at snapshot time; resume it on recipient side.
        target.execute("UPDATE jobs SET state='pending',error='' WHERE state='processing'")
        # Source-machine absolute paths are not needed to run cached images.
        target.execute("UPDATE jobs SET source='bundle:image-snapshot'")
        target.commit()
        source.close()
        target.close()
        entries.append((snapshot, "06-实验结果/自动标注工作区/queue.sqlite3"))
        completed = 0
        for key, state in rows:
            folder = workspace / "items" / key
            image = folder / "image.png"
            if not image.exists():
                raise FileNotFoundError("Missing cached image: " + key)
            entries.append((image, image.relative_to(ROOT).as_posix()))
            prediction = folder / "prediction.json"
            if state == "done":
                if not prediction.exists():
                    raise FileNotFoundError(prediction)
                entries.append((prediction, prediction.relative_to(ROOT).as_posix()))
                completed += 1
                review = folder / "review.json"
                if review.exists():
                    # Atomic write in service means each read is one complete review version.
                    captured = Path(temp) / (key + "-review.json")
                    captured.write_bytes(review.read_bytes())
                    entries.append((captured, review.relative_to(ROOT).as_posix()))
        manifest = {
            "created": datetime.now().isoformat(), "images": len(rows), "completed_candidates": completed,
            "pending_or_failed": len(rows) - completed, "runtime_sources": RUNTIMES,
            "mac_target": "Apple Silicon arm64; CPU inference default",
            "mac_validation": "Not executed on physical macOS hardware",
            "weights": {name: digest(ROOT / name) for name in (
                "05-模型权重/sam2.1_t.pt", "05-模型权重/grounding-dino-tiny/model.safetensors",
                "06-实验结果/训练记录/week1_resnet18/best.pt")},
        }
        print(f"Packaging {len(rows)} images, {completed} completed candidates, {len(entries)} files", flush=True)
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for index, (path, relative) in enumerate(entries, 1):
                info = zipfile.ZipInfo(prefix + relative)
                info.create_system = 3
                mode = 0o755 if relative.endswith(".command") else 0o644
                info.external_attr = (0o100000 | mode) << 16
                info.compress_type = zipfile.ZIP_STORED if path.name.endswith(".tar.gz") else zipfile.ZIP_DEFLATED
                info._compresslevel = 1
                with path.open("rb") as source_file, archive.open(info, "w", force_zip64=True) as dest:
                    shutil.copyfileobj(source_file, dest, length=4 * 1024 * 1024)
                if index % 50 == 0:
                    print(f"Packaged {index}/{len(entries)} files", flush=True)
            archive.writestr(prefix + "PACKAGE_INFO.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr(prefix + "04-数据/自动标注输入/把新图片放在这里.txt",
                             "把新图片放在此文件夹，也可以从网页上传。已入队的245张图片在输出工作区 items 下的 image.png 快照中。\n")
        print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                          "images": len(rows), "completed_candidates": completed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
