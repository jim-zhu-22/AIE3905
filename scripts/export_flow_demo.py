"""Export a self-contained, read-only walkthrough of real completed jobs."""
import argparse
import base64
import json
import mimetypes
from pathlib import Path


STAGES = ("preprocess.json", "segmentation.json", "material_input.json", "material_result.json", "result.json")


def snapshot(directory):
    directory = directory.resolve()
    result = json.loads((directory / "perception.json").read_text(encoding="utf-8"))
    docs = {name: json.loads((directory / name).read_text(encoding="utf-8")) for name in STAGES}
    for name, data in docs.items():
        if data["run_id"] != result["run_id"] or data["frame_id"] != result["frame_id"]:
            raise ValueError(f"Cross-task artifact: {name}")
    paths = set(STAGES) | {"report.md", "overlay.jpg", "perception.json"}
    paths.update(name for name in ("rgb.png", "depth.png") if (directory / name).is_file())
    paths.update(item["mask_path"] for item in result["instances"])
    for item in docs["material_input.json"]["items"]:
        paths.update(item[key] for key in ("crop_path", "crop_mask_path") if item.get(key))
    files = {}
    for name in sorted(paths):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {name}")
        mime = {".json": "application/json", ".md": "text/markdown"}.get(path.suffix)
        mime = mime or mimetypes.guess_type(name)[0] or "application/octet-stream"
        files[name] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return {"result": result, "docs": docs, "files": files}


def build(directories, destination):
    root = Path(__file__).resolve().parents[1]
    index = (root / "api_service/demo.html").read_text(encoding="utf-8")
    payload = json.dumps({"snapshots": [snapshot(path) for path in directories]}, ensure_ascii=False)
    # Artifact strings remain data, even if a label contains a script terminator.
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    css = (root / "api_service/static/demo.css").read_text(encoding="utf-8")
    script = (root / "api_service/static/flow.js").read_text(encoding="utf-8")
    shell = (root / "api_service/static/demo_shell.js").read_text(encoding="utf-8")
    html = index.replace('<link rel="stylesheet" href="/assets/demo.css">', '<style>' + css + '</style>')
    html = html.replace('<body class="customer-demo">', '<body class="customer-demo offline-demo">')
    html = html.replace('<script src="/assets/demo_shell.js"></script>', '<script>window.PIPELINE_DEMO_DATA=' + payload + ';</script><script>' + shell + '</script>')
    html = html.replace('<script src="/assets/flow.js"></script>', '<script>' + script + '</script>')
    preferences = (root / "api_service/static/demo_settings.js").read_text(encoding="utf-8")
    html = html.replace('<script src="/assets/demo_settings.js"></script>', '<script>' + preferences + '</script>')
    upload = (root / "api_service/static/demo_upload.js").read_text(encoding="utf-8")
    html = html.replace('<script src="/assets/demo_upload.js"></script>', '<script>' + upload + '</script>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(f"Exported {len(directories)} real task snapshots: {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.result, args.output)
