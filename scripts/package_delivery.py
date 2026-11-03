"""Create a source distribution without local datasets, environments or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build(destination):
    root = Path(__file__).resolve().parents[1]
    files = set()
    files.update(root.glob("*.md"))
    for folder in ("api_service", "perception", "material_recognition", "scripts", "tests", "docs", "configs", "examples",
                   "week1_model_selection/scripts", "week1_model_selection/tests", "week1_model_selection/segmentation_practice"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and ".local." not in path.name:
                if path.suffix.lower() in {".py", ".md", ".html", ".css", ".js", ".cjs", ".json", ".csv", ".txt", ".lock", ".yaml", ".ps1"}:
                    files.add(path)
    for name in ("README.md", ".gitignore", ".dockerignore", "compose.api.yaml", "api_service/Dockerfile", "pyproject.toml", "requirements.txt", "requirements-perception.txt",
                 "run_perception.ps1", "week1_model_selection/requirements.txt", "week1_model_selection/MODEL_MANIFEST.md",
                 "week1_model_selection/README.md", "data/enterprise/README.md", "data/enterprise/manifest.template.csv"):
        files.add(root / name)
    studio = root / "annotation_studio"
    files.update(studio.glob("*.md"))
    for extension in ("*.py", "*.ps1", "*.cmd", "*.command", "*.txt"):
        files.update(studio.glob(extension))
    for path in (studio / "02-代码").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".html", ".ps1", ".js", ".cjs"}:
            files.add(path)
    files.add(studio / ".gitignore")
    manifest = {"version": "0.5.0", "files": {}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            manifest["files"][name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        assert len(archive.namelist()) == len(set(archive.namelist()))
    print(f"Packaged {len(files)} files: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    build(parser.parse_args().output)
