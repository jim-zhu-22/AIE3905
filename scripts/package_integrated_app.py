"""Package the integrated app with explicitly selected local samples and evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from package_delivery import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = {"version": "0.5.0", "files": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.zip"
        build(source)
        with ZipFile(args.output, "w", ZIP_DEFLATED) as archive:
            def add(name, payload):
                manifest["files"][name] = hashlib.sha256(payload).hexdigest()
                archive.writestr(name, payload)
            with ZipFile(source) as code:
                for name in code.namelist():
                    add("code/" + name, code.read(name))
            for sample in ("frame_211", "frame_211_rgb"):
                directory = args.data_dir / "samples" / sample
                if not (directory / "segmentation.json").is_file():
                    raise FileNotFoundError(directory)
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        add("samples/" + sample + "/" + path.relative_to(directory).as_posix(), path.read_bytes())
            for name in ("tests_all.txt", "tests_legacy.txt", "environment.json", "workbench_v05.png",
                         "live_verification/verification.json", "live_verification/new_rgb.zip",
                         "live_verification/resume_external_rgb.zip"):
                add("evidence/" + name, (args.data_dir / name).read_bytes())
            add("README.md", (root / "docs/app_pipeline_merge_20261026.md").read_bytes())
            archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name, checksum in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum, name
    print(f"Verified {len(manifest['files'])} files: {args.output}")


if __name__ == "__main__":
    main()
