"""Create the local progress delivery without environments or model weights."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    files = []
    for name in ("api_service", "week1_model_selection/segmentation_practice", "output/api/samples/frame_211"):
        files.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    names = [".dockerignore", "compose.api.yaml", "scripts/start_api.ps1", "scripts/check_api.py",
             "scripts/lock_api_dependencies.py", "scripts/package_api_progress.py", "tests/test_api_service.py",
             "docs/api_progress_20261026.md", "docs/api_review_20261026.md",
             "week1_model_selection/requirements.txt", "week1_model_selection/MODEL_MANIFEST.md",
             "week1_model_selection/README.md", "output/api_verification_v02.json",
             "output/api_clean_verification_v02.json", "output/sample_response.json",
             "output/api_unit_tests_v02.txt", "output/api_openapi_v02.json", "output/api_environment_v02.json"]
    files.extend(root / name for name in names)
    archive = root / "deliverables/api_progress_20261026_v02.zip"
    archive.parent.mkdir(exist_ok=True)
    prefix = "api_progress_20261026_v02/"
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        output.writestr(prefix + "README.md", """# AIE API delivery v0.2, 2026-10-26

Read docs/api_review_20261026.md for evaluation, fixes and limitations.
Read api_service/README.md for setup and requests.

Use Python 3.12, install api_service/requirements.lock, then run:
python -m uvicorn api_service.app:app --host 127.0.0.1 --port 8000 --workers 1 --log-config api_service/logging.json

Open http://127.0.0.1:8000/ to read the included archived frame_211 sample.
23 contract tests passed. 52 actual HTTP checks passed in both the working
environment and an isolated fresh environment. Contract tests use synthetic
model doubles. Real model inference and Docker execution remain unverified.

This local delivery includes 34 archived sample masks and their overlay/report,
but contains no Python environments, model weights, or original RGB-D dataset.
""")
        for path in sorted(set(files)):
            output.write(path, prefix + path.relative_to(root).as_posix())
    with ZipFile(archive) as output:
        bad_file = output.testzip()
        if bad_file is not None:
            raise ValueError(f"Invalid ZIP member: {bad_file}")
        print(f"Created {archive}: {len(output.namelist())} files, {archive.stat().st_size} bytes")


if __name__ == "__main__":
    main()
