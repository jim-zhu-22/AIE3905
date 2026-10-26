"""Package the local browser app, module adapters and verification evidence."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root=Path(__file__).resolve().parents[1]
    files=[]
    for folder in ('api_service','material_recognition','week1_model_selection/segmentation_practice','output/api/samples/frame_211'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.name!='modules.local.json' and p.suffix not in {'.pt','.pth','.pyc'})
    names=['.dockerignore','compose.api.yaml','scripts/start_api.ps1','scripts/check_api.py','scripts/check_workspace.py',
           'scripts/lock_api_dependencies.py','scripts/package_app.py','tests/test_api_service.py','tests/test_workspace.py',
           'docs/app_integration.md','docs/app_progress_20261026.md','week1_model_selection/requirements.txt',
           'docs/api_progress_20261026.md','docs/api_review_20261026.md','docs/continuous_input.md','docs/local_inference.md',
           'scripts/send_images.py','scripts/check_continuous.py','scripts/check_model_environment.py','examples/perception_client.py',
           'tests/test_continuous.py','tests/image_queue.test.cjs','output/backend_tests_v04.txt','output/image_queue_tests_v04.txt',
           'output/continuous_http_v04.json','output/continuous_ui_v04.png',
           'week1_model_selection/MODEL_MANIFEST.md','week1_model_selection/README.md',
           'output/app_verification_v03.json','output/app_clean_verification_v03.json','output/app_sample_result_v03.json',
           'output/api_verification_v03.json','output/app_legacy_tests_v03.txt','output/app_workspace_tests_v03.txt',
           'output/app_workbench_v03.png']
    files.extend(root/name for name in names)
    target=root/'deliverables/AIE_app_20261026_v04.zip'
    target.parent.mkdir(exist_ok=True)
    prefix='AIE_app_v04/'
    with ZipFile(target,'w',ZIP_DEFLATED) as archive:
        archive.writestr(prefix+'README.md','''# AIE local perception app v0.4

Read docs/app_progress_20261026.md for the short progress report.
Read docs/app_integration.md for setup and teammate module integration.
Read docs/continuous_input.md for continuous upload and external image producers.
Read docs/local_inference.md for local CPU/GPU dependencies, weights and preflight.

Python 3.12: install api_service/requirements.lock, then run from this directory:
python -m uvicorn api_service.app:app --host 127.0.0.1 --port 8000 --workers 1 --log-config api_service/logging.json

Open http://127.0.0.1:8000/ . The existing frame_211 sample is included.
Existing source integrations: GroundedSegmenter and material_recognition.Predictor.
Extensible module adapters: Python, separate CLI environment, HTTP endpoint.
48 backend tests and 7 browser-queue tests passed. Three sequential image deliveries
through the external client and live SSE completion events were verified with a
synthetic model. Real model inference is provided by the downloader's own CPU/GPU environment, not the delivery machine.
Real neural model inference, material weights and Docker execution are NOT verified.
No model weights, Python environments or private module configuration are included.
''')
        for path in sorted(set(files)):
            archive.write(path,prefix+path.relative_to(root).as_posix())
    with ZipFile(target) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP validation failed')
        print(f'Packaged {len(archive.namelist())} files: {target} ({target.stat().st_size} bytes)')


if __name__=='__main__':
    main()
