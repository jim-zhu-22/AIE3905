"""Exercise a running local app with real inputs; this does not measure accuracy."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_pipeline_bundle import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--rgb', type=Path, required=True)
    parser.add_argument('--depth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rgb, depth = args.rgb.read_bytes(), args.depth.read_bytes()
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        health = client.get('/health').raise_for_status().json()
        assert health['model_ready']
        modules = client.get('/v1/modules').raise_for_status().json()
        assert any(v['id']=='material' and v['state']=='ready' for v in modules['modules'])
        jobs = []
        requests = [
            ('resume_external_rgb', '/v1/jobs/from-sample', {'sample_id':'frame_211','modules':'["material"]'}, {'rgb':('rgb.png',rgb)}),
            ('new_rgb', '/v1/perceive', {'prompt':'box. bag. bottle. carton. packet.','frame_id':args.rgb.stem,
                                       'modules':'["material"]'}, {'rgb':('rgb.png',rgb),'depth':('depth.png',depth)})]
        for name, endpoint, data, files in requests:
            created = client.post(endpoint, data=data, files=files)
            assert created.status_code == 202, created.text
            job = created.json()
            deadline = time.monotonic() + 240
            while job['status'] in {'queued','running'} and time.monotonic() < deadline:
                time.sleep(.5)
                job = client.get('/v1/jobs/'+job['id']).raise_for_status().json()
            assert job['status'] == 'partial', job
            result = client.get(job['result_ref']).raise_for_status().json()
            assert result['instance_count'] == 34
            assert result['execution']['new_inference'] == (name == 'new_rgb')
            assert result['execution']['new_attribute_inference']
            for item in result['instances']:
                prediction = item['attributes']['material']
                assert prediction['status'] == 'unknown' and prediction['values']['top1_label']
                mask = client.get(item['mask_ref']).raise_for_status().content
                with Image.open(BytesIO(mask)) as im:
                    assert im.size == (1280,960) and im.getbbox() == tuple(item['bbox_xyxy'])
            canonical = client.get(result['pipeline_result_ref']).raise_for_status().json()
            assert canonical['schema_version'] == 'pipeline-1.0' and canonical['status'] == 'partial'
            archive = client.get(result['download_ref']).raise_for_status().content
            (args.output / f'{name}.zip').write_bytes(archive)
            target = args.output / name
            target.mkdir()
            with ZipFile(BytesIO(archive)) as z:
                assert z.testzip() is None
                for entry in z.infolist():
                    path = (target / entry.filename).resolve()
                    assert path.is_relative_to(target.resolve())
                    if not entry.is_dir():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(z.read(entry))
            checked = validate(target / job['id'])
            jobs.append({'path':name,'job':job,'execution':result['execution'],'validation':checked,'mask_http_checks':34})
            print(f'{name}: 34 masks, material candidates, canonical files and ZIP passed', flush=True)
        report = {'health':health, 'modules':modules, 'runs':jobs, 'status':'passed',
                  'scope':'Real model functionality; no ground-truth accuracy or performance claim'}
        (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
