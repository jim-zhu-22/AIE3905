"""Verify the running app using the archived sample, without claiming new inference."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',type=Path,default=Path('output/app_verification_v03.json'))
    args=parser.parse_args()
    report={'checked_at':datetime.now(timezone.utc).isoformat(),'base_url':args.url,'complete':False,
            'new_model_inference_verified':False,'docker_verified':False,'checks':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def call(path,expected=200,payload=None):
        data=json.dumps(payload).encode() if payload is not None else None
        request=Request(args.url+path,data=data,headers={'Content-Type':'application/json'})
        try:
            response=urlopen(request,timeout=20)
        except HTTPError as exc:
            response=exc
        with response:
            content=response.read()
            status=response.code
            request_id=response.headers.get('X-Request-ID')
        report['checks'].append({'method':request.get_method(),'path':path,'expected':expected,
                                 'actual':status,'passed':status==expected and bool(request_id)})
        if status!=expected or not request_id:
            raise AssertionError(f'{path}: unexpected response {status}')
        return content

    try:
        for path in ('/','/assets/app.js','/assets/styles.css','/v1/modules','/v1/jobs','/v1/integration-guide',
                     '/v1/progress-report','/v1/modules/template','/openapi.json'):
            call(path)
        job=json.loads(call('/v1/jobs',202,{'sample_id':'frame_211','modules':['material']}))
        deadline=monotonic()+25
        while monotonic()<deadline:
            current=json.loads(call('/v1/jobs/'+job['id']))
            if current['status'] not in {'queued','running'}:
                break
            sleep(.15)
        if current['status']!='partial':
            raise AssertionError(f'Unexpected archived task state: {current}')
        result=json.loads(call(current['result_ref']))
        if result['execution']['new_inference'] or result['instance_count']!=34:
            raise AssertionError('Wrong result origin or count')
        if any(item['attributes']['material']['status']!='skipped' for item in result['instances']):
            raise AssertionError('Archived overlay was used as a model input')
        for key in ('overlay_ref','report_ref','json_ref'):
            call(result[key])
        for item in result['instances']:
            with Image.open(BytesIO(call(item['mask_ref']))) as image:
                if list(image.size)!=result['image_size'] or image.convert('L').getbbox()!=tuple(item['bbox_xyxy']):
                    raise AssertionError('Mask does not agree with result')
        archive_data=call(result['download_ref'])
        with ZipFile(BytesIO(archive_data)) as zipped:
            if zipped.testzip() is not None or not any(path.endswith('perception.json') for path in zipped.namelist()):
                raise AssertionError('Result archive incomplete')
        call('/v1/jobs/missing_task',404)
        call('/v1/jobs',400,{'sample_id':'frame_211','modules':['missing_module']})
        report.update(complete=True,job_id=job['id'],instance_count=result['instance_count'],
                      archived_modules_skipped=True,zip_bytes=len(archive_data))
        args.output.with_name('app_sample_result_v03.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as exc:
        report['failure']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['checks_passed']=sum(item['passed'] for item in report['checks'])
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"PASS: {report['checks_passed']} app HTTP checks; 34 archived masks; ZIP export; explicit skipped modules")


if __name__=='__main__':
    main()
