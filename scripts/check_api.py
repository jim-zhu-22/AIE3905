"""Verify real HTTP calls and save evidence, including failed verification runs."""

import argparse
import base64
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image


def verify(args, report):
    checks = report['checks']

    def call(path, expected=200, payload=None, *, raw=None, content_type=None):
        started = perf_counter()
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        request = Request(args.url + path, data=body, headers={'Content-Type': content_type or 'application/json'})
        check = {'method': request.get_method(), 'path':path, 'expected_status':expected, 'passed':False}
        checks.append(check)
        try:
            try:
                response = urlopen(request, timeout=30)
            except HTTPError as exc:
                response = exc
            with response:
                content, code, headers = response.read(), response.code, response.headers
            check.update(actual_status=code, bytes=len(content))
            if code != expected:
                raise AssertionError(f'{path}: expected {expected}, got {code}')
            if not headers.get('X-Request-ID'):
                raise AssertionError(f'{path}: missing request ID')
            if code >= 400:
                assert json.loads(content)['error']['request_id'] == headers['X-Request-ID']
            check['passed'] = True
            return content
        except Exception as exc:
            check['error'] = str(exc)[:300]
            raise
        finally:
            check['elapsed_ms'] = round((perf_counter() - started) * 1000, 2)

    health = json.loads(call('/health'))
    report['health'] = health
    call('/')
    call('/openapi.json')
    call('/docs')
    caps = json.loads(call('/v1/capabilities'))
    assert caps['api_version'] == health['api_version']
    call('/v1/examples')
    result = json.loads(call(f'/v1/examples/{args.sample_id}'))
    assert result['instance_count'] == len(result['instances'])
    assert result['execution']['new_inference'] is False
    report['sample_instances'] = result['instance_count']
    image_data = None
    for key in ('overlay_ref', 'report_ref', 'segmentation_ref'):
        content = call(result[key])
        if key == 'overlay_ref':
            image_data = content
    for item in result['instances']:
        with Image.open(BytesIO(call(item['mask_ref']))) as image:
            assert list(image.size) == result['image_size']
            assert not any(image.convert('L').histogram()[1:255])
            assert image.convert('L').getbbox() == tuple(item['bbox_xyxy'])
    call('/v1/examples/missing_sample', 404)
    call(f'/artifacts/samples/{args.sample_id}/unknown.txt', 404)
    call('/v1/segment', 422, {'prompt':'box.'})
    call('/v1/segment', 400, {'prompt':'box.', 'image_base64':'invalid!'})
    call('/v1/runs/missing_run', 404)
    call('/ready', 200)
    if not health['model_ready']:
        call('/v1/segment', 503, {'prompt':'box.', 'image_base64':base64.b64encode(image_data).decode()})
        boundary = 'aie-api-check-boundary'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nbox.\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image.jpg"\r\n'
                'Content-Type: image/jpeg\r\n\r\n').encode() + image_data + f'\r\n--{boundary}--\r\n'.encode()
        call('/v1/segment/upload', 503, raw=body, content_type=f'multipart/form-data; boundary={boundary}')
    args.output.with_name('sample_response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--sample-id', default='frame_211')
    parser.add_argument('--output', type=Path, default=Path('output/api_verification_v02.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'checked_at':datetime.now().astimezone().isoformat(), 'base_url':args.url,
              'complete':False, 'new_model_inference_verified':False, 'docker_verified':False, 'checks':[]}
    try:
        verify(args, report)
        report['complete'] = True
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        raise
    finally:
        report['checks_passed'] = sum(item['passed'] for item in report['checks'])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"PASS: {report['checks_passed']} HTTP checks; {report['sample_instances']} archived instances. Evidence: {args.output}")


if __name__ == '__main__':
    main()
