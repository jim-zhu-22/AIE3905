"""Task orchestration and teammate adapter tests, using synthetic model outputs."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image

from api_service.app import Settings, create_app
from api_service.modules import InstanceContext, ModuleRegistry, ModulePrediction
from api_service.workspace import JobStore
from test_api_service import SyntheticModel, encoded_image, make_bundle


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_bundle(self.root / 'samples' / 'fixture')

    def tearDown(self):
        self.temp.cleanup()

    def client(self, *, model=False, registry=None, data=None, factory=None):
        return TestClient(create_app(Settings(mode='model' if model else 'archive', data_dir=data or self.root),
            model_factory=factory or (lambda _: SyntheticModel()), registry=registry))

    def registry(self, modules, injected=None):
        path = self.root / 'modules.json'
        path.write_text(json.dumps({'modules':modules}), encoding='utf-8')
        return ModuleRegistry(path, injected=injected)

    def wait(self, client, job):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            current = client.get('/v1/jobs/' + job['id']).json()
            if current['status'] not in {'queued','running'}:
                return current
            time.sleep(.03)
        self.fail('Task did not finish')

    def image_job(self, client, modules=None, depth=None):
        files = {'rgb':('rgb.png',base64.b64decode(encoded_image()),'image/png')}
        if depth:
            files['depth'] = ('depth.png',depth,'image/png')
        return client.post('/v1/perceive', data={'prompt':'box.','modules':json.dumps(modules or [])}, files=files)

    def test_sample_task_download_and_persistent_history(self):
        with self.client() as client:
            created = client.post('/v1/jobs', json={'sample_id':'fixture'})
            self.assertEqual(created.status_code, 202, created.text)
            job = self.wait(client, created.json())
            self.assertEqual(job['status'], 'completed')
            result = client.get(job['result_ref']).json()
            self.assertEqual(result['instance_count'], 1)
            self.assertFalse(result['execution']['new_inference'])
            self.assertEqual(result['execution']['source_run_id'], 'fixture_run')
            self.assertIsNone(result['image_ref'])
            self.assertEqual(client.get(result['instances'][0]['mask_ref']).status_code, 200)
            archive = client.get(result['download_ref'])
            self.assertEqual(archive.status_code, 200)
            with ZipFile(BytesIO(archive.content)) as zipped:
                self.assertTrue(any(path.endswith('/perception.json') for path in zipped.namelist()))
        with self.client() as client:
            self.assertEqual(client.get('/v1/jobs').json()['total'], 1)
            self.assertEqual(client.get(job['result_ref']).status_code, 200)

    def test_attributes_succeed_and_failed_module_preserves_segmentation(self):
        class Good:
            def analyze(self, context):
                self.assertions = context.rgb_path.is_file() and context.mask_path.is_file()
                return {'label':'paper','score':.8,'source':'test_module','model_version':'test-v1'}
        class Broken:
            def analyze(self, context):
                raise RuntimeError('synthetic module error')
        registry = self.registry([{'id':key,'title':key,'enabled':True,'adapter':'python','factory':'unused:factory'} for key in ('material','geometry')],
                                 {'material':Good(),'geometry':Broken()})
        with self.client(model=True, registry=registry) as client:
            with self.assertLogs('aie.api', level='ERROR'):
                created = self.image_job(client, ['material','geometry'])
                self.assertEqual(created.status_code, 202, created.text)
                job = self.wait(client, created.json())
            self.assertEqual(job['status'], 'partial')
            result = client.get(job['result_ref']).json()
            attrs = result['instances'][0]['attributes']
            self.assertEqual(attrs['material']['label'], 'paper')
            self.assertEqual(attrs['geometry']['status'], 'error')
            self.assertTrue(result['execution']['new_inference'])
            self.assertEqual(client.get(result['image_ref']).status_code, 200)

    def test_sample_does_not_use_overlay_as_original_rgb(self):
        class Never:
            def analyze(self, context):
                raise AssertionError('Overlay must not be classified')
        registry = self.registry([{'id':'material','title':'material','enabled':True,'adapter':'python','factory':'unused:factory'}], {'material':Never()})
        with self.client(registry=registry) as client:
            job = self.wait(client, client.post('/v1/jobs', json={'sample_id':'fixture','modules':['material']}).json())
            result = client.get(job['result_ref']).json()
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['instances'][0]['attributes']['material']['status'], 'skipped')

    def test_cli_adapter_runs_in_separate_interpreter(self):
        registry = self.registry([{'id':'physical','title':'physical','enabled':True,'adapter':'command',
            'command':[sys.executable,'-m','api_service.extensions.command_template','--request','{request}','--response','{response}']}])
        with self.client(model=True, registry=registry) as client:
            job = self.wait(client, self.image_job(client,['physical']).json())
            result = client.get(job['result_ref']).json()
            self.assertEqual(result['instances'][0]['attributes']['physical']['status'], 'unknown')
            self.assertFalse((self.root/'workspace/results'/job['id']/'module_work').exists())

    def test_http_adapter_sends_crops_and_not_local_paths(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                content = b'{"label":"cuboid","score":0.7}'
                self.send_response(200); self.send_header('Content-Length',str(len(content))); self.end_headers(); self.wfile.write(content)
            def log_message(self,*args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            registry = self.registry([{'id':'geometry','title':'geometry','enabled':True,'adapter':'http',
                                       'endpoint':f'http://127.0.0.1:{server.server_port}/analyze'}])
            with self.client(model=True, registry=registry) as client:
                job = self.wait(client,self.image_job(client,['geometry']).json())
                result = client.get(job['result_ref']).json()
                self.assertEqual(result['instances'][0]['attributes']['geometry']['label'],'cuboid')
            self.assertNotIn('rgb_path',received[0])
            with Image.open(BytesIO(base64.b64decode(received[0]['image_base64']))) as crop:
                self.assertEqual(crop.size,(4,2))
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_import_nested_bundle_and_refuse_duplicate(self):
        output = BytesIO()
        with ZipFile(output,'w') as archive:
            for path in (self.root/'samples/fixture').rglob('*'):
                if path.is_file(): archive.write(path,'nested/frame/'+path.relative_to(self.root/'samples/fixture').as_posix())
        with self.client() as client:
            response = client.post('/v1/bundles/import',data={'sample_id':'from_peer'},files={'bundle':('sample.zip',output.getvalue(),'application/zip')})
            self.assertEqual(response.status_code,201,response.text)
            self.assertEqual(client.get('/v1/examples/from_peer').status_code,200)
            self.assertEqual(client.post('/v1/bundles/import',data={'sample_id':'from_peer'},files={'bundle':('sample.zip',output.getvalue(),'application/zip')}).status_code,409)

    def test_unconfigured_models_and_invalid_modules_are_explicit(self):
        with self.client() as client:
            self.assertEqual(self.image_job(client).status_code,503)
            self.assertEqual(self.image_job(client,['missing']).status_code,422)
            self.assertEqual(client.post('/v1/jobs',json={'sample_id':'fixture','modules':['missing']}).status_code,400)
            self.assertEqual(client.get('/v1/jobs/missing').status_code,404)
            self.assertEqual(client.get('/v1/modules').json()['modules'][0]['state'],'unavailable')

    def test_result_query_does_not_expose_private_inputs(self):
        with self.client() as client:
            job = self.wait(client,client.post('/v1/jobs',json={'sample_id':'fixture'}).json())
            self.assertEqual(client.get('/v1/jobs/'+job['id']+'/artifacts/../jobs.sqlite3').status_code,404)
            self.assertEqual(client.get('/v1/jobs/'+job['id']+'/artifacts/options.json').status_code,404)

    def test_queued_cancel_is_atomic_and_queue_is_bounded(self):
        entered, release = threading.Event(),threading.Event()
        class Slow(SyntheticModel):
            def predict(self,*args):
                entered.set()
                if not release.wait(6): raise RuntimeError('test timeout')
                return super().predict(*args)
        with self.client(model=True,factory=lambda _:Slow()) as client:
            try:
                first = self.image_job(client).json()
                self.assertTrue(entered.wait(3))
                second = self.image_job(client).json()
                self.assertEqual(client.post('/v1/jobs/'+second['id']+'/cancel').status_code,200)
                self.assertEqual(client.get('/v1/jobs/'+second['id']).json()['status'],'cancelled')
                client.app.state.workspace.max_pending=1
                self.assertEqual(self.image_job(client).status_code,429)
            finally:
                release.set()
            self.assertEqual(self.wait(client,first)['status'],'completed')

    def test_workspace_owner_lock_prevents_two_workers_on_same_storage(self):
        with self.client() as first:
            with self.client() as second:
                self.assertTrue(first.get('/v1/modules').json()['workspace_available'])
                self.assertFalse(second.get('/v1/modules').json()['workspace_available'])
                self.assertEqual(second.get('/v1/jobs').status_code,503)
                self.assertEqual(second.get('/v1/app/ready').status_code,503)

    def test_interrupted_running_task_is_not_reported_as_success_after_restart(self):
        store = JobStore(self.root/'workspace')
        from api_service.schemas import SegmentOptions
        store.create('job_interrupted','upload',SegmentOptions(prompt='box.'),[],None,'test_request')
        store.update('job_interrupted',status='running')
        with self.client() as client:
            result=client.get('/v1/jobs/job_interrupted').json()
            self.assertEqual(result['status'],'failed')
            self.assertIn('重启',result['error'])

    def test_static_app_and_integration_docs_are_available(self):
        with self.client() as client:
            for path in ('/','/assets/app.js','/assets/styles.css','/v1/modules/template','/v1/integration-guide','/v1/progress-report','/v1/app/ready'):
                self.assertEqual(client.get(path).status_code,200,path)
            schema=client.get('/openapi.json').json()
            self.assertIn('/v1/perceive',schema['paths'])

    def test_attribute_scores_and_unimplemented_values_are_not_fabricated(self):
        with self.assertRaises(ValueError): ModulePrediction(status='ok',score=.7)
        with self.assertRaises(ValueError): ModulePrediction(status='unknown')
        with self.assertRaises(ValueError): ModulePrediction(status='ok',label='paper',score=2)

    def test_raw_depth_is_retained_but_not_marked_processed(self):
        output=BytesIO();Image.new('I;16',(8,6),1000).save(output,format='PNG')
        with self.client(model=True) as client:
            response=self.image_job(client,depth=output.getvalue())
            self.assertEqual(response.status_code,202,response.text)
            job=self.wait(client,response.json())
            result=client.get(job['result_ref']).json()
            self.assertFalse(result['depth']['depth_usable'])
            self.assertIsNotNone(result['depth_ref'])
            self.assertEqual(client.get(result['depth_ref']).status_code,200)
            archive=client.get(result['download_ref'])
            with ZipFile(BytesIO(archive.content)) as zipped:
                self.assertTrue(any(path.endswith('/depth.png') for path in zipped.namelist()))
            wrong=BytesIO();Image.new('I;16',(1,1),1000).save(wrong,format='PNG')
            self.assertEqual(self.image_job(client,depth=wrong.getvalue()).status_code,400)

    def test_existing_material_predictor_contract_receives_rgb_crop(self):
        sizes=[]
        class Predictor:
            def __init__(self,checkpoint): pass
            def predict(self,image):
                sizes.append(image.size)
                return {'label':'paper','score':.8,'scores':{'paper':.8},'model_version':'fixture'}
        checkpoint=self.root/'fake_checkpoint.pt';checkpoint.write_bytes(b'test fixture')
        with patch.dict(sys.modules,{'material_recognition.infer':SimpleNamespace(Predictor=Predictor)}):
            registry=self.registry([{'id':'material','title':'material','enabled':True,'adapter':'material',
                                     'options':{'checkpoint':str(checkpoint)}}])
            with self.client(model=True,registry=registry) as client:
                job=self.wait(client,self.image_job(client,['material']).json())
                result=client.get(job['result_ref']).json()
                self.assertEqual(result['instances'][0]['attributes']['material']['values']['region'],'object_bbox')
                self.assertEqual(result['model_versions']['material'],'fixture')
        self.assertEqual(sizes,[(4,2)])

    def test_invalid_zip_paths_never_write_outside_import_directory(self):
        original=json.loads((self.root/'samples/fixture/segmentation.json').read_text())
        original['instances'][0]['mask_path']='../escape.png'
        output=BytesIO()
        with ZipFile(output,'w') as archive:
            archive.writestr('segmentation.json',json.dumps(original))
            archive.writestr('report.md','report')
            archive.writestr('overlay.jpg',b'bad')
            archive.writestr('../escape.png',b'bad')
        with self.client() as client:
            response=client.post('/v1/bundles/import',data={'sample_id':'bad_zip'},
                                 files={'bundle':('bad.zip',output.getvalue(),'application/zip')})
            self.assertEqual(response.status_code,400)
        self.assertFalse((self.root/'workspace/escape.png').exists())
        self.assertFalse((self.root/'samples/bad_zip').exists())

    def test_damaged_workspace_database_does_not_break_legacy_viewer(self):
        (self.root/'workspace').mkdir()
        (self.root/'workspace/jobs.sqlite3').write_bytes(b'not a database')
        with self.client() as client:
            self.assertEqual(client.get('/health').status_code,200)
            self.assertEqual(client.get('/v1/examples/fixture').status_code,200)
            self.assertEqual(client.get('/v1/jobs').status_code,503)


if __name__=='__main__':
    unittest.main()
