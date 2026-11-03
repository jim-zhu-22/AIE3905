import base64
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
from api_service.app import default_model, Settings
from api_service.import_sample import import_sample
from api_service.schemas import SegmentationResult
from perception.io.pipeline_contract import SavedSegmenter
from scripts.validate_pipeline_bundle import validate
from test_api_service import encoded_image, SyntheticModel
import test_workspace as workspace_tests


class IntegrationTests(unittest.TestCase):
    setUp = workspace_tests.WorkspaceTests.setUp
    tearDown = workspace_tests.WorkspaceTests.tearDown
    client = workspace_tests.WorkspaceTests.client
    registry = workspace_tests.WorkspaceTests.registry
    wait = workspace_tests.WorkspaceTests.wait
    image_job = workspace_tests.WorkspaceTests.image_job

    def with_original(self):
        source = self.root / 'samples/fixture'
        (source / 'rgb').mkdir()
        Image.new('RGB', (8, 6), 'red').save(source / 'rgb/frame.png')
        (source / 'preprocess.json').write_text(json.dumps({'frame_id':'fixture_frame','image_size':[8,6],'rgb_path':'rgb/frame.png'}))
        data = json.loads((source / 'segmentation.json').read_text())
        data.update(preprocess_ref='preprocess.json')
        data['depth'].update(processing_status='reference_only', source_depth_path='unresolved/source.png')
        (source / 'segmentation.json').write_text(json.dumps(data))
        return source

    def material(self):
        class Candidate:
            def analyze(self, context):
                return {'status':'unknown', 'label':'unknown', 'score':.8, 'reason':'decision_policy_not_validated',
                        'values':{'top1_label':'paper','scores':{'paper':.8,'plastic':.2}}, 'model_version':'test'}
        return self.registry([{'id':'material','title':'material','enabled':True,'adapter':'python','factory':'x:y'}], {'material':Candidate()})

    def test_v04_bundle_import_original_and_pipeline_export(self):
        source = self.with_original()
        archive = BytesIO()
        with ZipFile(archive, 'w') as z:
            for path in source.rglob('*'):
                if path.is_file():
                    z.write(path, 'frame/' + path.relative_to(source).as_posix())
        with self.client(registry=self.material()) as client:
            response = client.post('/v1/bundles/import', data={'sample_id':'merged'}, files={'bundle':('frame.zip',archive.getvalue())})
            self.assertEqual(response.status_code, 201, response.text)
            job = self.wait(client, client.post('/v1/jobs', json={'sample_id':'merged','modules':['material']}).json())
            result = client.get(job['result_ref']).json()
            self.assertFalse(result['execution']['new_inference'])
            self.assertTrue(result['execution']['new_attribute_inference'])
            self.assertEqual(result['status'], 'partial')
            canonical = client.get(result['pipeline_result_ref']).json()
            self.assertEqual(canonical['schema_version'], 'pipeline-1.0')
            self.assertEqual(canonical['instances'][0]['material']['top1_label'], 'paper')
            self.assertEqual(canonical['instances'][0]['material']['label'], 'unknown')
            directory = self.root / 'workspace/results' / job['id']
            self.assertEqual(validate(directory)['status'], 'passed')
            self.assertEqual(len(SavedSegmenter(directory / 'segmentation.json').instances), 1)
            with ZipFile(BytesIO(client.get(result['download_ref']).content)) as z:
                self.assertIn(job['id'] + '/result.json', z.namelist())
                self.assertIn(job['id'] + '/rois/i000_bbox_0.png', z.namelist())
                self.assertEqual(len([n for n in z.namelist() if n.endswith('preprocess.json')]), 1)

    def test_original_rgb_can_be_supplied_without_segmentation_model(self):
        with self.client(registry=self.material()) as client:
            response = client.post('/v1/jobs/from-sample', data={'sample_id':'fixture','modules':'["material"]'},
                                   files={'rgb':('rgb.png',base64.b64decode(encoded_image()))})
            self.assertEqual(response.status_code, 202, response.text)
            job = self.wait(client, response.json())
            result = client.get(job['result_ref']).json()
            self.assertTrue(result['execution']['new_attribute_inference'])
            self.assertEqual(result['instances'][0]['instance_id'], 'i000')

    def test_wrong_original_rgb_is_rejected_before_job_creation(self):
        out = BytesIO()
        Image.new('RGB', (2, 2)).save(out, format='PNG')
        with self.client() as client:
            response = client.post('/v1/jobs/from-sample', data={'sample_id':'fixture'}, files={'rgb':('rgb.png',out.getvalue())})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(client.get('/v1/jobs').json()['total'], 0)

    def test_import_does_not_follow_external_original_path(self):
        source = self.with_original()
        data = json.loads((source / 'preprocess.json').read_text())
        data['rgb_path'] = '../outside.png'
        (source / 'preprocess.json').write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            import_sample(source, self.root, 'bad')

    def test_empty_status_is_compatible_in_both_directions(self):
        source = self.root / 'samples/fixture/segmentation.json'
        data = json.loads(source.read_text())
        data.update(instances=[], instance_count=0, status='no_target')
        self.assertEqual(SegmentationResult.model_validate(data).status, 'no_match')
        with self.client(model=True, factory=lambda _: SyntheticModel(empty=True)) as client:
            job = self.wait(client, self.image_job(client).json())
            result = client.get(job['result_ref']).json()
            self.assertEqual(result['status'], 'no_match')
            directory = self.root / 'workspace/results' / job['id']
            self.assertEqual(validate(directory)['status'], 'passed')

    def test_default_model_uses_modular_detector_metadata(self):
        weight = self.root / 'sam.pt'
        weight.write_bytes(b'fixture')
        model = SimpleNamespace(model_info={}, detection_backend=SimpleNamespace(box_threshold=.25,text_threshold=.2), nms_threshold=.5)
        with patch('perception.pipeline.segmentation.GroundedSegmenter', return_value=model):
            result = default_model(Settings(sam_weight=weight))
        self.assertEqual(result.model_info['thresholds']['box'], .25)


if __name__ == '__main__':
    unittest.main()
