"""Contract tests use synthetic masks, never claim real neural-model accuracy."""

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import threading

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from api_service.app import Settings, create_app, load_bundle
from api_service.import_sample import import_sample
from api_service.schemas import SegmentationResult
from api_service.storage import contained_file


def encoded_image():
    output = BytesIO()
    Image.new("RGB", (8, 6), "white").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def make_bundle(directory):
    (directory / "masks").mkdir(parents=True)
    mask = Image.new("L", (8, 6))
    mask.paste(255, (1, 2, 5, 4))
    mask.save(directory / "masks/i000.png")
    Image.new("RGB", (8, 6)).save(directory / "overlay.jpg")
    (directory / "report.md").write_text("# Synthetic fixture\n", encoding="utf-8")
    result = dict(schema_version="pipeline-1.0", run_id="fixture_run", frame_id="fixture_frame",
                  stage="segmentation", status="ok", prompt="box.", image_size=[8, 6],
                  model_version="synthetic_fixture", instance_count=1, depth={"depth_usable": False},
                  instances=[dict(instance_id="i000", object_label="box", bbox_xyxy=[1, 2, 5, 4],
                                  mask_path="masks/i000.png", det_score=0.8, seg_score=None)])
    (directory / "segmentation.json").write_text(json.dumps(result), encoding="utf-8")


class SyntheticModel:
    def __init__(self, empty=False, raises=False):
        self.calls = 0
        self.empty = empty
        self.raises = raises

    def predict(self, image, prompt, mode):
        self.calls += 1
        if self.raises:
            raise RuntimeError("synthetic inference failure")
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[2:4, 1:5] = True
        instances = [] if self.empty else [SimpleNamespace(mask=mask, label="box", det_score=0.8)]
        return SimpleNamespace(image=image, instances=instances, latency_ms={"total": 1},
                               model_info={"source": "synthetic_test_double"})


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.sample = self.data / "samples" / "fixture"
        make_bundle(self.sample)
        self.settings = Settings(data_dir=self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive_state_and_no_fake_inference(self):
        with TestClient(create_app(self.settings)) as client:
            self.assertFalse(client.get("/health").json()["model_ready"])
            self.assertEqual(client.get("/ready").status_code, 200)
            response = client.post("/v1/segment", json={"prompt": "box.", "image_base64": encoded_image()})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "model_not_ready")

    def test_sample_and_every_artifact_can_be_retrieved(self):
        with TestClient(create_app(self.settings)) as client:
            result = client.get("/v1/examples/fixture").json()
            self.assertFalse(result["execution"]["new_inference"])
            self.assertEqual(result["execution"]["source_run_id"], "fixture_run")
            self.assertEqual(result["instances"][0]["bbox_xyxy"], [1, 2, 5, 4])
            for ref in [result["overlay_ref"], result["report_ref"], result["segmentation_ref"],
                        result["instances"][0]["mask_ref"]]:
                self.assertEqual(client.get(ref).status_code, 200)

    def test_unknown_and_undeclared_files_are_not_published(self):
        (self.sample / "private.txt").write_text("private")
        with TestClient(create_app(self.settings)) as client:
            self.assertEqual(client.get("/v1/examples/unknown").status_code, 404)
            self.assertEqual(client.get("/artifacts/samples/fixture/private.txt").status_code, 404)
            self.assertEqual(client.get("/artifacts/samples/fixture/%2e%2e/private.txt").status_code, 404)

    def test_invalid_inputs(self):
        with TestClient(create_app(self.settings)) as client:
            self.assertEqual(client.post("/v1/segment", json={"prompt": "box."}).status_code, 422)
            self.assertEqual(client.post("/v1/segment", json={"image_base64": encoded_image(), "prompt": "  "}).status_code, 422)
            self.assertEqual(client.post("/v1/segment", json={"image_base64": "broken!", "prompt": "box."}).status_code, 400)
            self.assertEqual(client.post("/v1/segment", json={"image_base64": encoded_image(), "prompt": "box.", "mode": "invalid"}).status_code, 422)

    def test_openapi_and_demo(self):
        with TestClient(create_app(self.settings)) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/docs").status_code, 200)
            schema = client.get("/openapi.json").json()
            self.assertIn("/v1/segment", schema["paths"])
            self.assertIn("SegmentRequest", schema["components"]["schemas"])

    def test_model_loaded_once_and_each_request_gets_a_new_run(self):
        model = SyntheticModel()
        loads = []
        def factory(settings):
            loads.append(settings)
            return model
        settings = Settings(mode="model", data_dir=self.data)
        with TestClient(create_app(settings, model_factory=factory)) as client:
            self.assertEqual(client.get("/ready").status_code, 200)
            payload = {"image_base64": encoded_image(), "prompt": "box.", "depth": {"depth_usable": True}}
            first = client.post("/v1/segment", json=payload)
            second = client.post("/v1/segment", json=payload)
            self.assertEqual(first.status_code, 200, first.text)
            a, b = first.json(), second.json()
            self.assertNotEqual(a["run_id"], b["run_id"])
            self.assertTrue(a["execution"]["new_inference"])
            self.assertFalse(a["depth"]["depth_usable"])
            self.assertEqual(a["instances"][0]["bbox_xyxy"], [1, 2, 5, 4])
            for ref in [a["segmentation_ref"], a["report_ref"], a["overlay_ref"], a["instances"][0]["mask_ref"]]:
                self.assertEqual(client.get(ref).status_code, 200)
            self.assertEqual(len(loads), 1)
            self.assertEqual(model.calls, 2)
            persisted = client.get('/v1/runs/' + a['run_id'])
            self.assertEqual(persisted.status_code, 200)
            self.assertEqual(persisted.json()['run_id'], a['run_id'])

    def test_no_targets_is_a_valid_result(self):
        settings = Settings(mode="model", data_dir=self.data)
        with TestClient(create_app(settings, model_factory=lambda _: SyntheticModel(empty=True))) as client:
            response = client.post("/v1/segment", json={"image_base64": encoded_image(), "prompt": "nothing."})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "no_target")
            self.assertEqual(response.json()["instances"], [])

    def test_failed_model_initialization_reports_not_ready(self):
        def broken(_):
            raise RuntimeError("synthetic missing dependencies")
        with self.assertLogs("aie.api", level="ERROR"):
            with TestClient(create_app(Settings(mode="model", data_dir=self.data), model_factory=broken)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/ready").status_code, 503)

    def test_failed_inference_reports_500_and_releases_lock(self):
        model = SyntheticModel(raises=True)
        with self.assertLogs("aie.api", level="ERROR"):
            with TestClient(create_app(Settings(mode="model", data_dir=self.data), model_factory=lambda _: model)) as client:
                payload = {"image_base64": encoded_image(), "prompt": "box."}
                self.assertEqual(client.post("/v1/segment", json=payload).status_code, 500)
                model.raises = False
                self.assertEqual(client.post("/v1/segment", json=payload).status_code, 200)

    def test_import_checks_binary_masks_boxes_and_does_not_overwrite(self):
        load_bundle(self.sample, check_pixels=True)
        imported, count = import_sample(self.sample, self.data / "another", "copy")
        self.assertEqual(count, 1)
        self.assertTrue((imported / "masks/i000.png").is_file())
        with self.assertRaises(FileExistsError):
            import_sample(self.sample, self.data / "another", "copy")
        bad = Image.new("L", (8, 6), 128)
        bad.save(self.sample / "masks/i000.png")
        with self.assertRaises(ValueError):
            load_bundle(self.sample, check_pixels=True)

    def test_upload_accepts_actual_image_and_returns_persisted_result(self):
        model = SyntheticModel()
        with TestClient(create_app(Settings(mode="model", data_dir=self.data), model_factory=lambda _: model)) as client:
            response = client.post('/v1/segment/upload', data={'prompt': 'box.', 'frame_id': 'upload_test',
                'depth_meta': '{"depth_unit":"mm","depth_scale_to_m":0.001}'},
                files={'image': ('image.jpg', base64.b64decode(encoded_image()), 'image/jpeg')})
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data['frame_id'], 'upload_test')
            self.assertEqual(data['depth']['depth_scale_to_m'], 0.001)
            self.assertFalse(data['depth']['depth_usable'])
            self.assertEqual(client.get('/v1/runs/' + data['run_id']).status_code, 200)
            self.assertEqual(data['execution']['request_id'], response.headers['x-request-id'])

    def test_form_errors_match_json_error_contract(self):
        with TestClient(create_app(self.settings)) as client:
            for meta in ('not-json', '{"depth_scale_to_m":-1}'):
                response = client.post('/v1/segment/upload', data={'prompt':'box.', 'depth_meta':meta},
                    files={'image': ('image.png', base64.b64decode(encoded_image()), 'image/png')})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['error']['code'], 'invalid_request')
                self.assertEqual(response.json()['error']['request_id'], response.headers['x-request-id'])

    def test_errors_are_traceable_and_do_not_echo_uploaded_data(self):
        with TestClient(create_app(self.settings)) as client:
            secret = 'PRIVATE_IMAGE_CONTENT_123456'
            response = client.post('/v1/segment', json={'image_base64':secret, 'prompt':'box.', 'unexpected':secret})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(secret, response.text)
            self.assertEqual(response.json()['error']['request_id'], response.headers['x-request-id'])
            other = client.get('/nonexistent')
            self.assertEqual(other.status_code, 404)
            self.assertNotEqual(other.headers['x-request-id'], response.headers['x-request-id'])

    def test_body_limit_checks_declared_and_chunked_bodies_before_parsing(self):
        settings = Settings(data_dir=self.data, max_request_bytes=64)
        with TestClient(create_app(settings)) as client:
            declared = client.post('/v1/segment', content=b'{}', headers={'content-length':'1000'})
            self.assertEqual(declared.status_code, 413)
            chunked = client.post('/v1/segment', content=iter([b'x'*40, b'y'*40]),
                                  headers={'content-type':'application/json'})
            self.assertEqual(chunked.status_code, 413)
            self.assertEqual(chunked.json()['error']['code'], 'request_too_large')

    def test_image_pixel_and_byte_limits_and_unsupported_format(self):
        payload = {'image_base64': encoded_image(), 'prompt':'box.'}
        with TestClient(create_app(self.settings)) as client:
            with patch('api_service.images.MAX_PIXELS', 10):
                self.assertEqual(client.post('/v1/segment', json=payload).status_code, 413)
            with patch('api_service.images.MAX_IMAGE_BYTES', 10):
                self.assertEqual(client.post('/v1/segment', json=payload).status_code, 413)
            gif = BytesIO()
            Image.new('RGB', (8,6)).save(gif, format='GIF')
            response = client.post('/v1/segment', json={'prompt':'box.',
                'image_base64':base64.b64encode(gif.getvalue()).decode()})
            self.assertEqual(response.status_code, 415)

    def test_result_schema_rejects_inconsistent_counts_status_and_geometry(self):
        original = json.loads((self.sample / 'segmentation.json').read_text())
        for field, value in [('instance_count', 2), ('status','no_target'), ('image_size',[0,6])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                SegmentationResult.model_validate({**original, field:value})
        original['instances'][0]['bbox_xyxy'] = [1,2,50,4]
        with self.assertRaises(ValueError):
            SegmentationResult.model_validate(original)

    def test_invalid_depth_metadata_is_rejected(self):
        with TestClient(create_app(self.settings)) as client:
            for depth in ({'depth_scale_to_m':0}, {'depth_size':[1280]}, {'depth_size':[True,10]}, {'surprise':3}):
                response = client.post('/v1/segment', json={'prompt':'box.', 'image_base64':encoded_image(), 'depth':depth})
                self.assertEqual(response.status_code, 422)

    def test_failed_copy_rolls_back_and_same_name_can_be_retried(self):
        target = self.data / 'import_target'
        with patch('api_service.import_sample.shutil.copyfile', side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(OSError):
                import_sample(self.sample, target, 'copy')
        self.assertEqual(list((target / 'samples').iterdir()), [])
        imported, count = import_sample(self.sample, target, 'copy')
        self.assertTrue((imported / 'segmentation.json').is_file())
        self.assertEqual(count, 1)

    def test_bad_model_mask_rolls_back_and_next_request_can_succeed(self):
        good = SyntheticModel()
        class BadModel:
            def predict(self, image, prompt, mode):
                prediction = good.predict(image, prompt, mode)
                prediction.instances[0].mask = np.zeros((1,1))
                return prediction
        api = create_app(Settings(mode='model', data_dir=self.data), model_factory=lambda _: BadModel())
        with TestClient(api) as client:
            with self.assertLogs('aie.api', level='ERROR'):
                self.assertEqual(client.post('/v1/segment', json={'prompt':'box.', 'image_base64':encoded_image()}).status_code, 500)
            self.assertEqual(list((self.data / 'runs').iterdir()), [])
            api.state.model = good
            response = client.post('/v1/segment', json={'prompt':'box.', 'image_base64':encoded_image()})
            self.assertEqual(response.status_code, 200)

    def test_busy_model_is_rejected_without_starting_another_inference(self):
        entered, release = threading.Event(), threading.Event()
        class BlockingModel(SyntheticModel):
            def predict(self, image, prompt, mode):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('test timed out')
                return super().predict(image, prompt, mode)
        model = BlockingModel()
        with TestClient(create_app(Settings(mode='model', data_dir=self.data), model_factory=lambda _: model)) as client:
            with ThreadPoolExecutor(max_workers=1) as pool:
                payload = {'prompt':'box.', 'image_base64':encoded_image()}
                future = pool.submit(client.post, '/v1/segment', json=payload)
                try:
                    self.assertTrue(entered.wait(3))
                    response = client.post('/v1/segment', json=payload)
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()['error']['code'], 'model_busy')
                    self.assertEqual(response.headers['retry-after'], '1')
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=5).status_code, 200)
        self.assertEqual(model.calls, 1)

    def test_archive_startup_needs_no_write_permissions_and_empty_archive_is_not_ready(self):
        with patch('pathlib.Path.mkdir', side_effect=PermissionError('read-only mount')):
            with TestClient(create_app(self.settings)) as client:
                self.assertEqual(client.get('/ready').status_code, 200)
                self.assertFalse(client.get('/v1/capabilities').json()['new_inference_available'])
        with TestClient(create_app(Settings(data_dir=self.data / 'empty'))) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertEqual(client.get('/ready').status_code, 503)

    def test_windows_and_parent_path_artifacts_are_rejected(self):
        for path in ('../outside.png','C:/outside.png',r'C:\outside.png', '/absolute.png', 'masks/../overlay.jpg'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                contained_file(self.sample, path)

    def test_unknown_run_is_not_retrievable(self):
        with TestClient(create_app(self.settings)) as client:
            response = client.get('/v1/runs/unknown')
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()['error']['code'], 'run_not_found')


if __name__ == "__main__":
    unittest.main()
