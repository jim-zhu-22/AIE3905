"""Sequential image admission, resumable notifications and idempotent replay."""

import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from api_service.app import Settings,create_app
from api_service.workspace import JobStore
from test_api_service import SyntheticModel,encoded_image,make_bundle
import test_workspace


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.root=Path(self.temp.name)
        make_bundle(self.root/'samples/fixture')

    def tearDown(self):
        self.temp.cleanup()

    def client(self,model=None):
        model=model or SyntheticModel()
        return TestClient(create_app(Settings(mode='model',data_dir=self.root),model_factory=lambda _:model))

    def send(self,client,key='same_image',prompt='box.'):
        return client.post('/v1/perceive',data={'prompt':prompt,'modules':'[]'},
            files={'rgb':('image.png',base64.b64decode(encoded_image()),'image/png')},headers={'Idempotency-Key':key})

    def test_retries_return_same_job_without_running_model_twice(self):
        model=SyntheticModel()
        with self.client(model) as client:
            a=self.send(client);b=self.send(client)
            self.assertEqual(a.status_code,202,a.text);self.assertEqual(b.status_code,202,b.text)
            self.assertEqual(a.json()['id'],b.json()['id'])
            test_workspace.WorkspaceTests.wait(self,client,a.json())
            self.assertEqual(model.calls,1)
            self.assertEqual(client.get('/v1/jobs').json()['total'],1)

    def test_same_key_with_changed_parameters_is_a_conflict(self):
        with self.client() as client:
            self.assertEqual(self.send(client).status_code,202)
            response=self.send(client,prompt='bottle.')
            self.assertEqual(response.status_code,409)
            self.assertEqual(response.json()['error']['code'],'idempotency_conflict')

    def test_concurrent_retries_share_a_single_admission(self):
        model=SyntheticModel()
        with self.client(model) as client:
            with ThreadPoolExecutor(max_workers=3) as pool:
                responses=list(pool.map(lambda _:self.send(client),range(3)))
            self.assertTrue(all(response.status_code==202 for response in responses))
            self.assertEqual(len({response.json()['id'] for response in responses}),1)
            test_workspace.WorkspaceTests.wait(self,client,responses[0].json());self.assertEqual(model.calls,1)

    def test_replay_survives_restart_even_when_model_is_not_enabled(self):
        with self.client() as client:
            original=self.send(client).json();test_workspace.WorkspaceTests.wait(self,client,original)
        with TestClient(create_app(Settings(data_dir=self.root))) as client:
            replay=self.send(client)
            self.assertEqual(replay.status_code,202,replay.text);self.assertEqual(replay.json()['id'],original['id'])

    def test_events_are_ordered_replayable_and_do_not_leak_submission_fingerprint(self):
        with self.client() as client:
            initial=client.get('/v1/events?once=true')
            self.assertEqual(initial.status_code,200,initial.text)
            self.assertIn('event: snapshot',initial.text)
            created=self.send(client).json();test_workspace.WorkspaceTests.wait(self,client,created)
            stream=client.get('/v1/events?after=0&once=true')
            ids=[int(line[4:]) for line in stream.text.splitlines() if line.startswith('id: ')]
            self.assertEqual(ids,sorted(set(ids)));self.assertGreater(len(ids),2)
            self.assertIn('"status":"completed"',stream.text)
            self.assertNotIn('fingerprint',stream.text);self.assertNotIn('client_key',stream.text)
            tail=client.get('/v1/events?once=true',headers={'Last-Event-ID':str(ids[-2])})
            self.assertIn(f'id: {ids[-1]}',tail.text)
            self.assertNotIn(f'id: {ids[0]}\n',tail.text)

    def test_future_or_expired_cursor_resets_to_current_snapshot(self):
        with self.client() as client:
            job=self.send(client).json();test_workspace.WorkspaceTests.wait(self,client,job)
            reset=client.get('/v1/events?after=999999&once=true')
            self.assertIn('event: snapshot',reset.text)
            self.assertIn(job['id'],reset.text)
            self.assertEqual(client.get('/v1/events?after=-1&once=true').status_code,422)

    def test_events_and_job_changes_are_committed_together(self):
        with self.client() as client:
            job=self.send(client).json();test_workspace.WorkspaceTests.wait(self,client,job)
            engine=client.app.state.workspace
            reset,events=engine.store.events_after(0)
            self.assertFalse(reset)
            self.assertEqual(events[0]['job']['status'],'queued')
            self.assertEqual(events[-1]['job']['status'],'completed')

    def test_existing_job_database_is_migrated_without_losing_history(self):
        folder=self.root/'legacy';folder.mkdir()
        with sqlite3.connect(folder/'jobs.sqlite3') as db:
            db.execute('''CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,created_at TEXT,updated_at TEXT,
                       source TEXT,frame_id TEXT,prompt TEXT,modules TEXT,sample_id TEXT,stage TEXT,
                       done INTEGER,total INTEGER,error TEXT,request_id TEXT)''')
            db.execute("INSERT INTO jobs VALUES ('old','completed','t','t','sample','f','box.','[]',NULL,'done',0,0,NULL,'request')")
        db.close()
        store=JobStore(folder)
        self.assertEqual(store.get('old')['frame_id'],'f')
        self.assertEqual(store.event_snapshot()['jobs'][0]['id'],'old')


if __name__=='__main__':unittest.main()
