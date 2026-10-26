"""Black-box client/SSE verification with an explicitly synthetic model only."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from PIL import Image
import uvicorn
from api_service.app import Settings,create_app
from examples.perception_client import PerceptionClient
from test_api_service import SyntheticModel


def read_event(stream):
    event={}
    while True:
        line=stream.readline().decode().rstrip('\r\n')
        if not line:
            if 'data' in event:
                event['data']=json.loads(event['data'])
                return event
            event={}
            continue
        if line.startswith(':'):
            continue
        key,_,value=line.partition(':')
        event[key]=value.lstrip()


def main():
    report={'complete':False,'real_model_inference_verified':False,'model':'synthetic_test_double'}
    target=ROOT/'output/continuous_http_v04.json';target.parent.mkdir(parents=True,exist_ok=True)
    try:
        with TemporaryDirectory() as directory:
            root=Path(directory);model=SyntheticModel()
            app=create_app(Settings(mode='model',data_dir=root/'data'),model_factory=lambda _:model)
            server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=0,log_level='error'))
            thread=threading.Thread(target=server.run,daemon=True);thread.start()
            try:
                deadline=time.monotonic()+10
                while not server.started and time.monotonic()<deadline:time.sleep(.02)
                if not server.started:raise RuntimeError('Test receiver did not start')
                port=server.servers[0].sockets[0].getsockname()[1]
                client=PerceptionClient(f'http://127.0.0.1:{port}')
                assert client.ready()['new_inference_available']
                paths=[]
                for index,color in enumerate(('red','green','blue')):
                    path=root/f'frame_{index}.png';Image.new('RGB',(8,6),color).save(path);paths.append(path)
                deliveries=[];notifications=[]
                with urlopen(client.base_url+'/v1/events',timeout=10) as stream:
                    assert read_event(stream)['event']=='snapshot'
                    for path in paths:
                        job=client.submit_image(path,prompt='box.',stable=True)
                        while True:
                            event=read_event(stream)
                            if event.get('event')=='job':
                                notifications.append(int(event['id']))
                                if event['data']['id']==job['id'] and event['data']['status']=='completed':break
                        result=client.result(job)
                        deliveries.append({'frame_id':result['frame_id'],'job_id':job['id'],'instances':result['instance_count']})
                replay=client.submit_image(paths[0],prompt='box.',stable=True)
                assert replay['id']==deliveries[0]['job_id']
                assert model.calls==3
                assert notifications==sorted(set(notifications))
                report.update(complete=True,images_processed=3,model_calls=3,replay_did_not_duplicate=True,
                              event_count=len(notifications),deliveries=deliveries)
            finally:
                server.should_exit=True;thread.join(timeout=10)
                if thread.is_alive():raise RuntimeError('Test server did not stop')
    finally:
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: three sequential images through the external client; live SSE completion events; safe retry. Synthetic model only.')


if __name__=='__main__':main()
