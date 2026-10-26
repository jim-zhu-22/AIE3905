"""Offline preflight for the downloader's own CPU/GPU environment. Installs nothing."""

import argparse
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REVISION="a2bb814dd30d776dcf7e30523b00659f4f141c71"
SAM_HASH="3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3"
DINO_HASH="1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    parser.add_argument('--sam-weight',type=Path,default=Path(os.getenv('AIE_SAM_WEIGHT',ROOT/'week1_model_selection/sam2.1_t.pt')))
    parser.add_argument('--verify-hashes',action='store_true')
    parser.add_argument('--load-model',action='store_true',help='Also load the real model locally without downloading')
    args=parser.parse_args()
    os.environ.setdefault('HF_HOME',str(ROOT/'.model_cache/huggingface'))
    os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'output/ultralytics'))
    os.environ['HF_HUB_OFFLINE']='1'
    report={'python':sys.version.split()[0],'executable':sys.executable,'device_requested':args.device,
            'packages':{},'checks':{},'installs_or_downloads_performed':False,'model_loaded':False}
    report['checks']['python_3_12']=sys.version_info[:2]==(3,12)
    loaded={}
    for package,import_name in [('fastapi','fastapi'),('uvicorn','uvicorn'),('Pillow','PIL'),('numpy','numpy'),
                                ('torch','torch'),('torchvision','torchvision'),('transformers','transformers'),('ultralytics','ultralytics')]:
        try:
            installed=version(package)
            loaded[import_name]=importlib.import_module(import_name)
            report['packages'][package]={'version':installed,'importable':True}
        except Exception as exc:
            report['packages'][package]={'importable':False,'reason':str(exc)[:300]}
    report['checks']['dependencies']=all(item['importable'] for item in report['packages'].values())
    torch=loaded.get('torch')
    cuda_available=bool(torch and torch.cuda.is_available())
    report['cuda_available']=cuda_available
    report['device_selected']='cuda' if args.device=='cuda' or (args.device=='auto' and cuda_available) else 'cpu'
    report['checks']['device']=args.device!='cuda' or cuda_available
    if torch and loaded.get('torchvision'):
        try:
            loaded['torchvision'].ops.box_iou(torch.zeros((1,4)),torch.ones((1,4)))
            report['checks']['torchvision_ops']=True
        except (RuntimeError,NotImplementedError) as exc:
            report['checks']['torchvision_ops']=False;report['torchvision_error']=str(exc)[:300]
    sam=args.sam_weight.resolve()
    with sam.open('rb') if sam.is_file() else open(os.devnull,'rb') as source:
        pointer=source.read(100).startswith(b'version https://git-lfs.github.com/spec/')
    report['sam_weight']=str(sam)
    report['checks']['sam_weight']=sam.is_file() and sam.stat().st_size>1024 and not pointer
    if args.verify_hashes and report['checks']['sam_weight']:
        report['checks']['sam_hash']=sha256(sam)==SAM_HASH
    try:
        from huggingface_hub import try_to_load_from_cache
        cached=try_to_load_from_cache('IDEA-Research/grounding-dino-tiny','model.safetensors',revision=REVISION,
                                     cache_dir=Path(os.environ['HF_HOME'])/'hub')
        report['checks']['dino_weight']=isinstance(cached,str) and Path(cached).is_file()
        if report['checks']['dino_weight'] and args.verify_hashes:
            report['checks']['dino_hash']=sha256(Path(cached))==DINO_HASH
        if loaded.get('transformers'):
            loaded['transformers'].AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny',revision=REVISION,local_files_only=True)
            report['checks']['dino_processor']=True
    except Exception as exc:
        report['checks']['dino_processor']=False;report['dino_error']=str(exc)[:300]
    report['ready']=all(report['checks'].values())
    if args.load_model and report['ready']:
        try:
            sys.path.insert(0,str(ROOT))
            from api_service.app import Settings,default_model
            default_model(Settings(mode='model',device=args.device,sam_weight=sam))
            report['model_loaded']=True
        except Exception as exc:
            report['ready']=False;report['model_error']=str(exc)[:500]
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['ready'] else 1


if __name__=='__main__':raise SystemExit(main())
