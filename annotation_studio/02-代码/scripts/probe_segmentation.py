"""Reproducible five-frame production smoke run, not an accuracy benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=ROOT/'../04-数据/映界_rgbd数据')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--indices',default='23,45,89,156,245',help='1-based positions in sorted *_color.jpg files')
    ap.add_argument('--device',choices=['cuda','cpu','auto'],default='auto')
    args=ap.parse_args()
    files=sorted(args.source.glob('*_color.jpg'))
    indices=[int(x) for x in args.indices.split(',')]
    if not indices or any(i<1 or i>len(files) for i in indices) or len(set(indices)) != len(indices):
        ap.error('Indices must be unique and within the source list')
    if args.source.resolve()==args.output.resolve() or args.source.resolve() in args.output.resolve().parents:
        ap.error('Output must be outside the input dataset')
    args.output.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for index in indices:
        target=args.output/f'frame_{index:03d}'
        command=[sys.executable,'-m','rgbd_perception.segmentation','--image',str(files[index-1]),'--output',str(target),'--device',args.device]
        done=subprocess.run(command,cwd=ROOT,text=True,encoding='utf-8',errors='replace',capture_output=True,env={**os.environ,'HF_HUB_DISABLE_PROGRESS_BARS':'1'})
        (args.output/f'frame_{index:03d}.log').write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
        if done.returncode:
            raise RuntimeError(f'Frame {index} failed. See saved log.')
        result=json.loads((target/'result.json').read_text(encoding='utf-8'))
        summary=dict(frame_index=index,frame_id=result['frame_id'],instances=len(result['instances']),rejected=result['rejected'],device=result['device'],latency_ms=result['latency_ms'],gpu_peak_allocated_mb=result['gpu_peak_allocated_mb'])
        summaries.append(summary)
        (args.output/'summary.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=True),flush=True)
    cards=''.join(f'<h2>Frame {s["frame_index"]}: {s["instances"]} candidates</h2><a href="frame_{s["frame_index"]:03d}/index.html"><img width="720" src="frame_{s["frame_index"]:03d}/overlay.jpg"></a>' for s in summaries)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>分割试跑</title><h1>五帧自动分割试跑</h1><p>无像素级真值，不能计算准确率。点击图片查看详情。</p>'+cards,encoding='utf-8')

if __name__=='__main__':main()
