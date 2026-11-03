"""Paired full-image / fixed-workspace experiment. Diagnostic references are partial."""
import argparse
import csv
import hashlib
import html
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from rgbd_perception.segmentation import Segmenter, export_result, box_iou, MODELS, clean_box
from scripts.probe_production_batch import SELECTION


def partial_matches(references, predictions, threshold=.5):
    if not references or not predictions:return []
    iou=np.array([[box_iou(r,p) for p in predictions] for r in references])
    rows,cols=linear_sum_assignment(-((iou>=threshold).astype(float)*1000+iou))
    return [dict(reference_index=int(i),prediction_index=int(j),bbox_iou=float(iou[i,j])) for i,j in zip(rows,cols) if iou[i,j]>=threshold]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=ROOT/'../04-数据/映界_rgbd数据')
    ap.add_argument('--search-region',nargs=4,type=int,default=[430,0,920,700])
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    args=ap.parse_args()
    if args.source.resolve()==args.output.resolve() or args.source.resolve() in args.output.resolve().parents:ap.error('Output must be outside source')
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[s for s in SELECTION if s['index'] in [23,45,89,156,245]]
    config=dict(indices=[s['index'] for s in cases],search_region=args.search_region,
        prompt='a package. a box. a bottle. a bag.',box_threshold=.25,text_threshold=.25,box_nms=.6,mask_duplicate_iou=.9,max_objects=50,
        changed_factor='detector input crop only; SAM always uses full original image',
        reference_note='Historical manually selected subset of 4 boxes/frame, not exhaustive or independently reviewed ground truth. Not an unseen-SKU test.')
    (args.output/'experiment.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.output/'partial_reference_boxes.json').write_text(json.dumps([{'frame_id':s['frame'],'frame_index':s['index'],'boxes':s['boxes']} for s in cases],indent=2),encoding='utf-8')
    engine=Segmenter(ROOT/'../05-模型权重',args.device);results=[]
    for case_index,s in enumerate(cases):
        path=args.source/(s['frame']+'_color.jpg')
        with Image.open(path) as im:image=im.convert('RGB')
        region=clean_box(args.search_region,image.size)
        preview=image.copy();ImageDraw.Draw(preview).rectangle(region,outline='yellow',width=4)
        preview.save(args.output/f'frame_{s["index"]:03d}_region.jpg',quality=95)
        for variant in (['full','crop'] if case_index%2==0 else ['crop','full']):
            if engine.device=='cuda':engine.torch.cuda.reset_peak_memory_stats()
            engine.sync();start=time.perf_counter()
            proposals,raw=engine.detect(image,config['prompt'],.25,.25,.6,50,region if variant=='crop' else None)
            engine.sync();detected=time.perf_counter()
            masks=engine.masks(image,proposals);engine.sync();segmented=time.perf_counter()
            metadata=dict(frame_id=s['frame'],input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode='text_grounding_sam',
                prompt=config['prompt'],models={k:{'repo':v[0],'revision':v[1]} for k,v in MODELS.items()},device=engine.device,depth_used=False,
                search_region_xyxy=region if variant=='crop' else None,raw_proposal_count=raw,filtered_proposal_count=len(proposals),
                thresholds={'box':.25,'text':.25,'box_nms':.6,'duplicate_mask_iou':.9},
                latency_ms={'detect':(detected-start)*1000,'sam':(segmented-detected)*1000},
                gpu_peak_allocated_mb=engine.torch.cuda.max_memory_allocated()/1024**2 if engine.device=='cuda' else None)
            result=export_result(image,proposals,masks,args.output/f'frame_{s["index"]:03d}_{variant}',metadata)
            matches=partial_matches(s['boxes'],[i['bbox_xyxy'] for i in result['instances']])
            record=dict(frame_index=s['index'],variant=variant,candidates=len(result['instances']),reference_matches=len(matches),
                reference_count=len(s['boxes']),matched_pairs=matches,latency_ms=metadata['latency_ms'],gpu_peak_allocated_mb=metadata['gpu_peak_allocated_mb'])
            results.append(record)
            (args.output/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(json.dumps(record),flush=True)
    cards=[]
    for case in cases:
        index=case['index'];pair={r['variant']:r for r in results if r['frame_index']==index}
        cards.append(f'<section><h2>第 {index} 帧</h2><a href="frame_{index:03d}_region.jpg">查看本次工作区域</a><div class="pair">'+''.join(
            f'<article><h3>{title}：{pair[v]["candidates"]} 个候选</h3><p>历史参考框命中 {pair[v]["reference_matches"]}/4（非整体召回率）</p><a href="frame_{index:03d}_{v}/index.html"><img src="frame_{index:03d}_{v}/overlay.jpg"></a></article>' for v,title in [('full','整图'),('crop','裁剪后检测')])+'</div></section>')
    page='<!doctype html><meta charset="utf-8"><title>工作区域优化对比</title><style>body{font:16px system-ui;max-width:1700px;margin:auto;padding:24px;background:#f4f6f8}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;padding:20px;margin:24px 0}img{width:100%}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>整图与工作区域检测对比</h1><p>唯一变化：检测前裁剪 [430,0,920,700]；模型、文字、阈值和 SAM 原图输入保持一致。该区域是当前相机的实验配置，未标定为所有场景的通用区域。</p><p>候选数不是正确商品数。每帧仅有4个历史参考框，不是完整真值；不能计算整图准确率或精确率。这5帧已参与调试，不能作为独立最终测试集。</p><a href="comparison.json">量化诊断 JSON</a>'+''.join(cards)
    (args.output/'index.html').write_text(page,encoding='utf-8')
    review=args.output/'annotation_review_20';review.mkdir()
    files=sorted(args.source.glob('*_color.jpg'))
    indices=sorted(set([s['index']-1 for s in cases]+[int(x) for x in np.linspace(0,len(files)-1,20)]))
    # Keep the actual 20 chosen images explicit; never call these reviewed labels.
    indices=[s['index']-1 for s in cases]+[i for i in indices if i not in [s['index']-1 for s in cases]][:15]
    indices.sort()
    manifest=[]
    for i in indices:
        shutil.copy2(files[i],review/files[i].name)
        manifest.append(dict(frame_index=i+1,frame_id=files[i].stem.removesuffix('_color'),rgb_file=files[i].name,annotation_status='not_started',sku_id='',sequence_id='',split='unassigned'))
    with (review/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    (review/'标注说明.txt').write_text('此目录为20帧待标注队列，不是已完成验证集。\n用既有标注工具按manifest序号找到原图，为所有可见商品画框，并人工核查；每帧注明已完成或跳过原因。选一部分补充像素级Mask。\n5张对比图片已参与调试，不可作为独立测试。先整理SKU与关联序列，再决定训练/验证/测试划分；当前所有split均为unassigned。\n当前Mask候选不能直接当作真值。\n',encoding='utf-8-sig')

if __name__=='__main__':main()
