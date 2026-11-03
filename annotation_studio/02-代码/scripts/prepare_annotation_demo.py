"""Refresh the existing annotation page and bundle segmentation candidates without changing raw data."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def encode_mask(mask):
    flat=np.asarray(mask,dtype=bool).ravel()
    changes=np.flatnonzero(flat[1:]!=flat[:-1])+1
    counts=np.diff(np.r_[0,changes,len(flat)]).tolist()
    if flat[0]:counts.insert(0,0)
    return dict(encoding='rle-row-major-v1',size=list(mask.shape),counts=counts)


def prepare(inspection, results):
    with (inspection/'pairs.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    frame_map={r['frame_id']:r for r in rows};objects=[];seen=set()
    for folder in results:
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        frame=result['frame_id'];row=frame_map[frame]
        if frame in seen:raise ValueError('Only one prediction variant per frame')
        seen.add(frame)
        rgb_path=Path(row['rgb_path'])
        if not rgb_path.is_file():
            rgb_path=inspection.parent.parent/'映界_rgbd数据'/(frame+'_color.jpg')
        if hashlib.sha256(rgb_path.read_bytes()).hexdigest()!=result['input_sha256']:raise ValueError('Prediction belongs to different RGB bytes')
        for instance in result['instances']:
            maskpath=(folder/instance['mask_ref']).resolve()
            if folder.resolve() not in maskpath.parents:raise ValueError('Mask path outside result folder')
            with Image.open(maskpath) as im:mask=np.asarray(im)>0
            if mask.shape!=(int(row['height']),int(row['width'])):raise ValueError('Mask size mismatch')
            material_prediction = instance.get('material') or {}
            objects.append(dict(object_id=f'object_{len(objects)+1:04d}',frame_id=frame,image_size=result['image_size'],
                object_bbox_xyxy=instance['bbox_xyxy'],contact_bbox_xyxy=None,coordinate_convention='original_image_pixels_xyxy_half_open',
                material=material_prediction.get('label',''),material_confidence=material_prediction.get('score'),
                material_scores=material_prediction.get('scores',{}),material_model_version=material_prediction.get('model_version'),
                sku_id='',annotation_status='needs_review',mask_review_status='needs_review',material_review_status='needs_review',
                auto_source_id=result['input_sha256']+':'+instance['instance_id'],mask_rle=encode_mask(mask)))
    payload=dict(schema_version=2,dataset_id='yingjie_rgbd_20260826_245',objects=objects)
    (inspection/'auto_annotations.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    (inspection/'auto_annotations.js').write_text('window.AUTO_ANNOTATIONS='+json.dumps(payload,ensure_ascii=False)+';',encoding='utf-8')
    frames=[dict(id=r['frame_id'],group=r['time_group'],rgb=f"previews/{int(r['preview_id']):04d}_rgb.jpg",depth=f"previews/{int(r['preview_id']):04d}_depth.png",width=int(r['width']),height=int(r['height'])) for r in rows]
    page=(ROOT/'scripts/rgbd_viewer.html').read_text(encoding='utf-8').replace('__FRAMES__',json.dumps(frames,ensure_ascii=False)).replace('__RANGE__','参见检查结果')
    scripts=['portable_annotation.js','auto_annotations.js','annotation_masks.js','annotation_export.js']
    page=page.replace('</script></html>','</script>'+''.join(f'<script src="{s}"></script>' for s in scripts)+'</html>')
    for name in ['portable_annotation.js','annotation_masks.js','annotation_export.js']:shutil.copy2(ROOT/'scripts'/name,inspection/name)
    target=inspection/'annotate.html'
    if target.exists() and not (inspection/'annotate_before_masks.html').exists():shutil.copy2(target,inspection/'annotate_before_masks.html')
    target.write_text(page,encoding='utf-8')
    print(json.dumps(dict(frames=len(frames),auto_candidates=len(objects),auto_frames=len(seen),page=str(target)),ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inspection',type=Path,default=ROOT/'../04-数据/processed/yingjie_rgbd_inspection')
    ap.add_argument('--results',nargs='+',type=Path,required=True)
    args=ap.parse_args();prepare(args.inspection,args.results)
