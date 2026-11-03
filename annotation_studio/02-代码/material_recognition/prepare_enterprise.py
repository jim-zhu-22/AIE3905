"""Export reviewed material crops and masks. Require explicit SKU/sequence splits."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from . import LABELS


def decode_mask(data, size):
    w,h=size
    if data.get('encoding')!='rle-row-major-v1' or data.get('size')!=[h,w]:raise ValueError('Invalid mask encoding/size')
    counts=data.get('counts',[])
    if not counts or len(counts)>w*h+1 or any(type(n)!=int or n<0 for n in counts) or sum(counts)!=w*h:raise ValueError('Invalid mask runs')
    flat=np.zeros(w*h,dtype=np.uint8);pos=0
    for i,n in enumerate(counts):
        if i%2:flat[pos:pos+n]=255
        pos+=n
    return flat.reshape(h,w)


def confirmed(row,kind):
    return row.get(kind+'_review_status')=='confirmed' and bool(row.get(kind+'_reviewed_by','').strip()) and bool(row.get(kind+'_reviewed_at'))


def convert(annotations, source, splits, output):
    data=json.loads(annotations.read_text(encoding='utf-8-sig'))
    if data.get('schema_version')!=2:raise ValueError('Expected annotation schema 2')
    if output.exists():raise FileExistsError('Choose a new output directory')
    if source.resolve()==output.resolve() or source.resolve() in output.resolve().parents:raise ValueError('Output must be outside raw directory')
    with splits.open(encoding='utf-8-sig',newline='') as f:split_rows=list(csv.DictReader(f))
    split_map={r['frame_id']:r for r in split_rows}
    if len(split_map)!=len(split_rows):raise ValueError('Duplicate frame in split table')
    groups=defaultdict(set);prepared=[];skipped=[];ids=set()
    for row in data['objects']:
        oid=row['object_id'];frame=row['frame_id']
        if oid in ids:raise ValueError('Duplicate object ID; merge files before conversion')
        ids.add(oid)
        mat=bool(confirmed(row,'material') and row.get('material') in LABELS)
        seg=bool(confirmed(row,'mask') and row.get('mask_rle'))
        if not(mat or seg):skipped.append(dict(object_id=oid,reason='no_confirmed_supported_label'));continue
        split_row=split_map.get(frame,{})
        split=split_row.get('split');sequence=split_row.get('sequence_id','').strip();sku=row.get('sku_id','').strip()
        if split not in {'train','validation','test'} or not sequence or not sku:raise ValueError(f'{oid}: assign split, sequence_id and sku_id before training export')
        for key in [('sku',sku),('sequence',sequence),('frame',frame)]:groups[key].add(split)
        if Path(frame).name!=frame or '/' in frame or '\\' in frame:raise ValueError('Invalid frame name')
        path=source/(frame+'_color.jpg')
        with Image.open(path) as im:
            size=im.size
            if row.get('image_size')!=list(size):raise ValueError('Image size mismatch')
        box=row['object_bbox_xyxy']
        if len(box)!=4 or any(type(v)!=int for v in box) or not(0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1]):raise ValueError('Invalid original-image bounding box')
        if row.get('coordinate_convention')!='original_image_pixels_xyxy_half_open':raise ValueError('Unsupported coordinates')
        mask=decode_mask(row['mask_rle'],size) if seg else None
        if seg:
            ys,xs=np.where(mask)
            if not len(xs) or [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]!=box:raise ValueError('Mask and box disagree')
        prepared.append((row,path,box,mask,mat,split,sequence,sku))
    leaks=[k for k,v in groups.items() if len(v)>1]
    if leaks:raise ValueError(f'Cross-split leakage: {leaks[:5]}')
    output.mkdir(parents=True);(output/'crops').mkdir();(output/'masks').mkdir()
    manifest=[];instances=[]
    for n,(row,path,box,mask,mat,split,sequence,sku) in enumerate(prepared,1):
        sid=f'enterprise_{n:06d}';base=dict(sample_id=sid,split=split,sku_id=sku,sequence_id=sequence,parent_image_id=row['frame_id'],annotation_status='confirmed',source_object_id=row['object_id'])
        if mat:
            crop_path=f'crops/{sid}.jpg'
            with Image.open(path) as im:im.convert('RGB').crop(box).save(output/crop_path,quality=95)
            manifest.append({**base,'label':row['material'],'label_id':LABELS.index(row['material']),'image_path':crop_path,'source':'reviewed_enterprise_object_rgb'})
        if mask is not None:
            mask_path=f'masks/{sid}.png';Image.fromarray(mask).save(output/mask_path)
            instances.append({**base,'rgb_path':str(path.resolve()),'mask_path':mask_path,'bbox_xyxy':box,'mask_reviewed_by':row['mask_reviewed_by'],'frame_annotation_complete':False})
    fields=['sample_id','split','label','label_id','image_path','source','sku_id','sequence_id','parent_image_id','annotation_status','source_object_id']
    with (output/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(manifest)
    (output/'instances.json').write_text(json.dumps(instances,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(material_samples=len(manifest),reviewed_masks=len(instances),skipped=skipped,notes=['Frame completeness is not certified. Do not treat unlabeled objects as detector background.','Geometry/physical attributes not annotated. Material uses RGB object bbox, retaining background.'])
    (output/'conversion_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['annotations','source','splits','output']:ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args();print(json.dumps(convert(args.annotations,args.source,args.splits,args.output),ensure_ascii=False,indent=2))
