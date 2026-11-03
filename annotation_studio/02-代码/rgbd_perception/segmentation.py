"""Grounding DINO text proposals or human boxes -> SAM masks, all local."""
import argparse
import hashlib
import html
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'grounding-dino-tiny': ('IDEA-Research/grounding-dino-tiny', 'a2bb814dd30d776dcf7e30523b00659f4f141c71'),
    'sam-vit-base': ('facebook/sam-vit-base', '70c1a07f894ebb5b307fd9eaaee97b9dfc16068f'),
}


def download_models(root, sam_only=False):
    from huggingface_hub import snapshot_download
    for name, (repo, revision) in MODELS.items():
        if sam_only and name != 'sam-vit-base':
            continue
        snapshot_download(repo, revision=revision, local_dir=root/name,
                          allow_patterns=['*.json', '*.txt', '*.safetensors'])


def clean_box(box, size, clip=False):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('Each box must contain x1, y1, x2, y2')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError('Box coordinates must be finite numbers')
    x1, y1, x2, y2 = box
    w, h = size
    if x1 >= x2 or y1 >= y2:
        raise ValueError('Box must have positive width and height')
    if not clip and not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('Manual box is outside the original image')
    result = [max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError('Clipped box has no image intersection')
    return result


def manual_proposals(data, frame_id, size):
    """Accept {boxes:[xyxy,...]} or the existing annotation viewer schema v2."""
    if not isinstance(data, dict):
        raise ValueError('Boxes JSON must be an object')
    if 'objects' in data:
        if data.get('schema_version') != 2 or not isinstance(data['objects'], list):
            raise ValueError('Expected annotation schema_version 2')
        rows = [r for r in data['objects'] if r.get('frame_id') == frame_id]
        if not rows:
            raise ValueError('No annotations match this frame_id; refusing to use another image boxes')
        for row in rows:
            if row.get('image_size') != list(size):
                raise ValueError('Annotation image_size does not match the input')
            if row.get('coordinate_convention') != 'original_image_pixels_xyxy_half_open':
                raise ValueError('Unsupported annotation coordinate convention')
        boxes = [r['object_bbox_xyxy'] for r in rows]
        ids = [r['object_id'] for r in rows]
    else:
        boxes = data.get('boxes')
        if not isinstance(boxes, list):
            raise ValueError('Expected a boxes list or schema v2 objects')
        ids = [None]*len(boxes)
    if len(boxes) > 1000:
        raise ValueError('Too many boxes; split the input into smaller jobs')
    return [dict(bbox=clean_box(b, size), label='manual_object', score=None, source_object_id=oid)
            for b, oid in zip(boxes, ids)]


def box_iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def filter_proposals(proposals, size, nms_threshold=.6, max_objects=50):
    keep=[]
    for proposal in sorted(proposals, key=lambda p: p['score'], reverse=True):
        try:
            proposal={**proposal, 'bbox':clean_box(proposal['bbox'], size, clip=True)}
        except ValueError:
            continue
        if all(box_iou(proposal['bbox'], p['bbox']) <= nms_threshold for p in keep):
            keep.append(proposal)
        if len(keep) >= max_objects:
            break
    return keep


def mask_box(mask):
    y, x = np.where(mask)
    return None if not len(x) else [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]


def restore_crop_boxes(proposals, region, original_size):
    """Processor already undoes model resize; only crop origin remains to restore."""
    x,y,x2,y2=clean_box(region,original_size)
    restored=[]
    for proposal in proposals:
        b=clean_box(proposal['bbox'],(x2-x,y2-y),clip=True)
        restored.append({**proposal,'bbox':[b[0]+x,b[1]+y,b[2]+x,b[3]+y]})
    return restored


class Segmenter:
    def __init__(self, model_root, device='auto', automatic=True):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
        self.torch=torch
        self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use --device cpu')
        torch.set_num_threads(8)
        self.sam_processor=SamProcessor.from_pretrained(model_root/'sam-vit-base', local_files_only=True)
        self.sam=SamModel.from_pretrained(model_root/'sam-vit-base', local_files_only=True).to(self.device).eval()
        self.detector=None
        if automatic:
            self.detector_processor=AutoProcessor.from_pretrained(model_root/'grounding-dino-tiny', local_files_only=True)
            self.detector=AutoModelForZeroShotObjectDetection.from_pretrained(model_root/'grounding-dino-tiny', local_files_only=True).to(self.device).eval()

    def sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def detect(self, image, prompt, threshold, text_threshold, nms_threshold, max_objects, search_region=None):
        if not prompt.strip():
            raise ValueError('Text prompt cannot be empty')
        region=clean_box(search_region,image.size) if search_region is not None else [0,0,image.width,image.height]
        detection_image=image.crop(region)
        inputs=self.detector_processor(images=detection_image, text=prompt, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            outputs=self.detector(**inputs)
        result=self.detector_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[(detection_image.height, detection_image.width)])[0]
        labels=result['text_labels'] if 'text_labels' in result else result['labels']
        proposals=[dict(bbox=b.tolist(), score=float(s), label=str(label), source_object_id=None)
                   for b,s,label in zip(result['boxes'].cpu(),result['scores'].cpu(),labels)]
        filtered=filter_proposals(proposals,detection_image.size,nms_threshold,max_objects)
        return restore_crop_boxes(filtered,region,image.size),len(proposals)

    def masks(self, image, proposals, batch_size=8):
        if not proposals:
            return []
        with self.torch.inference_mode():
            image_inputs=self.sam_processor(images=image, return_tensors='pt').to(self.device)
            embeddings=self.sam.get_image_embeddings(image_inputs['pixel_values'])
            masks=[]
            for start in range(0,len(proposals),batch_size):
                boxes=[p['bbox'] for p in proposals[start:start+batch_size]]
                inputs=self.sam_processor(images=image,input_boxes=[boxes],return_tensors='pt').to(self.device)
                output=self.sam(image_embeddings=embeddings,input_boxes=inputs['input_boxes'],multimask_output=False)
                restored=self.sam_processor.post_process_masks(output.pred_masks.cpu(),inputs['original_sizes'].cpu(),inputs['reshaped_input_sizes'].cpu())[0]
                for m, score in zip(restored,output.iou_scores[0].cpu()):
                    masks.append((m[0].numpy().astype(bool),float(score[0])))
            return masks


class MaterialPredictor:
    """Optional existing five-class baseline; does not turn predictions into labels."""
    def __init__(self, checkpoint):
        import torch
        from material_recognition import LABELS
        from material_recognition.model import build_model
        from material_recognition.data import transform
        self.torch=torch;self.labels=list(LABELS);self.path=checkpoint
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if list(state['labels']) != self.labels:
            raise ValueError('Material checkpoint class order mismatch')
        self.model=build_model(False).eval();self.model.load_state_dict(state['model'])
        self.preprocess=transform(False)

    def predict(self,crop):
        with self.torch.inference_mode():
            scores=self.model(self.preprocess(crop).unsqueeze(0)).softmax(1)[0].tolist()
        best=int(np.argmax(scores))
        return dict(label=self.labels[best],score=scores[best],scores=dict(zip(self.labels,scores)),
                    model_version=self.path.parent.name, region='mask_bounding_box_rgb',calibrated=False)


def export_result(image, proposals, predictions, out, metadata, material=None):
    """Each mask stays independent; overlapping candidates are not treated as a partition."""
    if len(proposals) != len(predictions):
        raise ValueError('Mismatch between boxes and masks')
    out.mkdir(parents=True,exist_ok=False)
    (out/'masks').mkdir();(out/'crops').mkdir()
    image.save(out/'input.jpg',quality=95)
    overlay=np.array(image).copy();instances=[];rejected=[];accepted_masks=[]
    colors=[(237,85,59),(38,167,225),(94,190,95),(241,178,55),(173,116,220),(220,91,163)]
    for i,(proposal,(mask,quality)) in enumerate(zip(proposals,predictions),1):
        if mask.shape != (image.height,image.width):
            raise ValueError('Mask must be in original image coordinates')
        bbox=mask_box(mask)
        if bbox is None:
            rejected.append(dict(proposal_index=i,reason='empty_mask'));continue
        # Different text boxes can converge to the same SAM object.
        if metadata.get('mode') == 'text_grounding_sam':
            duplicate=None
            for old_id,old_mask in accepted_masks:
                union=np.count_nonzero(mask | old_mask)
                if union and np.count_nonzero(mask & old_mask)/union > .9:
                    duplicate=old_id;break
            if duplicate:
                rejected.append(dict(proposal_index=i,reason='duplicate_mask_iou_gt_0.9',kept_instance=duplicate));continue
        identity=f'i{i:04d}';mask_ref=f'masks/{identity}.png';crop_ref=f'crops/{identity}.jpg'
        accepted_masks.append((identity,mask))
        Image.fromarray(mask.astype('uint8')*255).save(out/mask_ref)
        crop=image.crop(bbox);crop.save(out/crop_ref,quality=95)
        rgba=image.convert('RGBA');rgba.putalpha(Image.fromarray(mask.astype('uint8')*255));rgba.crop(bbox).save(out/f'crops/{identity}_cutout.png')
        color=colors[(i-1)%len(colors)]
        overlay[mask]=(overlay[mask]*.58+np.array(color)*.42).astype('uint8')
        instances.append(dict(instance_id=identity,proposal_bbox_xyxy=proposal['bbox'],bbox_xyxy=bbox,
            mask_ref=mask_ref,crop_ref=crop_ref,mask_area_px=int(mask.sum()),
            source_object_id=proposal.get('source_object_id'),grounding_label=proposal['label'],
            detection_score=proposal['score'],sam_predicted_iou=quality,
            material=material.predict(crop) if material else None,
            geometry=None,physical_attributes=None,annotation_status='needs_review'))
        if metadata.get('search_region_xyxy'):
            x1,y1,x2,y2=metadata['search_region_xyxy']
            instances[-1]['mask_outside_search_region_fraction']=float(1-mask[y1:y2,x1:x2].sum()/mask.sum())
    visual=Image.fromarray(overlay);draw=ImageDraw.Draw(visual)
    for i,instance in enumerate(instances):
        b=instance['bbox_xyxy'];color=colors[(int(instance['instance_id'][1:])-1)%len(colors)]
        draw.rectangle(b,outline=color,width=3);draw.text((b[0],max(0,b[1]-15)),instance['instance_id'],fill='white',stroke_width=2,stroke_fill='black')
    visual.save(out/'overlay.jpg',quality=95)
    result={**metadata,'schema_version':'0.1-segmentation','status':'candidates' if instances else 'no_target',
            'image_size':[image.width,image.height],'coordinate_convention':'original_image_pixels_xyxy_half_open',
            'mask_encoding':'binary PNG 0 background 255 foreground','instances':instances,'rejected':rejected,
            'warnings':['Predictions require review; masks may overlap or omit objects.',
                        'Grounding labels are object descriptions, not material ground truth.',
                        'SAM predicted IoU is a model quality estimate, not measured IoU.',
                        'Geometry and physical attribute prediction are not implemented.']}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def material_cell(instance):
        prediction = instance.get('material')
        if not prediction:
            return '未运行'
        return html.escape(f"{prediction['label']} ({prediction['score']:.3f})")

    rows=''.join('<tr><td>'+html.escape(str(x['instance_id']))+'</td><td>'+html.escape(x['grounding_label'])+'</td><td>'+str(x['detection_score'])+'</td><td>'+f'{x["sam_predicted_iou"]:.3f}'+'</td><td>'+material_cell(x)+'</td><td><a href="'+x['mask_ref']+'">Mask</a> / <a href="'+x['crop_ref']+'">ROI</a></td></tr>' for x in instances)
    page='<!doctype html><meta charset="utf-8"><title>自动分割结果</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:15px}img{max-width:100%}td,th{padding:10px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>物体分割候选</h1><p>当前结果需要人工复核，不代表准确率。材质列是 ResNet-18 候选，不是人工真值；Depth 未参与分割。</p><p>模式：'+html.escape(metadata.get('mode',''))+'；候选数：'+str(len(instances))+'</p><img src="overlay.jpg"><p><a href="input.jpg">原图</a> · <a href="result.json">完整 JSON</a> · <a href="annotation_draft.json">框标注草稿</a></p><table><tr><th>编号</th><th>目标描述</th><th>检测分数</th><th>SAM质量估计</th><th>材质候选（置信度）</th><th>文件</th></tr>'+rows+'</table>'
    (out/'index.html').write_text(page,encoding='utf-8')
    draft={'schema_version':2,'objects':[dict(object_id='object_'+str(i+1).zfill(4),frame_id=metadata['frame_id'],
        rgb_file=metadata['frame_id']+'_color.jpg',depth_file=metadata['frame_id']+'_depth.png',image_size=list(image.size),
        time_group='',object_bbox_xyxy=x['bbox_xyxy'],contact_bbox_xyxy=None,coordinate_convention='original_image_pixels_xyxy_half_open',
        material='',sku_id='',annotation_status='needs_review',mask_ref=x['mask_ref']) for i,x in enumerate(instances)]}
    (out/'annotation_draft.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--boxes-json',type=Path,help='Manual boxes or exported viewer schema v2; skips text detection')
    ap.add_argument('--prompt',default='a package. a box. a bottle. a bag.')
    ap.add_argument('--output',type=Path,required=True,help='New output directory; never overwrites a result')
    ap.add_argument('--models',type=Path,default=ROOT/'../05-模型权重')
    ap.add_argument('--download',action='store_true',help='Download pinned public models; inference otherwise stays offline')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--box-threshold',type=float,default=.25)
    ap.add_argument('--text-threshold',type=float,default=.25)
    ap.add_argument('--nms-threshold',type=float,default=.6)
    ap.add_argument('--max-objects',type=int,default=50)
    ap.add_argument('--mask-batch-size',type=int,default=8)
    ap.add_argument('--material-checkpoint',type=Path,help='Optional trusted existing ResNet-18 checkpoint')
    ap.add_argument('--search-region',nargs=4,type=int,metavar=('X1','Y1','X2','Y2'),help='Crop before detection; restore boxes and SAM masks to original image coordinates')
    args=ap.parse_args()
    if args.output.exists():ap.error('Output already exists; choose a new directory')
    image_path=args.image.resolve();out=args.output.resolve()
    if image_path.parent == out or image_path.parent in out.parents:
        ap.error('Output must be outside the input image directory')
    for value in [args.box_threshold,args.text_threshold,args.nms_threshold]:
        if not 0 <= value <= 1:ap.error('Thresholds must lie in [0,1]')
    if args.max_objects < 1 or args.mask_batch_size < 1:ap.error('Counts must be positive')
    with Image.open(image_path) as im:
        if im.getexif().get(274,1) != 1:ap.error('Normalize EXIF orientation before annotation and segmentation')
        image=im.convert('RGB')
    frame_id=image_path.stem.removesuffix('_color')
    if args.search_region and args.boxes_json:ap.error('--search-region applies to text detection only, not manual boxes')
    if args.search_region:args.search_region=clean_box(args.search_region,image.size)
    proposals=manual_proposals(json.loads(args.boxes_json.read_text(encoding='utf-8-sig')),frame_id,image.size) if args.boxes_json else None
    if args.download:download_models(args.models,bool(args.boxes_json))
    loading=time.perf_counter();engine=Segmenter(args.models,args.device,automatic=proposals is None);engine.sync()
    load_ms=(time.perf_counter()-loading)*1000
    if engine.device=='cuda':engine.torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();raw_count=len(proposals) if proposals is not None else 0
    if proposals is None:proposals,raw_count=engine.detect(image,args.prompt,args.box_threshold,args.text_threshold,args.nms_threshold,args.max_objects,args.search_region)
    engine.sync();detected=time.perf_counter()
    predictions=engine.masks(image,proposals,args.mask_batch_size);engine.sync();segmented=time.perf_counter()
    material=MaterialPredictor(args.material_checkpoint) if args.material_checkpoint else None
    metadata=dict(frame_id=frame_id,input_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        mode='manual_boxes_sam' if args.boxes_json else 'text_grounding_sam',prompt=None if args.boxes_json else args.prompt,
        models={k:{'repo':v[0],'revision':v[1]} for k,v in MODELS.items() if not args.boxes_json or k=='sam-vit-base'},
        device=engine.device,depth_used=False,raw_proposal_count=raw_count,filtered_proposal_count=len(proposals),
        search_region_xyxy=args.search_region,detection_input_size=([args.search_region[2]-args.search_region[0],args.search_region[3]-args.search_region[1]] if args.search_region else list(image.size)),
        thresholds={'box':args.box_threshold,'text':args.text_threshold,'box_nms':args.nms_threshold,'duplicate_mask_iou':.9},
        score_calibration='uncalibrated',latency_ms={'model_load':load_ms,'detect':(detected-start)*1000,'sam':(segmented-detected)*1000},
        gpu_peak_allocated_mb=engine.torch.cuda.max_memory_allocated()/1024**2 if engine.device=='cuda' else None)
    result=export_result(image,proposals,predictions,out,metadata,material)
    result['latency_ms']['after_load_total_including_export_and_material']=(time.perf_counter()-start)*1000
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'instances':len(result['instances']),'device':engine.device,'latency_ms':result['latency_ms']},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
