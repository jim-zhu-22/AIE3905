"""Prioritize existing automatic candidates without changing their review status."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw


def build(workspace, raw, output, limit=40):
    output.mkdir(parents=True, exist_ok=False)
    names = {hashlib.sha256(path.read_bytes()).hexdigest(): path.name for path in raw.glob('*_color.jpg')}
    candidates = []
    for path in sorted((workspace / 'items').glob('*/prediction.json')):
        prediction = json.loads(path.read_text(encoding='utf-8'))
        review_path = path.with_name('review.json')
        if review_path.exists() and json.loads(review_path.read_text(encoding='utf-8')).get('confirmed') is True:
            continue
        for item in prediction.get('objects', []):
            scores = sorted(item.get('model_scores', {}).values(), reverse=True)
            confidence = float(item.get('model_confidence', 0))
            margin = scores[0] - scores[1] if len(scores) > 1 else 0
            entropy = -sum(float(p) * math.log(max(float(p), 1e-12)) for p in scores)
            label = item.get('model_material', '')
            reasons = []
            if confidence < .7 or margin < .2:
                reasons.append('材质不确定')
            if label in {'metal', 'wood', 'fabric'}:
                reasons.append('当前现场人工集缺少该类')
            if item.get('label') in {'cabinet', 'chair', 'table'}:
                reasons.append('疑似设备或背景，需确认是否为目标')
            if not reasons:
                reasons.append('常规复核')
            priority = entropy + (1 if label in {'metal', 'wood', 'fabric'} else 0) + (.5 if len(reasons) > 1 else 0)
            candidates.append({'frame_key': path.parent.name, 'frame_name': names.get(path.parent.name, path.parent.name),
                'object_id': item['id'], 'bbox': item['bbox'], 'proposed_material': label,
                'confidence': confidence, 'margin': margin, 'priority': priority,
                'reason': '；'.join(reasons), 'status': 'needs_review', 'final_material': '',
                'reviewer': '', 'source_prediction': str(path), 'image': str(path.with_name('image.png'))})
    selected, per_frame = [], Counter()
    for item in sorted(candidates, key=lambda item: (-item['priority'], item['frame_key'], item['object_id'])):
        if per_frame[item['frame_key']] >= 2:
            continue
        selected.append(item)
        per_frame[item['frame_key']] += 1
        if len(selected) >= limit:
            break
    (output / 'crops').mkdir()
    cards = []
    for index, item in enumerate(selected):
        with Image.open(item['image']) as source:
            image = source.convert('RGB')
        x1, y1, x2, y2 = item['bbox']
        box = (max(0, int(x1)), max(0, int(y1)), min(image.width, int(x2)), min(image.height, int(y2)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = image.crop(box)
        crop.thumbnail((320, 240))
        crop.save(output / 'crops' / f'{index:03d}.jpg')
        item['crop'] = f'crops/{index:03d}.jpg'
        image.thumbnail((640, 480))
        sx, sy = image.width / source.width, image.height / source.height
        ImageDraw.Draw(image).rectangle((box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy), outline='red', width=3)
        image.save(output / 'crops' / f'{index:03d}_context.jpg')
        cards.append(f'<article><h3>{index+1}. {html.escape(item["frame_name"])}</h3>'
                     f'<img src="crops/{index:03d}_context.jpg"><img src="{item["crop"]}">'
                     f'<p>模型建议：{html.escape(item["proposed_material"])}，分数 {item["confidence"]:.3f}</p>'
                     f'<p>{html.escape(item["reason"])}</p><p>状态：待人工复核</p></article>')
    fields = ['frame_key','frame_name','object_id','proposed_material','confidence','reason','status','final_material','reviewer','source_prediction','crop']
    with (output / 'review_queue.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(selected)
    (output / 'review_queue.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>优先复核清单</title>'
        '<style>body{font-family:Arial,"Microsoft YaHei";max-width:1200px;margin:30px auto;background:#f4f6f8;color:#24334b}article{background:white;margin:18px 0;padding:18px;border-radius:10px}img{max-width:48%;vertical-align:middle;margin-right:10px}</style>'
        '<h1>优先复核清单</h1><p>这些是机器候选，不是人工标签。请确认是否为目标商品，再确认可见表面材质；不确定的保留待核实。'
        'CSV 可填写 final_material 和 reviewer，审核完成后需重新校验再导入训练。此页面用于复核浏览，不自动修改原标注。</p>' + ''.join(cards), encoding='utf-8')
    summary = {'all_candidates': len(candidates), 'selected': len(selected), 'frames': len(per_frame),
               'status': 'needs_review', 'automatic_labels_used_as_truth': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=40)
    args = parser.parse_args()
    build(args.workspace, args.raw, args.output, args.limit)
