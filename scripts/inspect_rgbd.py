"""Read-only RGBD audit and local review artifacts; no material labels inferred."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def inspect(source, output, template):
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'RGBD source directory does not exist: {source}')
    if source == output or source in output.parents:
        raise ValueError('Output must be outside the original data directory')
    output.mkdir(parents=True, exist_ok=True)
    previews = output / 'previews'
    previews.mkdir(exist_ok=True)
    files = sorted(p for p in source.rglob('*') if p.is_file())
    def collect(suffix):
        result = {}
        for p in files:
            if p.name.endswith(suffix):
                key = p.relative_to(source).as_posix()[:-len(suffix)]
                result[key] = p
        return result
    colors, depths = collect('_color.jpg'), collect('_depth.png')
    keys = sorted(colors.keys() & depths.keys())
    rows, errors, depth_samples = [], [], []
    rgb_hashes, depth_hashes = defaultdict(list), defaultdict(list)
    previous = None
    batch = 0
    for i, key in enumerate(keys):
        try:
            with Image.open(colors[key]) as im:
                im.load(); rgb_size, rgb_mode = im.size, im.mode
                rgb = im.convert('RGB'); rgb.thumbnail((640, 480))
                rgb.save(previews / f'{i:04d}_rgb.jpg', quality=86)
            with Image.open(depths[key]) as im:
                im.load(); depth = np.array(im)
            if depth.ndim != 2 or depth.dtype != np.uint16:
                raise ValueError(f'Expected uint16 single-channel depth, got {depth.shape}/{depth.dtype}')
            timestamp = datetime.strptime(Path(key).name, '%Y_%m_%d_%H_%M_%S_%f')
            gap = (timestamp - previous).total_seconds() if previous else 0.0
            if previous is None or gap > 60:
                batch += 1
            previous = timestamp
            positive = depth[depth > 0]
            small = depth[::16, ::16]
            depth_samples.extend(small[small > 0].tolist())
            quantiles = np.percentile(positive, [1, 50, 99]).tolist() if positive.size else [None] * 3
            row = dict(frame_id=key, timestamp=timestamp.isoformat(timespec='milliseconds'),
                       rgb_path=str(colors[key]), depth_path=str(depths[key]),
                       width=rgb_size[0], height=rgb_size[1], rgb_mode=rgb_mode,
                       depth_width=depth.shape[1], depth_height=depth.shape[0], depth_dtype=str(depth.dtype),
                       size_match=rgb_size == (depth.shape[1], depth.shape[0]),
                       zero_fraction=float((depth == 0).mean()), saturation_fraction=float((depth == 65535).mean()),
                       positive_p01=quantiles[0], positive_p50=quantiles[1], positive_p99=quantiles[2],
                       raw_min=int(depth.min()), raw_max=int(depth.max()), gap_seconds=gap,
                       time_group=f'time_{batch:02d}', preview_id=i)
            rows.append(row)
            rgb_hashes[hashlib.sha256(colors[key].read_bytes()).hexdigest()].append(key)
            depth_hashes[hashlib.sha256(depths[key].read_bytes()).hexdigest()].append(key)
        except Exception as exc:
            errors.append({'frame_id': key, 'error': str(exc)})
    if not rows:
        raise ValueError(f'No readable pairs: {errors}')
    lo, hi = np.percentile(depth_samples, [1, 99]) if depth_samples else (0, 1)
    if hi <= lo:
        hi = lo + 1
    cmap = matplotlib.colormaps['turbo']
    frames = []
    for row in rows:
        with Image.open(row['depth_path']) as im:
            depth = np.array(im)
        normalized = np.clip((depth.astype(float) - lo) / (hi - lo), 0, 1)
        visualization = (cmap(normalized)[..., :3] * 255).astype('uint8')
        visualization[depth == 0] = 0
        im = Image.fromarray(visualization); im.thumbnail((640, 480))
        im.save(previews / f"{row['preview_id']:04d}_depth.png")
        frames.append(dict(id=row['frame_id'], time=row['timestamp'], group=row['time_group'],
                           rgb=f"previews/{row['preview_id']:04d}_rgb.jpg",
                           depth=f"previews/{row['preview_id']:04d}_depth.png",
                           width=row['width'], height=row['height'], zero=row['zero_fraction'],
                           median=row['positive_p50'], size_match=row['size_match']))
    stats = dict(source=str(source), file_count=len(files), total_bytes=sum(p.stat().st_size for p in files),
                 rgb_count=len(colors), depth_count=len(depths), paired_count=len(keys), readable_pairs=len(rows),
                 missing_depth=sorted(colors.keys() - depths.keys()), missing_rgb=sorted(depths.keys() - colors.keys()),
                 other_files=[str(p.relative_to(source)) for p in files if not p.name.endswith(('_color.jpg', '_depth.png'))],
                 errors=errors, dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
                 size_mismatches=[r['frame_id'] for r in rows if not r['size_match']],
                 zero_fraction_min=min(r['zero_fraction'] for r in rows),
                 zero_fraction_median=float(np.median([r['zero_fraction'] for r in rows])),
                 zero_fraction_max=max(r['zero_fraction'] for r in rows),
                 max_saturation_fraction=max(r['saturation_fraction'] for r in rows),
                 raw_min=min(r['raw_min'] for r in rows), raw_max=max(r['raw_max'] for r in rows),
                 start=rows[0]['timestamp'], end=rows[-1]['timestamp'],
                 time_groups=dict(Counter(r['time_group'] for r in rows)),
                 median_gap_seconds=float(np.median([r['gap_seconds'] for r in rows[1:]])) if len(rows) > 1 else None,
                 exact_duplicate_rgb_groups=[v for v in rgb_hashes.values() if len(v) > 1],
                 exact_duplicate_depth_groups=[v for v in depth_hashes.values() if len(v) > 1],
                 visualization_raw_range=[float(lo), float(hi)],
                 depth_unit='unconfirmed', geometric_alignment='unconfirmed',
                 annotation_status='unlabeled', split_status='not_assigned')
    (output / 'summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output / 'pairs.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    selected = sorted(set(np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int).tolist()))
    fig, axes = plt.subplots(4, 3, figsize=(13, 13))
    for ax in axes.flat:
        ax.axis('off')
    for ax, idx in zip(axes.flat, selected):
        row = rows[idx]
        ax.imshow(Image.open(previews / f"{row['preview_id']:04d}_rgb.jpg"))
        ax.set_title(f"Frame {idx+1} | {row['timestamp'][11:]} | {row['time_group']}", fontsize=9)
    fig.suptitle('Production RGB overview - chronological samples', fontsize=15)
    fig.tight_layout(); fig.savefig(output / 'rgb_contact_sheet.jpg', dpi=150); plt.close(fig)
    selected_pairs = selected[1::3][:4] or selected[:1]
    fig, axes = plt.subplots(len(selected_pairs), 2, figsize=(11, 4 * len(selected_pairs)), squeeze=False)
    for (left, right), idx in zip(axes, selected_pairs):
        row = rows[idx]
        left.imshow(Image.open(previews / f"{row['preview_id']:04d}_rgb.jpg"))
        with Image.open(row['depth_path']) as im:
            depth = np.array(im)
        masked = np.ma.masked_where(depth == 0, depth)
        cm = matplotlib.colormaps['turbo'].copy(); cm.set_bad('black')
        image = right.imshow(masked, cmap=cm, vmin=lo, vmax=hi)
        fig.colorbar(image, ax=right, fraction=0.035, pad=0.02, label='Raw depth value (unit unconfirmed)')
        left.set_title(f"RGB | {row['timestamp'][11:]}")
        right.set_title(f"Depth | zero pixels {row['zero_fraction']:.1%}")
        left.axis('off'); right.axis('off')
    fig.tight_layout(); fig.savefig(output / 'rgb_depth_examples.jpg', dpi=130); plt.close(fig)
    with (output / 'objects.template.csv').open('w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(['object_id','frame_id','rgb_path','depth_path','sku_id','sequence_id','time_group',
                               'object_x1','object_y1','object_x2','object_y2',
                               'contact_x1','contact_y1','contact_x2','contact_y2',
                               'material','annotation_status','review_notes'])
    page = template.read_text(encoding='utf-8').replace('__FRAMES__', json.dumps(frames, ensure_ascii=False))
    page = page.replace('__RANGE__', f'{lo:.0f}–{hi:.0f}')
    (output / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--template', type=Path, default=Path(__file__).with_name('rgbd_viewer.html'))
    a = p.parse_args()
    inspect(a.source, a.output, a.template)
