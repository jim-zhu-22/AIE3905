import collections
import glob
import json
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else 'output/*/*/segmentation.json'
cnt = collections.Counter()

if len(glob.glob(pat)) == 0:
    print('没有匹配到 segmentation.json，请把 glob 模式作为第一个参数传入')
    print('例如: python check_filtered.py "output/new_smoke/*/roi_filter/segmentation.json"')
    sys.exit(1)

for p in sorted(glob.glob(pat)):
    d = json.load(open(p, encoding='utf-8'))
    fid = str(d.get('frame_id', p))[-8:]
    for f in d.get('filtered', []):
        cnt[f['reason']] += 1
        print(fid, f['reason'], f['bbox'], f.get('mask_area'), f.get('object_label'))

print('---')
for k, v in cnt.items():
    print(k, v)
