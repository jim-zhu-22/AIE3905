import os
import sys
import glob
from collections import Counter
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else "."
if not os.path.isdir(root):
    print("目录不存在:", root)
    sys.exit(1)

print("根目录:", root)

counts = {}
samples = {}
ext = Counter()
for dp, dn, fn in os.walk(root):
    rel = os.path.relpath(dp, root)
    counts[rel] = len(fn)
    samples[rel] = sorted(fn)[:3]
    for f in fn:
        ext[os.path.splitext(f)[1].lower()] += 1

print("\n子目录与文件数:")
for k in sorted(counts):
    print("  %-46s %d" % (k, counts[k]))
    for f in samples[k]:
        print("      " + f)

print("\n扩展名统计:", dict(ext))

rgb = []
for pat in ("**/*color*.jpg", "**/*color*.png", "**/*rgb*.jpg", "**/*.jpg"):
    rgb = sorted(glob.glob(os.path.join(root, pat), recursive=True))
    if rgb:
        break

print("\nRGB 候选:", len(rgb))
for p in rgb[:5]:
    print("   ", os.path.relpath(p, root))

if rgb:
    im = Image.open(rgb[0])
    print("首帧尺寸:", im.size, im.mode)
    d0 = os.path.dirname(rgb[0])
    print("\n首帧同目录文件（前 8 个）:")
    for f in sorted(os.listdir(d0))[:8]:
        print("   ", f)

    dps = sorted(glob.glob(os.path.join(root, "**/*depth*"), recursive=True))
    if dps:
        dm = Image.open(dps[0])
        print("\n深度样例:", os.path.relpath(dps[0], root), dm.size, dm.mode)

    # 配对检查：看 _color 换成 _depth 后是否存在
    base = os.path.splitext(rgb[0])[0].replace("_color", "")
    for suf in ("_depth.png", "_depth.jpg", "_d.png", ".png"):
        cand = base + suf
        if os.path.exists(cand):
            dm = Image.open(cand)
            print("配对成功:", os.path.relpath(cand, root), dm.size, dm.mode)
            break
    else:
        print("按 _color->_depth 规则没配到深度图，需按上面同目录文件的实际命名改规则")
