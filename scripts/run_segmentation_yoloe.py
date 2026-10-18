"""
YOLOE 分割入口：加载 YOLOE-11s-seg，用文本提示词（set_classes）做
开放词汇检测 + 实例分割，输出与 Grounding DINO + SAM 相同格式的 instances。

注意：
1. 首次运行 set_classes 会自动下载 mobileclip 文本编码器（约 572MB）
   并自动安装 ultralytics/CLIP 包，需要网络，耐心等待。
2. 经验值：YOLOE 对 "box/bag/bottle/carton/packet" 这类词的置信度很低
   （大多在 0.05~0.15），所以默认 conf=0.05、imgsz=1280 才有合理检出。
"""
import os

import numpy as np
from PIL import Image
from ultralytics import YOLOE

_CLASSES = ["box", "bag", "bottle", "carton", "packet"]
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "yoloe-11s-seg.pt")

_MODEL = None  # 模块级缓存：模型 + 文本嵌入只生成一次，多帧共用


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLOE(_MODEL_PATH)
        print("[info] YOLOE 正在生成文本嵌入（首次会下载 mobileclip 约 572MB）...")
        _MODEL.set_classes(_CLASSES)
    return _MODEL


def run_segmentation(image_path, prompt=None, conf=0.05, imgsz=1280):
    """返回 (PIL 原图, instances 列表)。
    instances[i] = {bbox_xyxy, mask, conf, class}
    mask 为原图尺寸的 bool 数组（H, W）。
    """
    model = _get_model()
    img_np = np.array(Image.open(image_path).convert("RGB"))
    r = model.predict(source=img_np, conf=conf, imgsz=imgsz, verbose=False)[0]

    instances = []
    if r.boxes is not None and len(r.boxes):
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            mask = masks[i] > 0.5 if masks is not None else None
            instances.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "conf": float(confs[i]),
                "mask": mask,
                "class": _CLASSES[i] if i < len(_CLASSES) else "",
            })
    return Image.fromarray(img_np), instances
