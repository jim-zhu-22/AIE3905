from pathlib import Path
import numpy as np
from PIL import Image
from perception.runtime import sha256

class Segmenter:
    def __init__(self, checkpoint, device):
        from ultralytics import SAM
        path = Path(checkpoint)
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError(f"SAM checkpoint missing or still a Git LFS pointer: {path}")
        self.model = SAM(str(path))
        self.device = device
        self.metadata = {"segmenter": str(path), "segmenter_sha256": sha256(path)}

    def predict(self, image, boxes):
        if not boxes:
            return []
        results = self.model.predict(source=np.array(image), bboxes=boxes, device=self.device, verbose=False)
        if not results or results[0].masks is None:
            raise RuntimeError("SAM returned no masks for nonempty detection boxes")
        raw_masks = results[0].masks.data.detach().cpu().numpy()
        if len(raw_masks) != len(boxes):
            raise RuntimeError("SAM mask count does not match detection count")
        masks = []
        for raw_mask in raw_masks:
            if raw_mask.shape != (image.height, image.width):
                raw_mask = np.asarray(Image.fromarray((raw_mask > .5).astype(np.uint8) * 255)
                                      .resize(image.size, Image.Resampling.NEAREST))
            masks.append(raw_mask > .5)
        return masks
