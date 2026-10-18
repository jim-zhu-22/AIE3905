import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.validate_rgbd_annotations import validate


class AnnotationSchemaTest(unittest.TestCase):
    def test_confirmed_mask_and_prompt_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            Image.new("RGB", (8, 6)).save(base / "rgb.jpg")
            Image.new("I;16", (8, 6)).save(base / "depth.png")
            mask = Image.new("L", (8, 6), 0)
            for y in range(1, 4):
                for x in range(2, 5):
                    mask.putpixel((x, y), 255)
            mask.save(base / "mask.png")
            data = {
                "schema_version": "1.0", "frame_id": "f1", "rgb_path": "rgb.jpg",
                "depth_path": "depth.png", "image_size": [8, 6],
                "depth_unit": "unconfirmed", "alignment": "unconfirmed",
                "targets": [{"prompt_id": "p1", "prompt": "瓶子", "instance_ids": ["o1"]}],
                "instances": [{"instance_id": "o1", "mask_path": "mask.png",
                               "bbox_xyxy": [2, 1, 5, 4], "geometry": "bottle",
                               "material": "plastic", "deformability": "uncertain",
                               "annotation_status": "confirmed"}]
            }
            self.assertEqual(validate(data, base), [])
            data["instances"][0]["bbox_xyxy"] = [0, 0, 3, 3]
            self.assertTrue(any("does not contain mask" in e for e in validate(data, base)))
            data["instances"][0]["bbox_xyxy"] = [2, 1, 5, 4]
            data["targets"][0]["instance_ids"] = ["missing"]
            self.assertTrue(any("unknown instance" in e for e in validate(data, base)))


if __name__ == "__main__":
    unittest.main()
