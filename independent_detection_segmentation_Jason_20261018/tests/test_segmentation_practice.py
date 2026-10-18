import unittest
import numpy as np

from segmentation_practice.pipeline import box_iou, deduplicate, fixed_tiles
from scripts.run_own_segmentation import match_boxes
from scripts.evaluate_confirmed_masks import mask_scores, match_masks


class SegmentationGeometryTests(unittest.TestCase):
    def test_tiles_cover_1280x960_with_expected_overlap(self):
        tiles = fixed_tiles(1280, 960)
        self.assertEqual(
            [box for _, box in tiles],
            [(0, 0, 704, 528), (576, 0, 1280, 528),
             (0, 432, 704, 960), (576, 432, 1280, 960)],
        )

    def test_nms_merges_same_object_from_neighboring_tiles(self):
        boxes = [
            {"box": [620, 450, 670, 500], "score": 0.8, "label": "box", "source_tiles": ["left"]},
            {"box": [622, 451, 672, 501], "score": 0.7, "label": "box", "source_tiles": ["right"]},
            {"box": [800, 450, 850, 500], "score": 0.6, "label": "box", "source_tiles": ["right"]},
        ]
        kept = deduplicate(boxes)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["source_tiles"], ["left", "right"])

    def test_box_matching_is_one_to_one(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
        pred = [[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10]]
        pairs, misses, false_positives = match_boxes(gt, pred)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(misses, [])
        self.assertEqual(len(false_positives), 1)
        self.assertEqual(box_iou(gt[0], pred[0]), 1.0)

    def test_mask_iou_dice_and_unmatched_instances(self):
        gt = [np.array([[1, 1], [0, 0]], dtype=bool),
              np.array([[0, 0], [1, 1]], dtype=bool)]
        pred = [np.array([[1, 1], [0, 0]], dtype=bool)]
        self.assertEqual(mask_scores(gt[0], pred[0]), (1.0, 1.0))
        pairs, fn, fp = match_masks(gt, pred)
        self.assertEqual((len(pairs), fn, fp), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
