import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.prepare_annotation_demo import encode_mask
from material_recognition.prepare_enterprise import convert,decode_mask
from material_recognition.data import ManifestDataset


class EnterpriseAnnotationsTest(unittest.TestCase):
    def fixture(self,root):
        source=root/'raw';source.mkdir();rows=[]
        for i in range(2):
            frame=f'frame{i}';Image.new('RGB',(20,15),'white').save(source/(frame+'_color.jpg'))
            mask=np.zeros((15,20),bool);mask[2:10,3:12]=True;mask[4,6]=False
            rows.append(dict(object_id=f'object_{i+1:04d}',frame_id=frame,image_size=[20,15],object_bbox_xyxy=[3,2,12,10],coordinate_convention='original_image_pixels_xyxy_half_open',sku_id=f'sku{i}',material='paper',mask_rle=encode_mask(mask),**{f'{k}_{field}':v for k in ['mask','material'] for field,v in [('review_status','confirmed'),('reviewed_by','human'),('reviewed_at','2026-10-03T00:00:00Z')]}))
        annotations=root/'annotations.json';annotations.write_text(json.dumps(dict(schema_version=2,objects=rows)),encoding='utf-8')
        splits=root/'splits.csv';splits.write_text('frame_id,split,sequence_id\nframe0,train,seq0\nframe1,validation,seq1\n',encoding='utf-8')
        return source,annotations,splits,rows

    def test_export_matches_existing_training_loader_and_exact_masks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src,ann,splits,rows=self.fixture(root);out=root/'out'
            result=convert(ann,src,splits,out);self.assertEqual(result['material_samples'],2);self.assertEqual(result['reviewed_masks'],2)
            ds=ManifestDataset(out/'manifest.csv','train');tensor,label=ds[0];self.assertEqual(tuple(tensor.shape),(3,224,224));self.assertEqual(label,0)
            with Image.open(out/'masks/enterprise_000001.png') as im:np.testing.assert_array_equal(np.array(im),decode_mask(rows[0]['mask_rle'],(20,15)))

    def test_drafts_excluded_and_cross_split_sku_or_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src,ann,splits,rows=self.fixture(root)
            for row in rows:
                row['material_review_status']='needs_review';row['mask_review_status']='needs_review'
            ann.write_text(json.dumps(dict(schema_version=2,objects=rows)),encoding='utf-8')
            report=convert(ann,src,splits,root/'draft');self.assertEqual(report['material_samples'],0);self.assertEqual(report['reviewed_masks'],0)
            for row in rows:row['material_review_status']='confirmed';row['sku_id']='same_sku'
            ann.write_text(json.dumps(dict(schema_version=2,objects=rows)),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'leakage'):convert(ann,src,splits,root/'leak')
            self.assertFalse((root/'leak').exists())
            rows[1]['sku_id']='different';ann.write_text(json.dumps(dict(schema_version=2,objects=rows)),encoding='utf-8')
            splits.write_text('frame_id,split,sequence_id\nframe0,train,same\nframe1,validation,same\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'leakage'):convert(ann,src,splits,root/'leak2')


if __name__=='__main__':unittest.main()
