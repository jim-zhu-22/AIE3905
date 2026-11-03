import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from rgbd_perception.segmentation import clean_box, manual_proposals, filter_proposals, export_result, restore_crop_boxes


class SegmentationContractTest(unittest.TestCase):
    def test_crop_coordinates_restore_once_without_rescaling(self):
        boxes=[dict(bbox=[1.2,2.3,40.1,30.2],score=.8)]
        out=restore_crop_boxes(boxes,[100,50,150,90],(200,100))
        self.assertEqual(out[0]['bbox'],[101,52,141,81])
        self.assertEqual(boxes[0]['bbox'],[1.2,2.3,40.1,30.2])
        clipped=restore_crop_boxes([dict(bbox=[-4,-3,60,50])],[100,50,150,90],(200,100))
        self.assertEqual(clipped[0]['bbox'],[100,50,150,90])
        with self.assertRaises(ValueError):restore_crop_boxes(boxes,[0,0,300,100],(200,100))

    def test_coordinates_and_rejection(self):
        self.assertEqual(clean_box([1.3,2.2,9.2,10.1],(20,20)),[1,2,10,11])
        for b in [[-1,0,10,10],[0,0,float('nan'),10],[2,2,1,3],[0,0,21,21]]:
            with self.assertRaises(ValueError):clean_box(b,(20,20))
        self.assertEqual(clean_box([-1,0,21,22],(20,20),clip=True),[0,0,20,20])

    def test_annotation_does_not_use_wrong_frame_or_size(self):
        row=dict(frame_id='f',object_id='object_0001',image_size=[20,20],object_bbox_xyxy=[1,2,10,12],coordinate_convention='original_image_pixels_xyxy_half_open')
        data={'schema_version':2,'objects':[row]}
        self.assertEqual(manual_proposals(data,'f',(20,20))[0]['source_object_id'],'object_0001')
        for frame,size in [('wrong',(20,20)),('f',(40,40))]:
            with self.assertRaises(ValueError):manual_proposals(data,frame,size)

    def test_duplicate_boxes_removed_without_dropping_separate_objects(self):
        proposals=[dict(bbox=[1,1,10,10],score=.8),dict(bbox=[1,1,10,10],score=.5),dict(bbox=[11,11,18,18],score=.7)]
        result=filter_proposals(proposals,(20,20))
        self.assertEqual(len(result),2);self.assertEqual(result[0]['score'],.8)

    def test_binary_masks_and_reference_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'result';image=Image.new('RGB',(20,12),'blue')
            m=np.zeros((12,20),bool);m[2:5,3:9]=True
            proposals=[dict(bbox=[2,1,10,6],label='box <script>',score=.7)]
            result=export_result(image,proposals,[(m,.9)],out,{'frame_id':'f','mode':'test'})
            inst=result['instances'][0]
            self.assertEqual(inst['bbox_xyxy'],[3,2,9,5])
            self.assertEqual(inst['mask_area_px'],18)
            mask=np.array(Image.open(out/inst['mask_ref']))
            self.assertEqual(mask.shape,(12,20));self.assertEqual(set(np.unique(mask)),{0,255})
            self.assertNotIn('<script>',(out/'index.html').read_text(encoding='utf-8'))
            draft=json.loads((out/'annotation_draft.json').read_text())
            self.assertEqual(draft['objects'][0]['material'],'')
            self.assertIsNone(inst['geometry']);self.assertIsNone(inst['material'])
            with self.assertRaises(FileExistsError):export_result(image,proposals,[(m,.9)],out,{'frame_id':'f'})

    def test_no_target_and_empty_mask_are_valid_results(self):
        with tempfile.TemporaryDirectory() as folder:
            result=export_result(Image.new('RGB',(20,12)),[],[],Path(folder)/'empty',{'frame_id':'f'})
            self.assertEqual(result['status'],'no_target');self.assertEqual(result['instances'],[])
            result=export_result(Image.new('RGB',(20,12)),[dict(bbox=[0,0,2,2],label='x',score=.8)],[(np.zeros((12,20),bool),.2)],Path(folder)/'zero',{'frame_id':'f'})
            self.assertEqual(result['status'],'no_target');self.assertEqual(result['rejected'][0]['reason'],'empty_mask')

    def test_different_proposals_for_the_same_sam_mask_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            mask=np.zeros((12,20),bool);mask[2:7,3:10]=True
            proposals=[dict(bbox=[1,1,10,10],score=.8,label='box'),dict(bbox=[3,2,14,11],score=.6,label='package')]
            result=export_result(Image.new('RGB',(20,12)),proposals,[(mask,.9),(mask,.8)],Path(folder)/'dedup',{'frame_id':'f','mode':'text_grounding_sam'})
            self.assertEqual(len(result['instances']),1)
            self.assertEqual(result['rejected'][0]['reason'],'duplicate_mask_iou_gt_0.9')

if __name__=='__main__':unittest.main()
