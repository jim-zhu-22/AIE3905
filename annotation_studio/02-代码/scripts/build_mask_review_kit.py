"""Package a frame-level SAM review draft together with the complete annotation kit."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from build_annotation_bundle import build
from prepare_annotation_demo import encode_mask

def make_draft(annotations,result_folder):
    draft=json.loads(annotations.read_text(encoding='utf-8-sig'))
    result=json.loads((result_folder/'result.json').read_text(encoding='utf-8'))
    lookup={i['source_object_id']:i for i in result['instances']}
    expected=[r for r in draft['objects'] if r['frame_id']==result['frame_id']]
    if set(lookup)!={r['object_id'] for r in expected}:raise ValueError('SAM results must preserve all source objects')
    for row in expected:
        pred=lookup[row['object_id']]
        path=(result_folder/pred['mask_ref']).resolve()
        if result_folder.resolve() not in path.parents:raise ValueError('Invalid mask path')
        with Image.open(path) as im:mask=np.array(im)>0
        if mask.shape!=(row['image_size'][1],row['image_size'][0]):raise ValueError('Mask size mismatch')
        row['object_bbox_xyxy']=pred['bbox_xyxy'];row['mask_rle']=encode_mask(mask)
        row['auto_source_id']=result['input_sha256']+':manual:'+row['object_id']
        row['contact_bbox_xyxy']=None
        for kind in ['mask','material']:
            row[kind+'_review_status']='needs_review';row[kind+'_reviewed_by']='';row[kind+'_reviewed_at']=''
        row['annotation_status']='needs_review'
    draft['frame_reviews']=[]
    return draft,result

def supplement(destination,annotations,result_folder):
    draft,result=make_draft(annotations,result_folder)
    raw=destination/'data/raw'/f"{result['frame_id']}_color.jpg"
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=result['input_sha256']:raise ValueError('Wrong raw image for SAM draft')
    example=destination/'examples';example.mkdir()
    shutil.copy2(annotations,example/'original_boxes.json')
    (example/'mask_review_draft.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(result_folder/'overlay.jpg',example/'sam_draft_preview.jpg')
    js='window.REVIEW_DRAFT='+json.dumps(draft,ensure_ascii=False)+';\n'
    js+='''(()=>{const box=document.createElement('div');box.className='card';box.innerHTML='<h2>已有27个框 继续检查模型轮廓</h2><p>27个框来自你们的人工标注；轮廓由SAM生成，全部待复核。原有10个纸质、17个塑料标签保留。</p><button id="loadReviewDraft" class="primary">载入27个物体的轮廓草稿</button> <a href="../examples/sam_draft_preview.jpg" target="_blank">查看模型轮廓预览</a>';document.querySelector('main').insertBefore(box,document.querySelector('.layout'));document.getElementById('loadReviewDraft').onclick=()=>{try{window.loadAnnotationData(window.REVIEW_DRAFT);}catch(e){message(e.message);}};})();'''
    (destination/'demo/review_draft.js').write_text(js,encoding='utf-8')
    page=destination/'demo/annotate.html'
    text=page.read_text(encoding='utf-8').replace('</html>','<script src="review_draft.js"></script></html>')
    text=text.replace('① 框住一个商品　② 选择包装材质　③ 继续框下一个商品','① 载入草稿或框商品　② 检查轮廓与材质　③ 保存本图　④ 按图号批量导出')
    page.write_text(text,encoding='utf-8')
    guide='''标注同学操作卡

1. 完整解压，用浏览器打开START_HERE.html（Windows和Mac都可以）。
2. 本次27个框：点击“载入27个物体的轮廓草稿”。导入会询问是否替换现有草稿，请先备份。
3. 填写实际复核人的姓名。选择商品，检查轮廓，有问题用画笔/橡皮修正；分别确认轮廓和材质。
4. 检查本图商品是否标全，勾选整帧完成，点击“保存本图标注”，继续下一张。
5. 在页面下方“已标注图片”列表，按图号勾选要交付的图片，可多选/全选。
6. 点击“导出选中图片的数据集（ZIP）”。完整标注包已包含离线导出资源，无需手动找原图；仅资源不完整时展开“原图缺失时使用”，选择本包data/raw。
7. 将下载的ZIP整体发给负责训练的同学，不需要另找原图、不需要写脚本转换。

保存本图是当前浏览器的本地保存；换浏览器、移动目录或清缓存前要导出ZIP或备份JSON。
需要继续编辑时，从ZIP取出annotations.json导回本工具。不要导入COCO文件。
Mask是模型草稿，不是你们已经完成的人工标签；已检查的商品逐个确认，不要为了导出批量假确认。

训练同学操作卡
解压导出的ZIP：images原始RGB，depth原始16位Depth，masks单通道0/255 PNG，previews检查图。
annotations.json保留所有草稿，用于继续标注；instances.json建立原图与Mask的关系。
annotations/instances_reviewed.json是正式COCO分割标注，只纳入Mask全部复核且整帧已检查的图片。
如果COCO的images为空，查看quality_report.json并退回复核，不要直接训练草稿。
先根据SKU和关联序列建立独立的训练/验证/测试划分；类别1统一为商品。材质作为独立属性保留。
COCO segmentation为按列未压缩RLE，可通过pycocotools.frPyObjects转换；也可直接读取mask_file的PNG。
当前没有在这份数据上训练模型，不提供准确率。项目的完整分割训练入口仍待开发。
'''
    (destination/'先看这里_标注与交接.txt').write_text(guide,encoding='utf-8-sig')
    (destination/'使用说明.txt').write_text(guide,encoding='utf-8-sig')
    for name in ['build_mask_review_kit.py']:
        shutil.copy2(Path(__file__).with_name(name),destination/'source'/name)
    # Local export fixture mirrors the browser contract; the real browser export is verified separately.
    sample=destination/'export_example';sample.mkdir()
    for sub in ['images','depth','masks','previews','annotations']:(sample/sub).mkdir()
    instances=[]
    for row in draft['objects']:
        frame=row['frame_id'];pred=next(i for i in result['instances'] if i['source_object_id']==row['object_id'])
        rgb_path=f'images/{frame}_color.jpg';depth_path=f'depth/{frame}_depth.png';mask_path=f'masks/{frame}/{row["object_id"]}.png'
        for p in [rgb_path,depth_path]:
            if not (sample/p).exists():shutil.copy2(destination/'data/raw'/Path(p).name,sample/p)
        (sample/mask_path).parent.mkdir(exist_ok=True);shutil.copy2(result_folder/pred['mask_ref'],sample/mask_path)
        instances.append({**row,'rgb_path':rgb_path,'depth_path':depth_path,'mask_path':mask_path,'training_ready':False,'blocking_reasons':['mask_unreviewed','frame_completeness_unconfirmed','sku_missing','sequence_missing','split_not_assigned']})
    shutil.copy2(result_folder/'overlay.jpg',sample/'previews'/f'{result["frame_id"]}.jpg')
    (sample/'annotations.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2),encoding='utf-8')
    (sample/'instances.json').write_text(json.dumps({'schema_version':'review-package-1','training_ready':False,'instances':instances},ensure_ascii=False,indent=2),encoding='utf-8')
    (sample/'annotations/instances_reviewed.json').write_text(json.dumps({'info':{'description':'No human-reviewed complete frame yet'},'images':[],'annotations':[],'categories':[{'id':1,'name':'object','supercategory':'object'}]},indent=2),encoding='utf-8')
    (sample/'quality_report.json').write_text(json.dumps({'frames':1,'objects':27,'reviewed_complete_frames':0,'reviewed_instances':0,'excluded_from_coco':[{'frame_id':result['frame_id'],'reason':'Mask及整帧未复核'}],'split_assigned':False},ensure_ascii=False,indent=2),encoding='utf-8')
    (sample/'README.txt').write_text('导出结构示例：1张原图＋1张原始深度＋27张SAM轮廓草稿。全部待复核，正式COCO标注为空是预期行为。\n'+guide,encoding='utf-8-sig')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations',type=Path,required=True);ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    build(Path(__file__).resolve().parents[1],args.output,lambda dest:supplement(dest,args.annotations,args.result))
