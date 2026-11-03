"""Build a local stakeholder review page from existing measured artifacts."""
import argparse
import html
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'segmentation').mkdir()
    comparison=ROOT/'../06-实验结果/测试输出/segmentation_optimization_20261003/paired_comparison'
    material=ROOT/'../06-实验结果/测试输出/2026-09-29_rgbd_extended_test'
    records=json.loads((material/'predictions.json').read_text(encoding='utf-8'))
    summary=json.loads((material/'summary.json').read_text(encoding='utf-8'))
    pairs=json.loads((comparison/'comparison.json').read_text(encoding='utf-8'))
    for folder in comparison.glob('frame_*'):
        if folder.is_dir():shutil.copytree(folder,out/'segmentation'/folder.name)
        elif folder.suffix=='.jpg':shutil.copy2(folder,out/'segmentation'/folder.name)
    for name in ['index.html','comparison.json','experiment.json','优化结论.md']:
        shutil.copy2(comparison/name,out/'segmentation'/name)
    for name in ['predictions.json','summary.json']:
        (out/'material').mkdir(exist_ok=True);shutil.copy2(material/name,out/'material'/name)
    shutil.copytree(material/'crops',out/'material/crops')
    (out/'assets').mkdir();(out/'documents').mkdir()
    inspect=ROOT/'../04-数据/processed/yingjie_rgbd_inspection'
    shutil.copy2(inspect/'rgb_depth_examples.jpg',out/'assets/rgb_depth_examples.jpg')
    specification=ROOT/'../01-项目技术规划书/02-技术规范与四人分工_20261008/RGBD物体感知项目技术规范书_四人分工讨论版_20261008.docx'
    shutil.copy2(specification,out/'documents/技术规范书与四人分工.docx')
    names={'paper':'纸质','plastic':'塑料','metal':'金属','wood':'木材','fabric':'织物'}
    material_cards=[]
    for r in records:
        truth=names.get(r.get('true_material'),'待人工复核')
        status='已复核一致' if r.get('correct') else '已复核不一致' if r.get('true_material') else '无真值'
        material_cards.append(f'<article class="sample"><img loading="lazy" src="material/{html.escape(r["crop_file"])}" alt="{html.escape(r["object_id"])}商品裁图"><div><b>{html.escape(r["object_id"])}</b><p>预测：{names[r["predicted_material"]]} · 分数 {r["score"]:.3f}</p><p>人工标签：{truth}</p><small>{status}</small></div></article>')
    pair_data={str(i):{v:next(r for r in pairs if r['frame_index']==i and r['variant']==v) for v in ['full','crop']} for i in [23,45,89,156,245]}
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AIE3905 项目阶段展示</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#142338;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:32px 24px 70px}header{padding:8px 0 26px;border-bottom:1px solid #ccd6e0}h1{font-size:34px;line-height:1.25;margin:10px 0}h2{font-size:24px;margin:0 0 12px}h3{font-size:18px}p{margin:8px 0}.eyebrow{color:#526779;font-size:13px;letter-spacing:1px}.muted,small{color:#54677d}.actions{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}a{color:#155eb3}a.button{background:#185cb4;color:white;text-decoration:none;border-radius:8px;padding:10px 18px}a.secondary{background:white;color:#185cb4;border:1px solid #b7c8da}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:24px 0}.stat,section{background:white;border:1px solid #dce2e8;border-radius:12px;padding:24px}.stat b{display:block;font-size:28px}.stat span{font-size:14px;color:#54677d}section{margin-top:22px}section img{max-width:100%;height:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}select{font:inherit;padding:7px;border-radius:6px;border:1px solid #bac7d4}.note{color:#795021;background:#fff8e8;padding:12px 16px;border-radius:6px;margin:16px 0}.samples{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.sample{display:flex;gap:12px;padding:14px;border:1px solid #dce2e8;border-radius:8px;font-size:13px;align-items:flex-start}.sample img{width:76px;height:90px;object-fit:contain;background:#f5f5f5}.sample p{margin:3px 0}details{margin-top:18px}summary{cursor:pointer}li{margin:7px 0}table{width:100%;border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #dce2e8;text-align:left}th{background:#eef3f8}@media(max-width:760px){main{padding:20px 14px}.stats,.pair,.samples{grid-template-columns:1fr}h1{font-size:27px}}
</style><main><header><div class="eyebrow">MIRRORSPACE × AIE3905 · 组内与合作方进度同步</div><h1>从产线图像到可复核的物体标注</h1><p>已完成数据接入、自动分割试跑与材质基线。当前重点：补齐可靠标注，验证小物体分割与现场材质泛化。</p><p class="muted">整理日期 2026年10月9日 · 展示结果来自9月29日与10月3日已有实验。本页为结果回放，未在此页面实时运行模型。</p><div class="actions"><a class="button" href="http://127.0.0.1:7862/annotate.html" target="_blank" rel="noopener">打开数据标注 Demo</a><a class="button secondary" href="segmentation/index.html" target="_blank" rel="noopener">完整五帧分割对比</a><a class="button secondary" href="documents/技术规范书与四人分工.docx">下载技术规范与分工</a></div></header>
<div class="stats"><div class="stat"><b>245 对</b><span>企业 RGB 与 Depth 完成配对和读取检查</span></div><div class="stat"><b>5 帧 · 45 个</b><span>裁剪检测后的待复核候选，不是正确物体计数</span></div><div class="stat"><b>6 帧 · 24 个 ROI</b><span>已有材质试跑，尚无完整现场测试真值</span></div></div>
<section><h2>01 数据与任务</h2><p>输入 RGB-D 与目标描述，期望输出实例 Mask、几何类别、材质及物理属性，再汇总为 JSON。当前几何、物理属性和完整端到端接口仍待实现。</p><details><summary>查看产线 RGB 与 Depth 示例</summary><img loading="lazy" src="assets/rgb_depth_examples.jpg" alt="产线RGB及深度可视化样例"></details><p class="muted">Depth为16位图像，单位与配准尚未确认。目前分割和材质推理使用RGB，深度图只作检查与展示。</p></section>
<section><h2>02 自动定位与物体轮廓</h2><p>Grounding DINO Tiny 生成候选框，SAM ViT-B 生成可见区域 Mask。比较时仅改变检测输入是否裁剪至工作区域，SAM保持原图输入。</p><label>选择场景 <select id="frame"><option value="23">第23帧 · 分散摆放</option><option value="45">第45帧 · 手部与遮挡</option><option value="89">第89帧 · 稀疏场景</option><option value="156">第156帧 · 密集场景</option><option value="245">第245帧 · 多物体场景</option></select></label><div class="pair"><article><h3 id="fullLabel"></h3><a id="fullLink"><img id="fullImage" alt="整图自动分割结果"></a><p id="fullStats"></p></article><article><h3 id="cropLabel"></h3><a id="cropLink"><img id="cropImage" alt="工作区域裁剪后的自动分割结果"></a><p id="cropStats"></p></article></div><p class="note">20个历史参考框的匹配数由7增至15，仅为局部诊断，不是整体准确率。仍存在漏检、误检和分割粒度问题；这5帧已用于调试，不能作为独立最终测试集。</p></section>
<section><h2>03 人工复核与训练数据</h2><ol><li>进入标注 Demo，填写标注人，跳到第23、45、89、156或245张。</li><li>载入本图自动候选；删除误检、补漏检，用画笔与橡皮修正物体轮廓。</li><li>选择外包装材质，分别确认轮廓与材质，再导出JSON。</li></ol><p>同一份复核数据可以服务分割和材质两个模块。转换器排除未确认标签，并检查SKU与关联序列跨训练／验证／测试集合的泄漏。</p><p class="muted">标注页载入预计算候选，不会点击后重新推理；当前未用这45个自动候选继续训练。分割训练器、几何与物理属性标注仍待补充。</p><a href="http://127.0.0.1:7862/annotate.html" target="_blank" rel="noopener">进入标注页面 →</a></section>
<section><h2>04 材质分类基线与问题</h2><p>现有 MINC ResNet-18 对人工选取的RGB区域进行五分类。下面的分数是未经校准的模型分数，不是材质真实概率。</p><p class="note">24个ROI中只有4个经用户确认，其中1个预测正确；其余20个缺少真值。不能计算整批现场准确率，现有模型需要现场数据适配。</p><div class="samples">FIRST_CARDS</div><details><summary>展开其余20个待复核预测</summary><div class="samples">OTHER_CARDS</div></details><p><a href="material/predictions.json">查看原始材质预测JSON</a></p></section>
<section><h2>05 下一阶段与协作</h2><table><tr><th>职责</th><th>近期交付</th></tr><tr><td>数据处理</td><td>首批10帧复核、标签口径、SKU／序列整理与划分</td></tr><tr><td>检测与分割</td><td>网页实时调用、小物体失败分析和分割评估</td></tr><tr><td>材质与物理属性</td><td>现场基线与微调对照，明确物理属性真值来源</td></tr><tr><td>几何与系统集成</td><td>几何基线、统一schema与端到端接口</td></tr></table><p>希望合作方补充：深度单位与配准说明、相机内参、SKU及外包装材质资料，以及硬度／可变形性的判断依据。</p><p class="muted">第一阶段交付目标：System Design v1＋Annotation Schema＋Baseline Plan。当前展示不代表最终识别性能验收。</p></section>
<p class="muted">本页运行于当前电脑。用于现场展示或屏幕共享；本地地址无法直接供对方远程访问。</p></main><script>const pairs=PAIR_DATA;function show(){const n=document.getElementById('frame').value;for(const v of ['full','crop']){const r=pairs[n][v],dir='segmentation/frame_'+n.padStart(3,'0')+'_'+v;document.getElementById(v+'Image').src=dir+'/overlay.jpg';document.getElementById(v+'Link').href=dir+'/index.html';document.getElementById(v+'Label').textContent=(v==='full'?'整图检测':'工作区域裁剪')+' · '+r.candidates+'个候选';document.getElementById(v+'Stats').textContent='历史参考框匹配 '+r.reference_matches+'/4（非整体召回率）';}}document.getElementById('frame').onchange=show;show();</script></html>'''
    page=page.replace('FIRST_CARDS',''.join(material_cards[:4])).replace('OTHER_CARDS',''.join(material_cards[4:])).replace('PAIR_DATA',json.dumps(pair_data))
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'README.txt').write_text('阶段展示入口：index.html。可直接打开静态结果；标注链接需要先启动本机7862服务。\n展示包含企业现场数据，仅用于项目合作范围。\n建议顺序：数据示例、五帧分割对比、标注复核、材质错误样例、下一阶段分工。\n结果为已有实验回放，不是本次实时推理。\n',encoding='utf-8-sig')
    print(json.dumps({'output':str(out),'segmentation_frames':5,'material_rois':len(records),'reviewed_material_rois':summary['reviewed_count']},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    build(ap.parse_args().output)
