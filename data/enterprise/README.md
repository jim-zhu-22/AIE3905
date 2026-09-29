# 企业数据接入

本目录保留接入说明与空清单模板。原始图像、已填标注、训练结果和模型权重仅在项目内部共享，不提交到公开仓库。

## 当前数据与工具

已收到的产线数据默认放在 data/映界_rgbd数据/，共245对RGB与16位Depth。数据检查与浏览操作见 [操作 README](../../docs/OPERATIONS.md) 第4至7节。

浏览器可以导出物体框、抓取接触面框、SKU和材质标注草稿。全部记录标记needs_review，尚不等于已复核训练样本。当前只有原有4个试跑区域经用户确认标签，不代表整批数据完成标注。

## 训练清单约定

空模板为本目录的manifest.template.csv，字段包括：

~~~text
sample_id,split,label,label_id,image_path,source,sku_id,batch_id,parent_image_id,roi_x1,roi_y1,roi_x2,roi_y2,annotation_status
~~~

- label对应paper、plastic、metal、wood、fabric，按实际接触外表面标注。
- split为train、validation或test；同SKU、同源帧及关联序列不得跨集合泄漏。
- annotation_status为confirmed表示完成复核，未确认的标注不进入训练。
- image_path相对清单所在目录解析；当前训练读取器读取整张所指图像，不会按ROI坐标自动裁剪，所以应先保存接触面裁剪图。
- 浏览器草稿尚未实现自动转换为训练清单。物体框与接触面框也不能代替实例Mask。

接触面材质边界见 [标注规范](../../docs/annotation_guideline.md)。完整Geometry与物理属性annotation schema仍属于Milestone 1待统一范围。深度单位、RGB与Depth对齐和相机参数尚未确认。
