# 企业数据接入

本目录用于企业现场数据的本地适配。原始图像、标注和训练结果不要直接提交到公开仓库。

目标是产生与公开数据一致的清单字段：

`sample_id,split,label,label_id,image_path,source,sku_id,batch_id,parent_image_id,roi_x1,roi_y1,roi_x2,roi_y2,annotation_status`

- `label` 对应 `paper`、`plastic`、`metal`、`wood`、`fabric`，标注实际接触表面。
- `split` 使用 `train`、`validation` 或 `test`；同一 SKU 只能出现在其中一个集合。
- 同一原图或连续帧派生的 ROI 要保持在同一集合；`parent_image_id` 供清单检查阻止同一原图跨集合泄漏。
- `annotation_status` 使用 `confirmed` 表示人工确认；其他状态暂不进入训练或正式评估。
- `image_path` 是相对于清单所在目录的路径。若保存的是已裁剪 ROI，坐标可留空。

现有标注工具的导出格式尚未提供。收到样例后再编写适配器并做标签质检。
具体边界案例见 `docs/annotation_guideline.md`。
