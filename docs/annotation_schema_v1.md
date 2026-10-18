# RGB-D 实例标注 Schema v1（讨论稿）

本格式供整帧、多实例标注与后续端到端评估使用。它与当前 `data/enterprise/manifest.template.csv`（单个材质 ROI 的训练清单）及浏览器导出的框选草稿**用途不同**；两者不能直接视为已复核的实例标注。

## 记录结构

每帧一个 JSON 文件，路径以该 JSON 文件所在目录为基准解析。原始图像、真实标注和模型权重不提交公开仓库。下面是**格式示例，非真实标注**：

```json
{
  "schema_version": "1.0",
  "frame_id": "example_frame_001",
  "rgb_path": "rgb/example_frame_001_color.jpg",
  "depth_path": "depth/example_frame_001_depth.png",
  "image_size": [640, 480],
  "depth_unit": "unconfirmed",
  "invalid_depth_values": [0],
  "alignment": "unconfirmed",
  "intrinsics": null,
  "batch_id": "example_batch",
  "split": "unassigned",
  "targets": [
    {
      "prompt_id": "prompt_001",
      "prompt": "左侧瓶子",
      "instance_ids": ["object_001"]
    }
  ],
  "instances": [
    {
      "instance_id": "object_001",
      "sku_id": null,
      "mask_path": "masks/example_frame_001_object_001.png",
      "bbox_xyxy": [120, 45, 310, 390],
      "contact_bbox_xyxy": [150, 80, 260, 230],
      "geometry": "bottle",
      "material": "plastic",
      "deformability": "uncertain",
      "annotation_status": "needs_review",
      "reviewer_id": null,
      "notes": "示例值；不得作为训练或测试真值"
    }
  ]
}
```

## 字段与规则

| 字段 | 规则 |
| --- | --- |
| `schema_version` | 必填字符串，当前 `1.0`；修改标签语义时升级版本 |
| `frame_id`、`rgb_path`、`depth_path`、`image_size` | 必填；`image_size` 为 `[width,height]`，两图尺寸不同时记录实际 Depth 尺寸并先配准 |
| `depth_unit` | `unconfirmed`、`mm`、`m` 等；不得从灰度图或数值范围猜测 |
| `invalid_depth_values` | 相机文档或实际检查确认的无效编码；初期记录 0，其他编码待核实 |
| `alignment` | `unconfirmed`、`aligned_to_rgb` 或 `unaligned`；同尺寸不代表已配准 |
| `intrinsics` | 已确认时记录 `fx,fy,cx,cy` 与来源，否则为 `null` |
| `batch_id`、`sku_id` | 缺失用 `null`，不要编造；划分时同 SKU、同源帧、近重复序列不得跨集合 |
| `targets` | 每条 prompt 可关联零个或多个 `instance_id`；零个表示真值中无匹配目标 |
| `mask_path` | 必填，仅当已绘制 mask 时填写；8 位单通道 PNG，与 RGB 同尺寸，值仅 0/255 |
| `bbox_xyxy` | 原图整数像素，`[x1,y1,x2,y2]`，半开区间，须包围 mask 前景 |
| `contact_bbox_xyxy` | 可选，必须位于物体 bbox 内，表示材质观察区域，不代替实例 mask |
| `geometry` | `cuboid/cylinder/bottle/vacuum_bag/puffed_bag/irregular/unknown` |
| `material` | `paper/plastic/metal/wood/fabric/unknown`，按实际可见接触外层 |
| `deformability` | `rigid/deformable/uncertain`；没有可验证依据时选 `uncertain` |
| `annotation_status` | `needs_review`、`confirmed`、`rejected`；只有 `confirmed` 可用于真值评估 |
| `reviewer_id`、`notes` | 记录复核人及争议理由；个人信息可用内部代号 |

## 标注流程与质量门槛

1. 从审计结果挑选覆盖六类外形、遮挡、深度缺失、反光及复合包装的代表帧；记录选择依据，避免只选容易样本。
2. 先写 prompt 与目标实例对应关系，再画**可见区域** mask 和框；对多个同类物体使用可区分的 prompt 或记录多实例匹配。
3. 独立填写几何、接触面材质、形变等级，不能以商品名称推断材质或用材质直接推断硬度。
4. 第二人复核争议样本，保留修订前后记录。`needs_review` 与 `rejected` 不进入训练/测试真值。
5. 划分前检查同 SKU、同一帧派生区域与时间邻近帧；没有 SKU 时只能报告场景划分，不称为严格跨 SKU 测试。

当前 `scripts/rgbd_viewer.html` 只导出物体框和接触面框且统一标为 `needs_review`；若继续使用，需增加 mask 绘制与复核流程，或将导出作为预标注再用专门工具补画 mask。旧材质 ROI 清单仍可供已有 ResNet-18 训练使用，不能覆盖本 schema 的实例字段。

保存标注后，在仓库根目录运行 `python scripts/validate_rgbd_annotations.py path/to/frame.json`。校验器检查必填字段、枚举、prompt 引用、bbox 范围和 mask 像素/尺寸；仅检查格式时可加 `--skip-files`。格式通过不等于标签语义已经人工复核。
