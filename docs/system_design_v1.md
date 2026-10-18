# RGB-D 物体感知系统设计 v1（Milestone 1）

状态：讨论稿。范围以 [README](../README.md) 的八周计划为准；本版定义接口与实验，不表示端到端系统已经实现。

## 1. 目标与边界

输入一对同一时刻的 RGB、16 位 Depth 图像及目标 prompt，输出与 prompt 对应的可见物体实例 mask、几何类、可见接触表面材质、形变属性和结构化 JSON。Depth 用于分割与几何线索，不要求完整三维重建、PLY 或 mesh。一个 prompt 可匹配零个、一个或多个实例；零匹配必须有明确状态。

## 2. 输入契约

| 字段 | 约定 | 当前状态 |
| --- | --- | --- |
| `frame_id` | 原始帧唯一 ID | 已有时间戳命名 |
| `rgb_path` | RGB JPG/PNG，原始像素坐标系 | 已有 JPG |
| `depth_path` | 原始 16 位单通道 PNG，禁止可视化图代替 | 已有 PNG |
| `prompt` | 描述目标的文本，例如“左侧红色瓶子” | 尚无现场标注 |
| `depth_unit`、`invalid_depth_values` | 深度物理单位与无效编码 | 待数据方确认；当前只能处理原始数值 |
| `alignment`、`intrinsics` | Depth 到 RGB 的配准关系及相机内参 | 待确认；同尺寸不足以证明对齐 |

统一坐标为原始 RGB 图像像素，`bbox_xyxy` 为左上包含、右下不包含。若对齐未确认，禁止把 Depth 像素与 RGB mask 逐点对应作物理尺寸或三维形状结论；可先完成 RGB-only 路线，并单独记录 Depth 探索结果。

## 3. 模块与数据流

```text
RGB + prompt ──→ 目标实例分割 ──→ mask、bbox、匹配分数
Depth + 配准信息 ──────────────→ mask 内深度质量检查与几何特征
RGB + mask + 可用深度 ────────→ geometry 标签
RGB + 接触面 ROI ─────────────→ material 标签
geometry + material + 图像证据 → deformability 标签或 uncertain
各模块结果 + 状态 ────────────→ JSON
```

| 模块 | 最小输入 | 输出 | 失败处理 |
| --- | --- | --- | --- |
| 目标实例分割 | RGB、prompt | 一个或多个二值 mask、bbox、匹配分数 | 零匹配返回 `no_match`；低分实例送复核 |
| 深度质量与特征 | Depth、mask、配准信息 | 有效像素占比、相对深度起伏等 | 配准/单位不明时标记 `depth_unavailable`，不用物理尺寸 |
| 几何识别 | mask、RGB、可信深度特征 | 六类之一或 `unknown` | 遮挡、深度缺失时允许拒判 |
| 材质识别 | 经复核的可见接触面 ROI | 五类之一或 `unknown` | 复合层、反光、证据不足时待复核 |
| 形变属性 | 可见外观、几何与人工证据 | `rigid` / `deformable` / `uncertain` | 静态 RGB-D 不足以证实硬度时用 `uncertain` |
| 结果组装 | 上述输出 | 每帧 JSON 与 mask 文件 | 保留模块状态、版本和未判定原因 |

分割 mask 指**可见区域**，不补全被遮挡部分。输出 JSON 不嵌入大幅像素矩阵，以相对路径引用无损二值 PNG；mask 必须与 RGB 同宽高、前景为 255、背景为 0。可选的 `contact_bbox_xyxy` 只界定用于材质判断的可见表面，不能代替 mask。

## 4. 标签语义（待联合确认）

几何标签：`cuboid`、`cylinder`、`bottle`、`vacuum_bag`、`puffed_bag`、`irregular`；`unknown` 是证据不足的拒判状态，和已确认但不属于前五类的 `irregular` 不同。`bottle` 优先于近似圆柱；有明确抽真空贴合轮廓的袋子标 `vacuum_bag`，有明显充气鼓起的袋子标 `puffed_bag`。仅凭单视角不能确认袋内状态时用 `unknown`。瓶体、纸盒等外形类与材质类独立标注。

材质沿用现有 `paper`、`plastic`、`metal`、`wood`、`fabric`，指拟接触的**可见最外层**。`unknown` 是拒识/待核实，不作为既有五类模型已学会的第六类。`hardness` 暂不输出数值：无接触、受力或可信产品信息时，单帧不能测得硬度；第一版只输出可观察或经人工确认的形变等级 `deformability`。若项目必须使用 `hardness` 字段，应先共同定义测量条件与等级。

## 5. 推理接口草案

请求包含 `frame_id`、RGB/Depth 文件或路径、`prompt` 与可选相机元数据。响应至少包含 `schema_version`、`frame_id`、`prompt`、`status`、`instances`。每个实例包含 `instance_id`、`mask_path`、`bbox_xyxy`、`geometry`、`material`、`deformability` 及各模块状态。示例见 [annotation_schema_v1.md](annotation_schema_v1.md)。缺失输入、尺寸不一致、mask 越界、无法解码等返回明确错误，不生成貌似成功的预测。

## 6. 当前事实与依赖

- 已有：245 对企业 RGB-D 的历史可读/配对检查记录；RGB-D 浏览与框选草稿；五类 RGB 材质 ResNet-18 基线。公开数据验证 Accuracy 75.20%、Macro-F1 0.7480，属于参与选模的验证集结果。
- 尚缺：原始企业数据在当前仓库副本中的可访问性、已复核实例 mask、SKU/场景 prompt、深度单位、相机内参与配准、几何和形变真值。
- 深度增强是否有效需在同一现场测试集做 RGB-only 与 RGB-D 对照；不能因增加 Depth 输入就预设提升。

## 7. Milestone 1 退出条件

团队共同确认 taxonomy 与疑难样例；完成少量代表性场景的试标和双人复核；确认或正式记录相机元数据缺口；冻结 v1 标注协议、按 SKU/近重复帧隔离的划分方案及评估脚本输入格式。以上完成前不宣称端到端效果。
