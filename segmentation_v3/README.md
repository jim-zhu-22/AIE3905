# segmentation_v3 · 实例分割后处理改造

本目录是「检测与分割」模块的独立实现，包含作业区约束、误检过滤、粘连分离三项改造，以及新一批数据的验证脚本与原始汇总。

## 与仓库 perception 包的关系

仓库现有的 `scripts/run_segmentation_sam2.py` 是 9 行兼容入口，实际逻辑在 `perception/pipeline/legacy.py`，对应第一轮选型阶段的原始形态（推理 + 叠加图 + `result.json`），不含本目录的后处理。

本目录不改动任何现有文件，可独立运行，两套并存不会互相影响。后续若要并入 `perception/pipeline/`，建议把 `filter` 与 `split` 两段抽出为 `perception/pipeline/postprocess.py`，由适配层调用，接口形态见文末。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run_segmentation_sam2.py` | 主脚本（v3.5）。Grounding DINO 出框 + SAM 2.1 出 mask，后接全部过滤与分离逻辑，输出 `segmentation.json` / `report.md` / `masks/` |
| `run_batch_compare.py` | 批量对照。同一批图跑多种预设并汇总为 `summary.md` / `summary.csv`。支持 `--limit` / `--stride` / `--pattern` 抽样，`--max-area-ratio` 覆盖面积阈值 |
| `inspect_dataset.py` | 数据集结构探查：目录树、文件数、扩展名、首帧尺寸与模式、RGB 与深度配对 |
| `check_filtered.py` | 汇总各帧被过滤的候选（原因 / 框 / mask 面积 / 标签），用于核查过滤是否误删 |

`run_batch_compare.py` 通过 `import run_segmentation_sam2` 调用主脚本，两者必须放在同一层目录。

## 环境

Python 3.13，依赖 `torch`、`numpy`、`Pillow`、`transformers`（Grounding DINO）、`ultralytics`（SAM 2.1）。纯 CPU 可运行。

权重放项目根 `models/`（已被 `.gitignore` 忽略，不入库）：

- `models/sam2.1_t.pt`
- Grounding DINO Tiny 走本地缓存。脚本默认置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，内网环境不触发联网检查。

## 用法

单帧（`tight` 为几何过滤 + NMS 的推荐组合，`--split-touching` 需显式开启）：

```powershell
python segmentation_v3\run_segmentation_sam2.py --image <RGB 图> --prompt "box. bag. bottle. carton. packet." ^
  --out-dir output\seg --preset tight --roi "450,0,900,690" --split-touching
```

批量抽样（按时间间隔取帧）：

```powershell
python segmentation_v3\run_batch_compare.py ^
  --images-dir <数据目录> --depth-dir <数据目录> ^
  --out-root output\new_smoke75 --presets roi_filter ^
  --roi "450,0,900,690" --max-area-ratio 0.75 --stride 10 --limit 33
```

`roi_filter` 预设默认值：`nms_iou=0.5`、`min_mask_area=100`、`max_area_ratio=0.35`、`max_aspect=8.0`、`split_touching=True`、`split_min_area=2000`。`--max-area-ratio` 只在本次运行内覆盖，不改预设文件。

查过滤原因：

```powershell
python segmentation_v3\check_filtered.py "output\new_smoke\*\roi_filter\segmentation.json"
```

## 本批数据的验证结果

数据 `映界_rgbd数据_1/tiangong-data-1`：331 帧，1280×960，RGB 与 `I;16` 深度同目录配对，采集时间 2026-08-27 18:07—18:52。按时间间隔抽取 33 帧做对照。

| 配置 | 实例总数 | 作业区外 | 拆开的粘连实例 |
| --- | --- | --- | --- |
| 整图基线 | 606 | 66（10.9%） | — |
| 输出端过滤，面积占比 0.35 | 523 | 0 | 4 |
| 输出端过滤，面积占比 0.75 | 527 | 0 | 8 |

原始逐帧汇总与过滤明细见 `results/`。

## 口径限制

- 33 帧为抽样，不代表 331 帧总体分布。全量在纯 CPU 上需两小时以上，尚未运行。
- 数据无人工标注真值。527 是模型输出的实例数，不等于「527 个商品」；拆分是否正确仅做过目视判断，无人工确认记录。
- 两轮总耗时（716.0s / 1321.0s）受本机负载波动影响，不作为性能结论：放开阈值仅使 2 个块额外进入分离流程，理论开销不足 1 秒，而低实例帧与零实例帧同样出现 1.5 至 2 倍放大。
- 深度未参与推理，仅作引用与可用性标记，本模块成果不能称为 RGB-D 融合。

## 待并入 perception 包的两个函数

`filter_instances(instances, roi, max_area_ratio, min_area, min_aspect)` → 返回保留实例与被过滤记录（`reason` / `bbox` / `mask_area`）。

`split_touching(mask, min_area)` → 返回拆出的 mask 列表，单物体原样返回一项。

两者均为纯 numpy 运算，不依赖模型权重，可直接由 `perception/pipeline/` 调用。
