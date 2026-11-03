最新版本 **v0.5（2026-10-26）**：应用层与真实模型链路已合并，启动和验证见 [合并报告](../docs/app_pipeline_merge_20261026.md)。下文较早的阶段记录保留作历史参考。

# AIE3905 模块化感知链路 v0.4

已完成 RGB + Prompt → Grounding DINO → SAM 2.1 → ResNet-18 材质分类 → 几何统计接口 → JSON。支持单图、CSV 批量、模型复用、逐帧错误记录和结果可追溯。真实深度需确认配准和单位后接入。

## 每个板块与模型的位置

| 目录 | 职责 |
| --- | --- |
| perception/models/grounding_dino | 检测模型加载、预处理、推理与后处理 |
| perception/models/sam2 | SAM 2.1 权重加载和实例 Mask |
| perception/models/resnet18 | 材质网络、标签、预处理和推理 |
| perception/models/sam1 | 可选 SAM 1 对照适配器 |
| perception/models/yoloe | 可选 YOLOE 对照模型 |
| perception/pipeline | 模型编排、单帧服务、批处理入口、旧接口适配 |
| perception/geometry | 框合并、像素面积、有效深度统计 |
| perception/attributes | 物理属性接口与未验证状态 |
| perception/io | CSV 读取、JSON 与 Mask 保存、叠加图 |
| configs | 参数与路径配置；本机配置为 *.local.json，不进入源码包 |
| material_recognition | 数据准备、训练与评估；模型实现从独立目录导入 |
| scripts | 数据检查、预标注、人工审核、旧版命令兼容入口 |
| tests | 数据、训练、输入校验和端到端编排回归 |

统一链路不再依赖 week1_model_selection 的实现。该目录的旧入口转发到新模块。SAM1/YOLOE 已独立存放，未在本轮运行模型对比。

## 本机直接运行

在项目根目录执行，使用已建立的项目环境，避免系统默认 Python 3.14：

```powershell
.\.venv\Scripts\python.exe -m perception.doctor --config configs/verification.local.json
./run_perception.ps1 --config configs/verification.local.json --manifest configs/verification.local.csv --output output/my_next_run
```

每次指定新的输出目录。verification.local.csv 指向本机已有的四张图片。当前模型已缓存；可设置 `$env:HF_HUB_OFFLINE='1'` 在断网条件下运行。

该 .venv 复用本机 Conda GPU 环境的 PyTorch 等基础库，项目新增依赖安装在 .venv。核心推理已实测。基础环境中的非项目包 hf-gradio 与项目 Gradio 客户端存在依赖声明冲突，因此此环境不等同于完全隔离且全量 pip check 通过的环境；新机器请按下节建立纯净环境。

## 新机器安装

使用 Python 3.12。下面是 CPU 安装命令；GPU 机器需选择与设备兼容的 PyTorch 构建。本次实测为 PyTorch 2.9.1+cu128 / TorchVision 0.24.1+cu128。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-perception.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

训练和既有数据工具另需根目录 requirements.txt。若安装 Gradio，保持 requirements-perception.txt 中的兼容版本。源码包不含企业图像、模型权重或本机环境。

复制 configs/pipeline.example.json 为自己的配置，修改 SAM 和材质权重路径。配置文件中的相对权重路径以配置目录为基准；命令行覆盖路径以工作目录为基准。DINO 首次运行会下载固定版本，也可将 dino_model 指向本地完整模型目录。权重版本与校验方法参见 week1_model_selection/MODEL_MANIFEST.md。材质模型必须是本项目五类 checkpoint。

```powershell
.\.venv\Scripts\python.exe -m perception.pipeline.run --config configs/my.json --rgb path/to/rgb.jpg --prompt 'box . bag . bottle .' --output output/single_run
.\.venv\Scripts\python.exe -m perception.pipeline.run --config configs/my.json --manifest examples/frames.csv --output output/batch_run
```

安装后也可使用 `.venv/Scripts/aie-perception.exe`、`aie-doctor.exe`。库调用使用 `PerceptionPipeline(config).process(rgb_path, prompt, output_dir, depth_path=None)`，模型只加载一次。

CSV 必需 frame_id、rgb_path；可选 depth_path、prompt。CSV 内相对图片路径以 CSV 所在目录为基准。批处理遇到单帧错误继续处理其余帧，并在 batch_summary.json 中记录错误；任何帧失败则退出码为 1。重复/不安全的 frame_id 会在运行前拒绝。

## 配置与输出

- device：cpu/cuda/auto；检测、分割、材质统一遵循所选设备。
- mode：full/tiled；使用固定分块可检验小目标漏检问题。
- crop_mode：默认 bbox，与已有材质训练方式一致；masked 是灰底 Mask 裁剪对照，需要重新评价。
- box_threshold/text_threshold/nms_threshold：检测与框去重阈值。通过配置修改，测试集不用于调参。
- aligned_depth：只有确认深度已配准到 RGB 后才能设为 true。
- depth_scale：原始深度乘以该值转换为米；只有已确认毫米单位时才用 0.001。

每帧输出 pipeline-1.0 的 preprocess.json、segmentation.json、material_input.json、material_result.json、最终 result.json，以及 report.md、masks、rois、RGB 副本和 overlay.jpg。旧聚合格式保留为 perception.json；原来 result.json 的分割原始内容改存 segmentation_raw.json。批量额外输出 batch_summary.json。JSON 记录输入来源、参数、模型权重哈希、版本、设备和耗时。model_load_ms 是一次初始化耗时，frame_processing_ms 不含模型加载及新增契约文件导出，不能作为新版本完整端到端耗时；material_and_geometry_ms 包含材质与几何统计。

默认 depth_processing=reference，仅复制深度文件和记录路径、尺寸、dtype、单位与配准状态，不计算深度像素统计。未知配准不会阻断 RGB 链路。旧统计能力需显式设置 depth_processing=statistics 且 aligned_depth=true，此时排除 0、NaN、Inf；没有单位时不输出米制距离。统计写在兼容 perception.json 中，正式契约仍不宣称深度质量验收。当前无等高线、三维尺寸、几何类别、质量或摩擦模型。材质 score 是未经校准的分类分数；拒识策略未验收，因此正式契约保留 top1_label/scores，label=unknown、status=unknown，帧状态 partial。旧 perception.json 中的 label 仍为原始模型候选。

## 接入团队分割包（2026-10-26）

```powershell
.\.venv\Scripts\python.exe -m perception.pipeline.run --config configs/my.json --segmentation path/to/frame/segmentation.json --rgb path/to/original_rgb.jpg --depth path/to/depth.png --output output/resumed_frame
```

--depth 可省略；不传时保留原包深度引用，并检查文件是否存在。必须提供同帧原始 RGB，不能使用带有叠加颜色的 overlay.jpg 进行材质推理。输入包缺少原始 RGB 哈希，所以只自动验证尺寸，帧身份由调用者确认。导入不加载 DINO/SAM，仅需有效 material_weight；配置中的 sam_weight 可保留为占位路径。当前 --segmentation 仅支持单帧，与 --manifest 互斥。

入口严格检查版本、阶段、实例数、唯一 ID、0/255 单通道原图尺寸 mask、非空前景、mask 重算的整数半开框、分数和安全相对路径。导入后保留上游 run_id/frame_id/instance_id。原生推理也统一用 mask 的整数框做材质裁剪；这会相对旧浮点检测框改变裁剪范围，需要重新评价材质效果。当前区域是 instance_bbox_fallback，未实现自动接触面选择。单实例材质运行错误写入 errors，其余实例继续。

比较已保存结果可运行 `python scripts/compare_saved_segmentations.py --left new/segmentation.json --right old/perception.json --output comparison.json`。该工具按 mask IoU 降序贪心一对一匹配，不依赖实例序号；仅评估预测一致性，不是人工真值精度。具体合并结论见 [合并报告](../docs/segmentation_merge_20261026.md)。

## 数据训练闭环

沿用 scripts/build_training_dataset.py 生成 needs_review 候选 → scripts/serve_annotation_review.py 人工审核 → scripts/approve_training_dataset.py 导出已确认训练清单 → material_recognition.train 训练 → material_recognition.evaluate 评估。具体参数使用各命令 --help，既有流程见 docs/OPERATIONS.md。自动预测不直接变为训练真值，SKU 未确认时不声称跨 SKU 评估。

## 验证和维护

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = (Get-Location).Path + ';' + (Join-Path (Get-Location) 'week1_model_selection')
.\.venv\Scripts\python.exe -m unittest discover -s week1_model_selection/tests -v
```

本轮实测记录见 docs/modular_validation_20261023.md。更新模型时优先修改对应 models 子目录，并同步配置、预处理和测试。通过统一接口运行可避免不同入口产生不同设备选择和预处理行为。


## 材质模型 v2 训练更新

新增按帧交叉验证和公开样本回放训练，说明见 [训练调优报告](../docs/material_tuning_20261023.md)。新权重使用 configs/material_v2.local.json 接入，原配置保留作对照。优先人工复核清单位于 output/priority_review_20261023/index.html。
