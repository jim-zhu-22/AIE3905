# 谢嘉翔：独立检测与分割实践

本目录是 2026-10-18 的第一周模型选型独立交付，与项目根目录原有的 `scripts/run_segmentation*.py` 分开。公开代码包保留推理、网页接口、评估和预标注源码，以及不含企业图像的[实验汇总](evaluation_summary_20261018.json)；选型结论见上一级的[谢嘉翔实测报告](../RGBD小物体实例分割模型选型实测报告_谢嘉翔.md)。原始企业 RGB-D 数据、标注工具和运行生成的 `results/` 不在公开包内。

这套代码独立于同学的 `scripts/run_segmentation*.py`。同一 Grounding DINO Tiny + SAM 2.1 Tiny 权重比较整图检测和固定 2×2 裁剪检测。两种模式都在原图上调用一次 SAM，输出原图尺寸的 Mask，因此 A/B 的主要变量是检测输入的尺度。当前两模型均使用 RGB；Depth 不参与本次预训练模型推理。

目录：`segmentation_practice/` 为模型代码，`scripts/` 为批量实验、网页接口、预标注与评估入口，`tests/` 为测试。[模型清单](MODEL_MANIFEST.md)记录检测器版本和分割器权重校验值。执行后会重新生成被 Git 忽略的 `results/`。

## 环境

以下命令从项目根目录执行。先进入本目录，再安装依赖：

```powershell
Set-Location .\week1_model_selection
..\.venv\Scripts\python.exe -m pip install torch torchvision
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本机以上述命令安装得到 PyTorch 2.14.1+cpu，用于流程验证。目标机器如需 CUDA，请按 [PyTorch 官方安装页](https://docs.pytorch.org/get-started/locally/)选择与设备相符的命令，在独立环境安装。不要把本机 GTX 1050 Ti 4GB 的速度当作目标设备速度。

首次准备模型时按[模型清单](MODEL_MANIFEST.md)取得固定版本的 `IDEA-Research/grounding-dino-tiny` 和 `sam2.1_t.pt`。SAM 权重放在上一级项目目录，或用 `--sam-weight` 指定本地路径。SAM 2.1 Tiny 交付权重随仓库根目录的 Git LFS 管理；DINO 权重仍按模型清单下载并核对哈希。网络受限且缓存完整时，在运行前设置 `$env:HF_HUB_OFFLINE='1'`。

## 批量实验

```powershell
# 先用 1 帧整图试通模型
..\.venv\Scripts\python.exe scripts\run_own_segmentation.py --max-frames 1 --mode full

# 同一模型依次跑 4 帧待复核框 + 工具包中的第 5 帧（第 5 帧尚无标注）
..\.venv\Scripts\python.exe scripts\run_own_segmentation.py --mode both
```

输出在 `results/own_segmentation/`：`summary.json`、每帧 `result.json`、`overlay.jpg`、`diagnostics.jpg` 与 `masks/*.png`。`summary.json` 的检测数字明确标记为探索性框评估：现有 108 框都为 `needs_review`，整帧完整性未确认，且没有人工 Mask。第 5 帧只输出预测与可视化，不计入指标。

固定裁剪规则：1280×960 图像切成四张 704×528，横向起点 0、576，纵向起点 0、432；每块运行 DINO，框映射至原图并按 box IoU 0.5 去重；统一在原图上运行 SAM。`source_tiles` 保留每个结果的切片来源。阈值、Prompt、模型和 SAM 输入在两模式一致。

## 网页与 segment 接口

```powershell
..\.venv\Scripts\python.exe scripts\segmentation_web.py
```

浏览器打开 `http://127.0.0.1:7860/`。上传 RGB 图，输入 Prompt，选择整图或裁剪，点击运行后获取叠加图、每实例 Mask、耗时与 JSON。网页请求 `POST /v1/segment`，请求格式为 `{"image_base64":"<base64 或 data URL>","prompt":"box. bag.","mode":"full"}`；响应包含原图坐标 `bbox_xyxy`、`det_score`、`mask_ref`、`overlay_ref`、`latency_ms` 和模型标识。服务默认只监听本机地址。

可用 `..\.venv\Scripts\python.exe scripts\check_segmentation_web.py` 发送一张真实图像并检查 HTTP 响应、叠加图和首个 Mask。已在本机验证成功。

模型跑过第 5 帧后，可生成现有离线标注工具能导入的框草稿：

```powershell
..\.venv\Scripts\python.exe scripts\export_segmentation_preannotations.py
```

文件为 `results/own_segmentation/preannotations.json`，所有记录仍是 `needs_review`，材质留空。**标注工具导入会替换当前浏览器草稿**，导入前先导出并备份原草稿。工具只接受框，模型 Mask PNG 保留在实验输出目录供单独人工修边与复核。

可用 `node scripts/validate_preannotations.js` 调用标注工具自己的导入校验函数，确认草稿格式能被接受。

## 汇报口径

本轮可以报告运行可用性、两策略在待复核框上的**探索性** TP/FN/FP、逐帧耗时及实例可视化。正式的漏检、误检、Mask IoU/Dice 必须等整帧标全且实例 Mask 经人工复核后计算。同学报告中 SAM 1 与 SAM 2.1 的 Mask IoU 0.948 是模型间一致性，不是人工真值精度。

取得符合项目标注规范的人工 Mask 后，将每帧 JSON 加入 `frame_annotation_complete: true`，并将该帧所有实例复核为 `confirmed`，再运行：

```powershell
..\.venv\Scripts\python.exe scripts\evaluate_confirmed_masks.py --annotations-dir path\to\confirmed_frames --mode full
..\.venv\Scripts\python.exe scripts\evaluate_confirmed_masks.py --annotations-dir path\to\confirmed_frames --mode tiled
```

评估脚本会跳过未完成或仍需复核的帧，计算一对一 Mask IoU≥0.5 的 TP/FN/FP、召回、精确率以及匹配实例平均 IoU/Dice。

若要修正模型轮廓，可直接在浏览器打开 `scripts/mask_review.html`，载入原始 RGB 和某个 `masks/iNNN.png`，用添加/擦除画笔修边，再下载原图尺寸 0/255 PNG。下载的文件仍需人为确认对应实例、保存到真值目录并由第二人复核；模型草稿不会自动变成真值。
