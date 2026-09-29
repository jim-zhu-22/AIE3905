# 操作 README

本说明覆盖已经实现的训练、推理、RGB-D检查、浏览框选与人工ROI试跑。完整的 Prompt 分割、几何与物理属性识别仍未实现。

## 1 环境与工作目录

所有命令使用 Windows PowerShell，并在仓库根目录执行。目录可以放在任意磁盘；当前项目机器位于 E:/AIE3905。

### 新机器建立 CPU 环境

先安装 Python 3.11 或3.12。以下以3.12为例，选择单独的环境名称，避免使用已有的失效环境：

~~~powershell
py -3.12 -m venv .venv
$Python = (Resolve-Path .venv/Scripts/python.exe).Path
& $Python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
& $Python -m pip install -r requirements.txt
& $Python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
~~~

CPU环境的 CUDA 为 False 是正常结果。PyTorch单独安装，requirements.txt包含其他运行依赖。

### 当前项目机器使用已有环境

旧 .venv 的 Python 3.11 基础路径已经失效。本机可使用现有 Conda 环境；其他成员不要直接复制本机绝对路径：

~~~powershell
Set-Location E:/AIE3905
$Python = 'C:/Users/17619/miniconda3/envs/GPU/python.exe'
& $Python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
~~~

本机已用 Python 3.12.12、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128 完成本轮CPU试跑。安装了CUDA版本不表示显卡性能问题已解决。GPU检查入口为 scripts/check_gpu.py；相关历史记录见 gpu_environment.md。

### 必需的本地资产

| 资产 | 默认位置 | 获得方式 |
| --- | --- | --- |
| MINC处理后数据 | data/processed/minc5/manifest.csv 及 images/ | 按第2节下载 |
| ImageNet预训练权重 | models/pretrained/ | 项目下载脚本 |
| 训练后的五类权重 | runs/week1_resnet18/best.pt | 项目内部共享，或按第2节自行训练 |
| 企业原始RGB-D | data/映界_rgbd数据/ | 企业提供的数据包，本地解压 |
| 检查与预测结果 | output/ | 按下面命令生成 |

只有ImageNet权重不能代替训练后的五类best.pt。载入checkpoint应使用本项目自己训练或可信成员提供的文件。

## 2 下载公开数据并训练材质分类器

下载五类完整数据，保留官方训练、验证和测试划分：

~~~powershell
& $Python -m material_recognition.prepare_probe --full --output-dir data/processed/minc5 --workers 8
& $Python -m material_recognition.validate_manifest data/processed/minc5/manifest.csv
& $Python -m material_recognition.download_weights
~~~

该路线使用数据集服务下载五类图像，不需要额外安装huggingface_hub。依赖网络；现有文件会复用。prepare_minc.py是已有本地Parquet数据时的另一条路线。

复现第一周训练预算，结果放入新目录：

~~~powershell
& $Python -m material_recognition.train --manifest data/processed/minc5/manifest.csv --device cpu --head-epochs 2 --full-epochs 1 --batch-size 32 --workers 0 --output-dir runs/week1_reproduction
~~~

预期生成 best.pt、config.json、history.json。与原实验采用相同轮数不保证得到完全一致的权重或指标。原实验还包含分阶段初始化，记录保留在本机runs下。

网络或时间有限时，只检查小样本流程：

~~~powershell
& $Python -m material_recognition.prepare_probe --output-dir data/processed/minc5_probe
& $Python -m material_recognition.train --manifest data/processed/minc5_probe/manifest.csv --device cpu --head-epochs 1 --full-epochs 0 --workers 0 --output-dir runs/smoke
~~~

小样本只验证流程，不用于判断模型最终表现。训练脚本的 --device cuda 会在CUDA不可用时报错；不要仅根据CUDA可用就判断真实运行速度。

## 3 评估与单区域推理 Demo

以下使用自行训练得到的权重。如果已有第一周权重，将路径改为 runs/week1_resnet18/best.pt：

~~~powershell
& $Python -m material_recognition.evaluate --checkpoint runs/week1_reproduction/best.pt --manifest data/processed/minc5/manifest.csv --split validation --output-dir output/minc_validation
& $Python -m material_recognition.demo --checkpoint runs/week1_reproduction/best.pt --port 7860
~~~

评估生成metrics.json和逐样本predictions.csv。Demo启动后打开 http://127.0.0.1:7860，上传单个已裁剪RGB接触面图像，点击“识别”。它不从整张图自动定位物体。

evaluate和Demo默认自动选择可用CUDA；如果本机GPU异常，可先在当前PowerShell窗口设置以下变量，使后续Python进程使用CPU：

~~~powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
~~~

需要恢复时新开终端，或者执行 Remove-Item Env:CUDA_VISIBLE_DEVICES。产线probe脚本固定使用CPU，不受此设置影响。

当前分数是未经校准的softmax输出，不代表真实正确概率。独立test划分应在方案冻结后评估，不用于选模型或调阈值。

## 4 检查企业 RGB-D 数据

解压后默认目录为 data/映界_rgbd数据/，文件按共同时间戳配对：

~~~text
2026_08_26_09_19_22_657_color.jpg
2026_08_26_09_19_22_657_depth.png
~~~

RGB为JPG；Depth要求16位单通道PNG，不能另存成8位图片。当前脚本解析时间戳格式 YYYY_MM_DD_HH_MM_SS_fff；其他命名规则需要适配。

~~~powershell
& $Python scripts/inspect_rgbd.py --source 'data/映界_rgbd数据' --output output/rgbd_review
~~~

| 输出 | 用途 |
| --- | --- |
| summary.json | 数量、缺失配对、解码错误、尺寸和深度统计 |
| pairs.csv | 每一对图像的路径及检查结果 |
| index.html、previews/ | RGB-D本地浏览页面和预览 |
| rgb_contact_sheet.jpg | 多帧RGB总览 |
| rgb_depth_examples.jpg | RGB与深度对照 |
| objects.template.csv | 物体标注字段模板 |

脚本不修改原始文件。先查看summary.json中的errors、missing_rgb、missing_depth及size_mismatches，生成页面并不表示每个输入都正常。当前245对的既有检查全部可读取、尺寸一致。

深度色标使用原始值，不自动换算为米。RGB与Depth同尺寸也不能证明几何对齐。按时间间隔生成的time_group不等于SKU或训练划分。

## 5 启动浏览器并导出框选草稿

在一个终端运行并保持该终端打开：

~~~powershell
& $Python -m http.server 7862 --bind 127.0.0.1 --directory output/rgbd_review
~~~

打开 http://127.0.0.1:7862。端口被占用时换成7863，并相应修改访问地址。

1. 使用上一帧、下一帧或滑块定位图片。
2. 选择“物体整体框”，在RGB图上拖动。
3. 选择“抓取接触面框”，在物体框内部拖动。
4. 材质、SKU不确定时留空，点击“记录物体”。
5. 可以继续记录其他物体或切换图片；最后点击“导出标注JSON”。
6. 浏览器会下载 yingjie_rgbd_annotations.json。将它保存至本地 output/annotations/，关闭页面前确认已导出。

草稿只存在当前页面内存，刷新或关闭后不会自动恢复。只画物体框不足以启用记录按钮，必须同时画接触面框；接触面必须包含在物体框内。

导出坐标为原图像素xyxy半开区间，即左上包含、右下不包含；所有记录均标记needs_review，选择了标签也不等于完成复核。页面导出JSON不是当前训练清单，尚需复核、裁剪及格式适配，不能直接输入train.py。Depth同坐标框仅供检查，不是三维定位结果。

## 6 单帧人工框材质测试

使用内部共享的第一周checkpoint复现原来的4个区域：

~~~powershell
& $Python -m scripts.probe_production_rgb --source 'data/映界_rgbd数据' --checkpoint runs/week1_resnet18/best.pt --output output/production_probe
~~~

预期生成material_probe.jpg、material_probe.json和probe_roi_1.jpg等裁剪图。此命令不依赖浏览服务，也不自动检测物体。默认帧是2026_08_26_09_19_22_657。

测试其他帧时，在本地output/my_boxes.json保存原图坐标列表，例如：

~~~json
[[100, 120, 220, 260], [300, 200, 420, 360]]
~~~

以上仅为格式示例，必须换成自己图片中真实的物体位置：

~~~powershell
& $Python -m scripts.probe_production_rgb --source 'data/映界_rgbd数据' --frame '你的时间戳前缀' --boxes-json output/my_boxes.json --checkpoint runs/week1_resnet18/best.pt --output output/custom_probe
~~~

可用浏览器导出记录中的object_bbox_xyxy或contact_bbox_xyxy构造此列表；必须只使用指定帧的坐标。更换帧时必须提供boxes-json，不能沿用参考图坐标。实际抓取表面实验优先使用经确认的contact_bbox；既有产线试跑使用的是物体框。

## 7 复现六帧扩展测试

该脚本针对既有245对数据中的固定6帧24个手工ROI，选择配置保存在脚本SELECTION中。它不是针对任意新图片的检测器。

~~~powershell
& $Python scripts/probe_production_batch.py --source 'data/映界_rgbd数据' --checkpoint runs/week1_resnet18/best.pt --output output/production_batch
~~~

| 输出 | 内容 |
| --- | --- |
| frames/ | 6对原始图像的本地副本 |
| crops/ | 24个无损PNG裁剪图 |
| figures/frame_*_results.png | 每帧原图框选与四个预测结果 |
| figures/all_24_predictions.png | 全部ROI预测总览 |
| figures/selected_frames.png | 六帧框选总览 |
| predictions.csv、predictions.json | 坐标、五类分数、预测与标签状态 |
| summary.json | 样本数、输出分布、局部正确率、环境及权重SHA-256 |
| code/selection.json | 本次固定采样配置快照 |

原有4个区域按用户确认记录为纸质、塑料、塑料、纸质；其余20个区域保持needs_review。全量accuracy为null，只有4个已确认标签的诊断结果可计算。换用自己的checkpoint会重新计算这些结果，不能期待复现原始分数。

记录的CPU前向耗时不含读取、预处理、分割、几何、绘图或服务请求，不是最终端到端延迟。

## 8 输出 JSON 的范围

当前存在两种不同用途的JSON：

- 浏览器标注草稿：schema_version、objects、物体框、接触面框、人工填写标签及needs_review状态。
- 模型试跑结果：object_id、frame_id、bbox、predicted_material、score、scores及真值确认状态。

这两种格式都不是最终端到端接口。Instance Mask、geometry、hardness/deformability和目标Prompt相关字段尚未贯通，不能将当前文件当作已完成的最终交付。

## 9 检查与常见问题

运行代码检查：

~~~powershell
& $Python -m unittest discover -s tests -v
~~~

| 问题 | 处理方式 |
| --- | --- |
| No Python at 指向旧路径 | 当前.venv失效，选用有效解释器或重建环境 |
| No module named material_recognition | 在仓库根目录运行，并使用文档中的 -m scripts.probe_production_rgb 形式 |
| 找不到best.pt | 权重不在Git中；取得内部checkpoint或先训练，再指定实际路径 |
| No readable pairs | 核对数据路径、文件后缀、时间戳格式和Depth位深 |
| 浏览页面打不开 | 确保已生成index.html，http.server仍运行，访问端口与启动参数一致 |
| 预测看起来明显错误 | 保存错误样例并复核真实标签；当前模型只是在现场试跑，分数不证明正确 |

## 10 提交范围

仓库保存源代码、操作文档和空标注模板。data/raw/、data/processed/、企业原始目录、runs/、models/、output/及权重文件均已忽略。data/enterprise/只保留README和manifest模板。

统一把新测试、浏览器导出和本地截图放在output/，不要使用git add -f提交企业资产。模型、数据和实测图片需通过项目内部渠道共享。
