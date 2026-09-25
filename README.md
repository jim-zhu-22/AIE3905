# 仓储机器人表面材质识别

MirrorSpace × AIE3905 课程项目。输入单个抓取区域的 RGB 图像，识别可见的接触表面材质。项目采用公开数据与企业现场数据双线推进，并以训练中未出现的新 SKU 评估泛化能力。

## 当前进度

已建立第一周的可运行基线：MINC-2500 五类提取、数据清单检查、ResNet-18 训练与评估、单图推理和 Gradio Demo。已下载 250 张五类均衡小样本，并用 ImageNet 预训练权重训练模型。50 张验证图上的准确率为 52%，Macro-F1 为 0.512；这个小样本结果只用于 POC，不代表完整数据或现场新 SKU 的效果。企业数据适配器要在收到标注导出样例后完成。当前执行记录见 `docs/week1.md`。

| 类别编号 | 标签 | 中文名称 |
| ---: | --- | --- |
| 0 | `paper` | 纸质 |
| 1 | `plastic` | 塑料 |
| 2 | `metal` | 金属 |
| 3 | `wood` | 木材 |
| 4 | `fabric` | 织物 |

`fabric` 表示织物，不能直接推断为纯棉。覆膜、贴纸或多种材质混合的区域需要按接触面重新确认标签。

## 数据

公开数据：[MINC-2500 split 1](https://huggingface.co/datasets/mcimpoi/minc-2500_split_1)。已筛选五类并保留原始 `train`、`validation`、`test` 划分，共下载 12,500 张图像；清单见 `data/processed/minc5/manifest.csv`。全部图片可解码，跨划分无字节级重复。原始及处理后图像均在 `data/` 下并被 Git 忽略。

企业数据放在 `data/enterprise/`，字段要求见该目录的说明。第一周先核对标签、ROI、SKU 和采集批次，并隔离最终测试 SKU。企业原始数据默认不进入公开仓库。

## 环境

本机使用 Python 3.11，第一周基线已在 CPU 版 PyTorch 上跑通。新环境在 Windows PowerShell 中可按以下方式安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

预训练 ResNet-18 权重可使用带续传与 SHA-256 校验的项目下载器取得：

```powershell
.\.venv\Scripts\python.exe -m material_recognition.download_weights
```

完整数据训练建议从 [PyTorch 官方安装页](https://docs.pytorch.org/get-started/locally/)选择与本机显卡匹配的 CUDA 安装命令；CUDA 安装包较大。CPU 也能运行，但完整训练会明显更慢。

## 第一周流程

网络较慢时，可先使用已下载的小样本走通完整流程：

```powershell
.\.venv\Scripts\python.exe -m material_recognition.prepare_probe
.\.venv\Scripts\python.exe -m material_recognition.validate_manifest data/processed/minc5_probe/manifest.csv
.\.venv\Scripts\python.exe -m material_recognition.train --manifest data/processed/minc5_probe/manifest.csv --output-dir runs/minc_probe --head-epochs 3 --full-epochs 0 --batch-size 16 --workers 0
.\.venv\Scripts\python.exe -m material_recognition.evaluate --checkpoint runs/minc_probe/best.pt --manifest data/processed/minc5_probe/manifest.csv --output-dir runs/minc_probe/evaluation
.\.venv\Scripts\python.exe -m material_recognition.demo --checkpoint runs/minc_probe/best.pt
```

小样本结果不可替代完整验证，也不可用于声称跨 SKU 泛化。公开 MINC 数据没有仓储 SKU 字段。

下载数据集并解压目标类别：

```powershell
.\.venv\Scripts\python.exe -m pip install huggingface_hub
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mcimpoi/minc-2500_split_1', repo_type='dataset', local_dir='data/raw/minc-2500_split_1', allow_patterns=['data/*.parquet','README.md'])"
.\.venv\Scripts\python.exe -m material_recognition.prepare_minc
.\.venv\Scripts\python.exe -m material_recognition.validate_manifest data/processed/minc5/manifest.csv
```

如果完整 Parquet 下载受网络限制，可以直接按官方划分抓取五个目标类别，避免传输其余 18 类。下载器会缓存页面并在服务限流时重试：

```powershell
.\.venv\Scripts\python.exe -m material_recognition.prepare_probe --full --output-dir data/processed/minc5 --workers 12
.\.venv\Scripts\python.exe -m material_recognition.validate_manifest data/processed/minc5/manifest.csv
```

小样本检查训练代码，再运行完整基线：

```powershell
.\.venv\Scripts\python.exe -m material_recognition.train --limit 64 --head-epochs 1 --full-epochs 0 --output-dir runs/smoke
.\.venv\Scripts\python.exe -m material_recognition.train
.\.venv\Scripts\python.exe -m material_recognition.evaluate --split validation
```

`--limit` 仅用于流程检查，取每个集合清单前面的样本，指标不能用于判断模型性能。训练根据验证集 Macro-F1 保存 `runs/minc_resnet18/best.pt`。确认配置后再运行一次独立测试：

```powershell
.\.venv\Scripts\python.exe -m material_recognition.evaluate --split test --output-dir runs/minc_resnet18/test
.\.venv\Scripts\python.exe -m material_recognition.demo
```

Demo 默认打开本机 `http://127.0.0.1:7860`，当前输入为已裁剪的单材质 ROI。模型分数不等于真实正确概率；范围外材质仍可能被高分误判。

## 十周研究路线

1. 第 1–2 周：公开与企业数据双线建立 ResNet-18 基线，完成模型切换 Demo。
2. 第 3–4 周：统一标签和验证协议，对比公开模型直接迁移、仅企业数据微调、公开加企业数据微调。
3. 第 5–7 周：比较 EfficientNet-B0，分析局部 ROI、包装文字、背景和拒识策略；第七周冻结模型。
4. 第 8–10 周：在独立企业测试集上评估未见 SKU，交付模型、标注规范、Material Recognition API、可视化 Demo 和实验报告；条件允许时开展真空吸盘验证。

公开数据与现场数据的指标分别报告。测试集不用于选模或调阈值。跨 SKU 成绩可以说明对新商品的迁移能力，但不能单独证明模型理解真实物理材质。

## 项目文件

- `material_recognition/prepare_minc.py`：从下载的 Parquet 提取五类并生成清单。
- `material_recognition/prepare_probe.py`：按官方划分下载试验小样本或五类全集。
- `material_recognition/download_weights.py`：下载并校验预训练 ResNet-18 权重。
- `material_recognition/validate_manifest.py`：检查类别映射、图像路径和 SKU 划分。
- `material_recognition/train.py`：ResNet-18 训练与验证选模。
- `material_recognition/evaluate.py`：输出指标、混淆矩阵和逐样本预测。
- `material_recognition/infer.py`、`demo.py`：统一推理及本地展示。
- `data/enterprise/README.md`：企业数据清单字段与接入约定。
- `docs/annotation_guideline.md`：接触面材质标注规范第一版。

代码及文档尚未声明开源许可证。数据与预训练权重须遵守各自的使用条款。
