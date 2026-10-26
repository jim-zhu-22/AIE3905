# AIE3905 RGB-D 物体感知项目

MirrorSpace × AIE3905 课程项目。目标是接收 RGB-D 图像及目标 Prompt，输出目标物体的实例 Mask、几何类别、材质与物理属性，并统一返回 JSON。

**当前已实现数据检查、RGB-D 浏览与人工框选、RGB 材质分类训练和试跑。完整端到端 pipeline 尚未实现。**

## 从这里开始

Milestone 1 已形成待审定的 [系统设计 v1](docs/system_design_v1.md)、[实例标注 Schema v1](docs/annotation_schema_v1.md) 和 [基线与评估计划 v1](docs/baseline_plan_v1.md)。其中 taxonomy、相机元数据和试标结果仍需与数据/应用方确认；文档不代表端到端 pipeline 已实现。

**[操作 README](docs/OPERATIONS.md)** 包含环境安装、数据与权重准备、训练、推理、浏览器框选、JSON 草稿导出和产线测试复现命令。所有命令在仓库根目录执行。

~~~powershell
git clone https://github.com/jim-zhu-22/AIE3905.git
Set-Location AIE3905
~~~

已有项目成员先更新自己的代码；企业数据需单独准备，SAM 2.1 Tiny 权重通过 Git LFS 下载，DINO 权重按模型清单准备。

## API 封装

项目已提供一个基于 FastAPI 的本地 HTTP API，将已有的 `pipeline-1.0` 分割结果和后续感知任务统一封装。API 当前支持两种使用方式：

- **已有成果查询**：读取导入的分割成果包，返回实例、类别、整数框、Mask 引用、叠加图和报告下载地址。
- **新图片与统一感知入口**：`POST /v1/segment/upload` 接收 PNG/JPEG 图片和目标 Prompt；`POST /v1/perceive` 以后台任务方式编排分割及已配置的属性模块，完成后通过任务接口查询结果。

### 启动 API

在仓库根目录执行以下命令。默认使用 `archive` 模式，仅读取已有成果；需要真实模型推理时使用 `model` 模式，并提供已安装模型依赖和权重的 Python 环境。

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r api_service\requirements.lock
powershell -ExecutionPolicy Bypass -File .\scripts\start_api.ps1
```

启动后可访问：

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:8000/` | 本地演示工作台 |
| `http://127.0.0.1:8000/docs` | Swagger 交互式 API 文档 |
| `GET /health`、`GET /v1/app/ready` | 服务和任务工作区状态 |
| `GET /v1/capabilities`、`GET /v1/modules` | 能力及属性模块状态 |

### 主要接口

| 方法和路径 | 用途 |
| --- | --- |
| `GET /v1/examples` | 查看已导入成果清单 |
| `GET /v1/examples/{sample_id}` | 读取已有 `pipeline-1.0` 分割结果 |
| `POST /v1/segment` | 使用 Base64 JSON 提交单张图片分割 |
| `POST /v1/segment/upload` | 以 multipart 表单直接上传图片进行分割 |
| `POST /v1/perceive` | 提交 RGB、Prompt、可选 Depth 和模块列表，返回后台任务编号 |
| `GET /v1/jobs/{job_id}` | 查询任务状态 |
| `GET /v1/jobs/{job_id}/result` | 获取已完成的 `perception-1.0` 结果 |
| `GET /v1/jobs/{job_id}/artifacts/{relative}` | 下载任务生成的 JSON、Mask、报告或图片 |

`/v1/perceive` 的任务成功返回 HTTP 202；属性模块通过 `api_service/modules.local.json` 配置，可使用仓库内置的 Python、command 或 HTTP adapter。Depth 当前只校验并保存元数据，不进行像素处理或物理尺寸测量。模型未准备好时接口会明确返回 503，不会用已有样例冒充新推理结果。

更完整的启动、调用、模块扩展和 Docker 说明见 [API 接入与启动文档](docs/app_integration.md)；API 原型说明见 [api_service/README.md](api_service/README.md)。API 测试可使用：

```powershell
.\.venv-api\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_api_service.py -v
```

## 本轮检测与分割成果

本轮模型选型已经从候选方案收敛为可复现的阶段性基线：

- 结论报告：[RGBD 小物体实例分割模型选型实测报告](第一轮模型选型对比报告_谢嘉翔.md)
- 首轮对照：[第一轮模型选型对比报告](第一轮模型选型对比报告_20261018.md)
- 独立实现：[第一周模型选型代码包](week1_model_selection/README.md)
- 模型与版本：[模型清单](week1_model_selection/MODEL_MANIFEST.md)
- 脱敏证据：[实验汇总 JSON](week1_model_selection/evaluation_summary_20261018.json)

当前阶段基线为 Grounding DINO Tiny + SAM 2.1 Tiny，默认整图检测后以框提示原图实例分割。报告中的匹配数是待复核框上的探索性证据，不是正式精度；企业原始 RGB-D 数据、标注和运行生成结果不纳入仓库。SAM 2.1 Tiny 权重通过 Git LFS 管理，DINO 权重按模型清单从 Hugging Face 固定修订下载。

## 已完成的功能

| 功能 | 代码入口 | 当前能力 |
| --- | --- | --- |
| MINC 数据准备 | material_recognition/prepare_probe.py、prepare_minc.py | 下载或提取五类数据，保留官方划分 |
| 训练与评估 | material_recognition/train.py、evaluate.py | ResNet-18 迁移学习、验证选模、逐样本预测与指标 |
| 单区域推理 Demo | material_recognition/infer.py、demo.py | 输入已裁剪 RGB ROI，输出五类分数 |
| RGB-D 数据检查 | scripts/inspect_rgbd.py | 配对、解码、尺寸、深度统计、重复文件检查及预览 |
| 浏览与人工标注 | scripts/rgbd_viewer.html | RGB/Depth 同帧查看、物体框和接触面框、JSON 草稿导出 |
| 单帧产线测试 | scripts/probe_production_rgb.py | 人工框选区域分类、裁剪图、结果图及 JSON |
| 六帧测试复现 | scripts/probe_production_batch.py | 固定6帧24个区域，输出原图副本、预测图片、CSV和JSON |

模型目前识别 paper、plastic、metal、wood、fabric 五类。Unknown 拒识尚未验证；fabric 不表示纯棉。Depth 当前用于检查和显示，未输入材质分类器。人工框不等于自动分割 Mask。

## 当前结果

- 公开数据：MINC 五类共12,500张；训练10,625张、验证625张、保留测试1,250张。既有 ResNet-18 基线验证 Accuracy 为75.20%，Macro-F1 为0.7480。验证集参与选模，不能作为独立测试成绩。
- 企业数据：已检查245对 RGB-D，全部可读取且配对完整。物体框、实例Mask、SKU及相机参数未随该批图像提供，深度单位与几何对齐尚未确认。
- 产线探索：人工选取6张图片中的24个区域。原有4个用户确认标签中1个预测正确；新增20个区域缺少真值，不能计算整体准确率，也不能把4个样本的25%推广到整批数据。
- 同一SKU可能重复出现在多帧中，尚未形成按SKU隔离的现场测试集。

本轮现场试跑使用 CPU，未重新训练。历史周报见 [第一周公开数据基线](docs/week1_report.md)；其中“企业数据等待提供”等内容属于当时状态，以本页当前进度为准。

## 八周里程碑

| 阶段 | 目标 | 交付 |
| --- | --- | --- |
| 第1–2周 | 数据与任务理解、架构、输入输出、taxonomy、标注格式、候选模块和指标 | System Design v1、Annotation Schema、Baseline Plan |
| 第3–4周 | RGB-D + Prompt → Mask → Geometry → Material / Physical Attributes → JSON | 可处理新样本的端到端 MVP |
| 第5–6周 | 根据误差选择主要瓶颈，对模型或数据进行优化 | Improved Pipeline、Quantitative Results、Failure Analysis |
| 第7–8周 | 整合、benchmark、端到端耗时与资源测量、文档和演示 | 完整 pipeline、标注流程、数据、评估脚本、JSON接口、最终Demo及技术文档 |

当前处于 Milestone 1 的数据理解与材质基线探索阶段。几何类别、物理属性标签和完整标注协议仍需讨论。RTX 5060 8GB 级别运行和单场景约1秒是长期目标，本阶段主要完成测量分析，不强制达到部署指标；不要求完整三维重建、PLY或mesh输出。

## 数据与文件

- 公开数据：[MINC-2500 split 1](https://huggingface.co/datasets/mcimpoi/minc-2500_split_1)。
- 企业接入：[数据与清单约定](data/enterprise/README.md)。
- 标签规范：[材质接触面标注初稿](docs/annotation_guideline.md)。
- 详细操作：[操作 README](docs/OPERATIONS.md)。
- 旧十周材质识别规划保留于 [历史项目规划](docs/project_plan.md)，当前整体范围以上述八周里程碑为准。

企业原始数据、标注和 output 结果仅保存在本地，不进入仓库。新生成的测试结果统一写入 output/。SAM 2.1 Tiny 交付权重随仓库的 Git LFS 指针管理；DINO 权重仍需按模型清单下载。仅凭克隆代码不能重现企业测试，仍需通过项目内部渠道取得相应数据，或按照操作文档自行训练。

代码尚未声明开源许可证；公开数据和预训练权重分别遵循各自使用条款。
