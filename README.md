# AIE3905 RGB-D 物体感知项目

MirrorSpace × AIE3905 课程项目。目标是接收 RGB-D 图像及目标 Prompt，输出目标物体的实例 Mask、几何类别、材质与物理属性，并统一返回 JSON。

**当前已实现数据检查、RGB-D 浏览与人工框选、RGB 材质分类训练和试跑。完整端到端 pipeline 尚未实现。**

## 从这里开始

**[操作 README](docs/OPERATIONS.md)** 包含环境安装、数据与权重准备、训练、推理、浏览器框选、JSON 草稿导出和产线测试复现命令。所有命令在仓库根目录执行。

~~~powershell
git clone https://github.com/jim-zhu-22/AIE3905.git
Set-Location AIE3905
~~~

已有项目成员先更新自己的代码；数据和模型文件需单独准备，不随 Git 下载。

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

企业原始数据、标注、模型权重和 output 结果仅保存在本地，不进入仓库。新生成的测试结果统一写入 output/。仅凭克隆代码不能重现企业测试，需通过项目内部渠道取得相应数据及权重，或按照操作文档自行训练。

代码尚未声明开源许可证；公开数据和预训练权重分别遵循各自使用条款。
