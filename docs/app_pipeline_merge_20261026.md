# 应用层与真实模型链路合并 v0.5

2026-10-26。本地分支 codex/annotation-pipeline 已合入远程 271431c（应用层 v0.3），并保留本地 v0.4 的模块化模型、训练工具和 pipeline-1.0 导出。合并前检查点为 bc06305。此次没有向 GitHub 推送。

## 已完成

- 浏览器工作台、任务历史、ZIP 导入、后台编排和模块适配接入真实 DINO Tiny + SAM2.1 Tiny + ResNet-18 权重。修复默认模型适配器读取旧阈值属性导致的初始化异常。
- 同学原分割包可在页面补传同帧原始 RGB 后继续材质推理，入口为 POST /v1/jobs/from-sample；该路径不加载或重跑分割也能运行。带 RGB 的 v0.4 包自动读取 preprocess.json 的 rgb_path，并快照保存原始 RGB/Depth。不会使用 overlay 作为输入。尺寸检查不能证明图像同帧，调用者仍需确认身份。
- 深度扩展字段 processing_status/source_depth_path 被明确支持；缺失外部深度引用会保留并提示。depth_usable 始终为 false，不新增深度像素算法。
- 空结果统一输出 no_match，仍接受上游旧 no_target，并支持读取旧任务历史。
- 每个后台任务导出五阶段 pipeline-1.0：preprocess.json、segmentation.json、material_input.json、material_result.json、result.json，以及每实例 mask、ROI、报告和可用原始输入。perception.json 保留为 perception-1.0 的界面视图，HTTP 下载地址只留在该视图，标准文件使用包内相对路径。
- 材质适配器保留 top1_label/score/scores、权重 SHA-256 与候选来源；未验证接受策略时 status=unknown，标准 label=unknown。接触面标记 instance_bbox_fallback/contact_verified=false。界面显示候选和待确认原因，不将推理成功当作预测已通过验收。
- execution.new_inference 继续表示是否新跑分割；new_attribute_inference 单独表示是否新执行并得到属性结果。页面与报告区分“旧分割＋新材质”和“新图完整推理”。

## 验证结果

79 项主测试通过（包含原 40 项 API/工作台测试、本地 33 项链路测试、新增 6 项集成测试），另 4 项历史分割测试通过。测试中的合成模型只证明程序行为。

本机实际 HTTP 验证使用已缓存权重、CUDA 设备，未重新训练：

| 路径 | 实例 | 实际执行 | 验证 |
|---|---:|---|---|
| 原分割包补传 RGB | 34 | 复用分割、重新执行材质 | 34 张 mask HTTP 访问、候选返回、五阶段引用、ROI 尺寸、分数归一化、ZIP 校验通过 |
| 原始 RGB + Depth 上传 | 34 | 新跑 DINO/SAM/材质，Depth 仅保存 | 同上全部通过 |
| 浏览器带原图样例 | 34 | 复用分割、重新执行材质 | 页面显示 34 实例、正确来源标记、候选；搜索 i025 与切换实例可用；刷新后可从历史重新打开新图片结果 |

两条接口验证共 68 个实例，结果状态均为 partial，因为材质接受策略尚未验收。该状态不是程序运行失败。精度、接触面质量与部署速度没有因此获得验收。

证据目录 output/api_merged_20261026：tests_all.txt、tests_legacy.txt、environment.json、live_verification/verification.json；该目录包含企业衍生数据，不进入源码 Git 提交。使用现有 GPU 环境验证，未重新声称全新环境或 Docker 通过。上游 40 项与全新环境验证记录保留为其历史结果。

## 在这台电脑启动

当前验证服务地址：http://127.0.0.1:8765/ 。样例 frame_211 只有分割；frame_211_rgb 含同帧原图和深度，可选择材质直接处理。

服务关闭后，在仓库目录用 PowerShell 启动：

```powershell
$env:AIE_DEVICE = 'cuda'
$env:HF_HUB_OFFLINE = '1'
$env:AIE_API_DATA_DIR = Join-Path (Get-Location) 'output/api_merged_20261026'
.\scripts\start_api.ps1 -Mode model -Python .\.venv\Scripts\python.exe -Port 8765
```

启动脚本自动读取存在的 api_service/modules.local.json。该文件已在本机配置现有候选材质权重与 CUDA；本机配置不进入 Git 或源码包。不要在当前服务仍运行时占用同一端口/工作区。

## 新机器与可移植交付

源码安装和模型依赖仍见 perception/README.md。轻量 archive 环境继续使用上游 api_service/requirements.lock；把 API 接入模型环境时使用 api_service/requirements-integrated.txt（本次实际验证的 HTTP 依赖版本）。完整模型环境与 checkpoint 需另行准备，源码包不附权重。

合并包包含 code/、samples/、evidence/ 和本报告。从 code/ 运行 `python -m api_service.import_sample --source ../samples/frame_211_rgb --sample-id frame_211_rgb` 即可导入样例，archive 模式能查看历史分割。开启材质需复制 modules.default.json 为 modules.local.json，配置 enabled=true、options.checkpoint 的绝对路径与 options.device，并安装模型依赖；启动脚本自动加载本地配置。

正式业务输出读取 result.json；保留的 perception.json 供现有页面/HTTP 客户端使用，两者语义在本报告中已明确。没有请求材质或没有原图时，正式结果中的材质为 skipped，原始分割仍保留。后续模块可继续使用 Python/command/HTTP 适配；本轮没有虚构几何或物理属性标签。

## 下一步边界

重新打包：源码使用 `python scripts/package_delivery.py --output ../deliverables/AIE3905_app_pipeline_v0.5_source_20261026.zip`；含样例与证据使用 `python scripts/package_app.py --data-dir output/api_merged_20261026 --output ../deliverables/AIE3905_app_pipeline_v0.5_merged_20261026.zip`。后者会校验每个文件的 SHA-256，不附本机模块配置、模型权重、虚拟环境和任务数据库。

Docker 仅有配置，未运行验证；几何、深度等高线、三维尺寸和物理属性算法未实现。材质接触面与拒识需独立验证，疑似设备背景误检应人工复核。本次不修改训练标签，不将自动候选转成真值。
