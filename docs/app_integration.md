# AIE 本地感知应用：启动与同学程序接入

应用地址为 http://127.0.0.1:8000/，提供工作台、任务记录、模块接入和使用说明。当前保持本机运行，不向外发布企业样例。

## 启动

使用 Python 3.12 安装 `api_service/requirements.lock`，在项目根目录运行 `scripts/start_api.ps1`。现有本机环境 `.venv-api` 可继续使用。任务、输入和结果存放于 `output/api/workspace/`，历史记录使用 SQLite。默认单进程、一个后台处理线程；同一数据目录由操作系统文件锁限制为一个工作区进程。

关闭页面不会取消任务；正常重启后保留历史，恢复排队任务，处理中的中断任务明确标记失败，避免生成虚假成功。仅支持取消排队中的任务。工作区不可写时仍可使用旧样例查询接口，任务接口返回明确错误。

## 已有分割程序：直接导入成果包

同学可继续输出 pipeline-1.0 格式的 `segmentation.json`、`report.md`、`overlay.jpg` 和 `masks/*.png`，压缩成 ZIP 后在工作台导入。支持外层目录；导入时检查路径、解压大小、实例数、图片格式、二值像素与整数框一致性。已有同名样例不会覆盖。

已有分割包中没有原始 RGB 时，只能复用分割结果，不能将带颜色的叠加图当作原始图输入材质分类器；请求的属性模块会标为 skipped 并说明原因。

## 新图片与统一入口

`POST /v1/perceive` 接受 multipart 表单：

| 字段 | 含义 |
| --- | --- |
| rgb | 原始 PNG/JPEG，必填 |
| prompt | 目标文本描述，必填 |
| frame_id / mode | 帧编号、full 或 tiled |
| modules | 模块 ID 的 JSON 数组，例如 ["material","geometry"] |
| meta | 深度元数据 JSON 字符串，可省略 |
| depth | 可选的同尺寸 16 位单通道 PNG |

请求成功返回 HTTP 202 和任务编号。通过 `GET /v1/jobs/{id}` 查询进度，完成后读取 `/result` 或 `/download`。原始深度文件仅传递给模块，上层始终标记 depth_usable=false；同尺寸不表示已经配准，当前不进行深度像素处理或物理尺寸测量。

旧 `/v1/segment` 和 `/v1/segment/upload` 保留，支持只调用分割的同学继续联调。原始分割为 pipeline-1.0；聚合的感知结果使用 perception-1.0，避免把增加的属性字段误认为旧分割合同。属性统一放在每个实例的 attributes[module_id] 中。

## 配置属性模块

复制 `api_service/modules.default.json` 为 `api_service/modules.local.json`，设置环境变量 `AIE_MODULE_CONFIG` 为这个文件的绝对路径，然后重启。模块配置属于维护者可信代码；浏览器不能上传或修改可执行模块配置。配置样例默认关闭全部属性模块，不生成假标签。

每个模块使用唯一 id 和以下四种 adapter 之一：

1. material：复用仓库已有 `material_recognition.infer.Predictor`。options.checkpoint 指向实际权重；使用原始 RGB 的实例框裁剪，输出明确标注 region=object_bbox。需要同学原来的 PyTorch 环境，不将未验证的模型分数当作准确率。
2. python：factory 为 `包.模块:工厂函数`，工厂接收 options 字典并返回实现 analyze(context) 的对象。模板见 `api_service/extensions/example_geometry.py`。Python 模块需遵守同步执行合同；无法约束任意 Python 函数的执行时长，长耗时或独立依赖建议使用 command 或 http。
3. command：command 为参数数组，可使用 {request} 和 {response} 两个文件路径占位符。支持显式使用同学自己的 Python/Conda 解释器，不经过 shell；timeout_s 为每实例超时，最大 60 秒。模板见 `api_service/extensions/command_template.py`。
4. http：endpoint 指向维护者配置的服务，发送 RGB 与 mask 的 bbox 裁剪 PNG Base64，以及实例 ID、原图框和深度元数据。不发送本机文件路径或未配准的原始深度像素；timeout_s 最大 60 秒。仅在模块被启用且任务请求该模块时调用。

Python context 提供 run_id、instance_id、rgb_path、mask_path、可选 depth_path、bbox_xyxy、depth_meta 和临时 work_dir。RGB/mask 路径均指向本次任务快照。模块不要修改共享输入；自身中间文件放在 work_dir 下。

CLI 的 request.json 包含同样的字段和 schema_version=module-1.0；response.json 与 Python/HTTP 的返回结构一致：

```json
{"status":"ok","label":"paper","score":0.82,"source":"teammate_material","model_version":"v1","values":{"region":"object_bbox"}}
```

允许状态为 ok、unknown、skipped、error。非 ok 必须有 reason；没有证据的字段使用 null 或 unknown，不填造出的分数。物理属性等多字段结果可以放在 values 字典中。一个模块失败不会覆盖其他模块的成功结果。

## 替换分割模型

默认复用仓库 `GroundedSegmenter.predict(image,prompt,mode)`。可以通过维护者环境变量 `AIE_SEGMENTATION_FACTORY=包.模块:工厂函数` 替换：工厂接收 Settings，返回具有相同 predict 方法的对象。

返回对象须有 image（原图同尺寸 PIL 图）、instances、model_info、latency_ms。每个 instance 须有 mask（原图尺寸的 0/1 NumPy 数组）、label、det_score；可按 `week1_model_selection/segmentation_practice/pipeline.py` 中的数据类适配。同学若只交付离线文件，可先使用 ZIP 导入方式，不必改写程序。

## 结果与扩展

统一结果保留实例编号、整数框、mask 地址，另有各模块状态、来源、版本、耗时和缺失原因。新增属性模块只需注册新 id，无需改页面和任务编排。完整下载包含 perception.json、segmentation.json、report.md、overlay.jpg、每实例 mask，以及任务存在时的原始 RGB。

Docker 配置仍待真实构建运行验证。容器根文件系统只读，但 output/api 数据挂载必须可写才能保存 SQLite、任务和结果；API 镜像只包含轻量依赖，模型运行环境另行准备。

容器默认用户为 10001:10001。Linux 主机需让该用户可读写挂载目录，或通过 AIE_UID/AIE_GID 指定主机目录所有者的 UID/GID。容器健康检查访问 `/v1/app/ready`，检查任务工作区可用；旧 `/ready` 继续保留分割服务自身的就绪含义。

## 连续图片输入（v0.4）

新增多图排队上传、SSE 完成通知、幂等提交和其他程序接收客户端，见 `docs/continuous_input.md`。真实推理由下载者使用本地 CPU/GPU，安装与权重要求见 `docs/local_inference.md`；预检脚本不会安装或下载任何内容。
