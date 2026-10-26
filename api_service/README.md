# AIE 本地感知应用 v0.3

当前已包含工作台、后台任务、历史记录、实例查看、同学成果包导入和可扩展模块接口。

- 完整应用说明：[接入与启动](../docs/app_integration.md)。
- 简短成果报告：[阶段成果](../docs/app_progress_20261026.md)。
- 本轮测试：23 项旧接口测试与 17 项应用测试通过；当前和全新环境完成实际应用流程验证。
- 重新打包：`python scripts/package_app.py`。

以下保留 v0.2 API 说明供旧调用方参考；当前任务与模块功能以应用接入文档为准。Docker 数据挂载在 v0.3 改为可写，实际运行尚未验证。

# AIE 分割 API 原型 v0.2

当前交付将同学的 `pipeline-1.0` 分割包作为 HTTP API 发布，并提供调用现有模型的适配入口。

| 功能 | 本机验证状态 |
| --- | --- |
| 服务状态、接口文档、样例读取、34 个 mask 与报告下载 | 已通过实际 HTTP 验证 |
| 输入校验、错误状态、模型加载与结果导出逻辑 | 23 项代码测试通过；模型逻辑使用合成测试替身 |
| 冻结依赖的独立复现 | 全新、不继承系统包的 Python 3.12 环境安装成功，并通过相同 52 项 HTTP 检查 |
| 新图片调用 Grounding DINO + SAM 真实推理 | 适配代码已写，当前没有 PyTorch 环境，未验证 |
| Docker | 配置已写；当前没有 Docker，未构建、未运行 |
| 深度像素处理、尺寸计算、材质和几何模块集成 | 未接入 |

## 今晚如何演示

本机已导入样例，输出位于 `output/api/samples/frame_211/`。从项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_api.ps1
```

打开 `http://127.0.0.1:8000/`，选择 frame_211，点击“读取已有结果”。页面会展示 34 个实例、结果 JSON、叠加图，以及原始报告和第一个 mask 的链接。页面明确标注这是已有结果。

随后打开 `/health` 展示服务状态，打开 `/docs` 查看和试用接口；`/docs` 的 Swagger UI 资源需要网络，首页演示和 API 请求本身不依赖外部网页资源。

如果已有服务正在使用 8000 端口，直接打开页面即可；也可通过启动脚本的 `-Port 8001` 参数另开端口。

## 在新环境准备

需要 Python 3.12。以下命令从项目根目录执行：

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install -r api_service\requirements.lock
```

API 运行依赖包括 FastAPI、Uvicorn、Pydantic、Pillow 和 python-multipart；requirements.lock 冻结了 16 个直接与间接依赖。新环境中的模型推理依赖需另行准备。当前电脑由已有的 Python 运行环境创建 `.venv-api`，启动脚本会优先使用它。

首次导入同学的结果包：

```powershell
.\.venv-api\Scripts\python.exe -X utf8 -m api_service.import_sample --source D:\Code\AIE\frame_211\frame_211
```

换电脑后将 `--source` 改成实际包目录。导入时检查 JSON 格式、实例数、文件路径、mask 尺寸、0/255 二值像素，以及 mask 的非零边界是否与整数半开区间 bbox 一致。目标目录已存在时不会覆盖；可以改用 `--sample-id another_sample`。

本地提交压缩包已包含这份导入后的样例，解压后无需重复导入。样例图像、权重、运行结果不加入 Git；压缩包用于本次本地交付。

## 接口约定

| 方法和路径 | 用途 |
| --- | --- |
| `GET /health` | 服务存活状态及 `model_ready` |
| `GET /ready` | archive 检查已有成果可读；model 检查模型就绪；未就绪返回 503 |
| `GET /v1/capabilities` | 当前模式、是否能处理新图、格式与大小限制 |
| `GET /v1/examples` | 已导入样例清单 |
| `GET /v1/examples/{sample_id}` | 已有分割结果，明确返回 `new_inference: false` |
| `POST /v1/segment` | Base64 RGB 图片、prompt、可选深度元数据；只在模型就绪时推理 |
| `POST /v1/segment/upload` | 直接上传 PNG/JPEG，表单字段为 image、prompt、frame_id、mode、depth_meta |
| `GET /v1/runs/{run_id}` | 按运行编号重新查询已完成的推理结果 |
| `GET /artifacts/{kind}/{bundle_id}/{relative}` | 下载已完成结果包中声明的文件 |
| `GET /openapi.json` / `/docs` | 接口定义及交互式文档 |

样例结果的主要字段沿用 `pipeline-1.0`：`run_id`、`frame_id`、`image_size`、`instance_count`、`instances`、`depth`、`warnings`、`errors`。`mask_path` 保留包内相对路径，新增 `mask_ref` 作为 API 文件地址。

`execution.mode` 区分 `archived_sample` 与 `model`，`execution.new_inference` 表示本次是否调用了模型。已有样例保留原运行编号，不生成虚假的新推理编号。样例原始 JSON 和报告按原文件发布；API 响应另外增加来源标记、文件链接及深度文件未随包交付的提示。

新推理的框按最终 mask 重算为整数半开区间 `[xmin,ymin,xmax+1,ymax+1]`；无匹配返回 HTTP 200、`status: no_target` 和空列表。缺参数或字段错误返回 422，损坏图片返回 400，过大图片或请求返回 413，格式不支持返回 415，模型未就绪或忙碌返回 503，推理异常返回 500。最多 15 MiB 解码图片、2000 万像素，总请求体上限 22 MiB；有无 Content-Length 都会检查实际接收长度。

深度元数据只记录、不参与推理；新推理响应明确将 `depth_usable` 设为 false。两种模式都不输出物理尺寸，也未接入统一 `/v1/perceive` 整链。

## 调用示例

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$result = Invoke-RestMethod http://127.0.0.1:8000/v1/examples/frame_211
$result.instance_count
Start-Process ('http://127.0.0.1:8000' + $result.overlay_ref)
```

提交新图片（在 archive 模式下会明确返回 503，不会返回样例替代结果）：

```powershell
$imagePath = 'C:\your\image.jpg'
$payload = @{
    frame_id = 'my_frame'
    image_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))
    prompt = 'box. bag. bottle. carton. packet.'
    mode = 'full'
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/segment -Method Post -ContentType 'application/json' -Body $payload
```

## 切换真实模型

需先按 `week1_model_selection/README.md` 准备 PyTorch、模型依赖及固定版本的 DINO 和 SAM 权重。代码复用 `GroundedSegmenter.predict`；服务启动时加载一次，串行执行推理，成功请求使用独立 run_id 保存文件。需要 SAM 路径不同时设置 `AIE_SAM_WEIGHT`，设备通过 `AIE_DEVICE` 指定（默认 cpu）。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_api.ps1 -Mode model -Python 'C:\your\model-environment\python.exe'
```

该解释器还需安装 `api_service/requirements.lock`。在 model 模式下，`GET /ready` 返回 200 后才能提交新图片。archive 模式的 `/ready` 返回 200 只表示样例服务就绪，不能据此推断模型可用；请同时检查 `/v1/capabilities` 的 `new_inference_available`。模型加载失败会记录服务日志并保持未就绪，不静默降级到样例结果。

## 验证

```powershell
.\.venv-api\Scripts\python.exe -m pip install -r api_service\requirements-dev.txt
.\.venv-api\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_api_service.py -v
.\.venv-api\Scripts\python.exe -X utf8 scripts\check_api.py
```

最后一个命令需要 API 正在运行，检查实际 HTTP 请求和所有 mask 下载，生成 `output/api_verification_v02.json` 和 `output/sample_response.json`。默认样例包的验证结果是 52 项 HTTP 检查通过。

测试中的合成模型用于验证接口适配、空结果、失败恢复、编号隔离和模型仅加载一次，不构成真实模型效果或速度验证。

## Docker 配置（未运行验证）

Docker 镜像仅包含 archive 服务及其依赖，不包含模型运行环境、权重和企业图像；已有样例目录通过只读挂载提供。服务以非 root 用户、单进程运行，容器根文件系统只读，并以 `/ready` 检查当前配置功能是否可用；本机尚未实际运行 Docker。

在项目根目录准备好 `output/api/samples/frame_211` 后执行：

```powershell
docker compose -f compose.api.yaml up --build -d
```

访问 `http://127.0.0.1:8000/`。当前电脑没有 Docker，因此只能汇报“已编写 Dockerfile 与 Compose 配置，待环境验证”。如果本机已有演示服务占用 8000，请先停止该服务或修改 Compose 中的宿主端口。

参考：[FastAPI 容器部署](https://fastapi.tiangolo.com/deployment/docker/)、[Dockerfile](https://docs.docker.com/reference/dockerfile/)、[Compose 数据卷](https://docs.docker.com/reference/compose-file/volumes/)。

## v0.2 的错误格式与追踪

所有错误统一为下面的结构；v0.1 的 `detail` 字段改为 `error`，调用方需要同步调整。每个 HTTP 响应都返回 `X-Request-ID`；日志使用相同编号。验证失败只提供字段位置和原因，不回显 Base64 或完整请求数据。

```json
{"error":{"code":"model_not_ready","message":"当前为 archive 模式，未启用新图片推理。","request_id":"...","fields":[]}}
```

`503 model_busy` 带 `Retry-After: 1`，模型未就绪带 `Retry-After: 5`。`X-Process-Time-Ms` 表示服务处理到响应开始的时间，不含响应网络传输；结果内的 `execution.api_elapsed_ms` 仅统计图片解码、推理及导出，具体范围见 `timing_scope`，不包含最终校验和原子发布。

## 上传示例

在 `/docs` 打开 `POST /v1/segment/upload`，点击 Try it out，可直接选择原始 PNG/JPEG 文件，无需手动转换 Base64。表单 `depth_meta` 可填深度元数据 JSON 字符串；没有就留空。不会读取深度像素，也不接受客户端声明的 `depth_usable: true` 作为深度可用的证据。

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/segment/upload -F "image=@C:/your/image.jpg" -F "prompt=box. bag."
```

图片按文件实际内容校验，而非信任扩展名或浏览器声明的 MIME。坐标采用图片存储的像素方向，不按 EXIF 自动旋转。当前同步推理每进程仅处理一张图，启动脚本固定一个 worker；不要通过增加 worker 让多份模型争用同一设备。

## 原子发布与代码结构

导入和推理先在数据目录中的隐藏临时目录写入，检查所有文件、像素、整数框和结构后才重命名为正式结果包。失败会清理本次临时文件；同名导入不会覆盖。已有样例与推理输出分目录保存，查询和下载不会发布半成品。

- `schemas.py`：请求、深度元数据、结果及错误结构，校验实例数、状态、唯一编号、框范围和有限分数。
- `images.py`：Base64/文件内容解码、PNG/JPEG 格式和像素限制。
- `storage.py`：文件路径检查、成果校验和原子发布。
- `errors.py`：解析前请求限制、统一错误、请求编号和处理日志。
- `app.py`：接口与现有模型之间的适配。

本轮评价、已修复问题及剩余验证项见 `docs/api_review_20261026.md`。真实模型与 Docker 尚未验证，不能据此声称整个 RGB-D 系统已完成。

本地提交包可用 `python scripts/package_api_progress.py` 重新生成；打包脚本仅包含列明的源码、文档、验证记录及 frame_211 样例，不包含虚拟环境和模型权重。需先运行验证脚本并准备记录文件。
