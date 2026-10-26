# 连续图片输入与完成通知 · v0.4

## 浏览器里使用

打开 http://127.0.0.1:8000/ ，选择“上传新图片”，一次选择或分多次添加 PNG/JPEG，再点击“开始 / 继续连续处理”。应用每次只发送一张，收到这张的最终状态后再发送下一张。开启后还可以继续添加图片。

每张图片完成后自动更新叠加图、实例、属性和下载地址，无需手动刷新。单个任务处理失败后继续下一张；参数错误或损坏图片会单独标记。模型未就绪、服务异常、网络中断或队列满时暂停发送并保留剩余项，排除问题后点击继续。暂停发送不会取消已经提交的任务。

“自动显示新结果”默认开启，其他程序提交的图片完成后也会显示在工作台。手动打开历史任务会关闭自动跟随，避免新结果打断查看。可以重新勾选来恢复跟随。

待上传文件保留在当前页面，关闭或刷新页面后需重新选择；已经提交的任务及结果继续保存在服务器。移除待发项或清理本页已完成项，不会删除服务器历史。

每张 RGB 最多 15 MiB，当前页面最多保留 100 项。一张深度图不能复用于一批 RGB：浏览器选择深度图时仅允许配对一张待发 RGB；其他程序可通过 API 为每张 RGB 单独提供 depth/meta。

## 其他程序提交图片

复用 `POST /v1/perceive`，multipart 字段保持不变。增加可选 `Idempotency-Key` 请求头：

- 同一张图片、相同参数的网络重试使用相同编号，服务器返回原任务，不重复入队。
- 同一编号配上不同图片或参数返回 409。
- 有意重新推理时生成新编号。
- HTTP 202 表示已接收，最终状态通过 `GET /v1/jobs/{id}` 查询，完成后读取 `/result`。

仓库提供一个只依赖 Python 标准库的客户端：

```python
from examples.perception_client import PerceptionClient

client = PerceptionClient("http://127.0.0.1:8000")
for image in ["images/frame_001.png", "images/frame_002.png"]:
    job = client.submit_image(image, prompt="box. bag.", modules=[])
    finished = client.wait(job)
    if finished["result_ref"]:
        result = client.result(finished)
        print(result["run_id"], result["instance_count"])
```

命令行按顺序发送：

```powershell
.\.venv-api\Scripts\python.exe -X utf8 scripts\send_images.py images\frame_001.png images\frame_002.png --prompt "box. bag."
```

如果其他程序向文件夹保存图片，可由发送器持续接收：

```powershell
.\.venv-api\Scripts\python.exe -X utf8 scripts\send_images.py --watch D:\incoming_images --prompt "box. bag."
```

发送器等文件大小和修改时间稳定后读取，逐张等待结果。生产程序优先先写临时文件，再重命名为 .png/.jpg，以免接收半张图片。相同文件和参数重启发送器后会回查原任务，不重复推理；已经失败的同一输入不会被无限重试，有意重跑请使用新提交编号。Ctrl+C 停止发送，已提交任务继续处理。

## 完成后自动通知

`GET /v1/events` 返回标准 SSE（text/event-stream）。初次连接发送 snapshot，随后发送 job 事件。事件只包含任务状态与结果地址，不包含图片像素。浏览器使用同源 EventSource，不额外安装前端依赖。

事件编号保存在 SQLite，与任务变化在同一事务中提交。断线后支持 Last-Event-ID 或 after 参数重放；旧编号超过保留范围或数据库恢复后编号不匹配，会重新发送快照。连接有心跳和自动重连，页面断线时降级为查询。`once=true` 可用于检查有限的事件响应。

## 验证与边界

队列、连续到图、错误继续、暂停恢复、幂等提交、并发重试、事件重放及旧数据库迁移均有自动化测试。模型分支使用合成测试替身验证，不代表真实模型识别效果或速度。

当前本机缺少 PyTorch 等真实推理依赖，服务仍以 archive 模式运行，真实图片提交会明确返回 503。需要用已准备好模型及权重的环境运行 `scripts/start_api.ps1 -Mode model -Python <模型环境解释器>`，才能实际分割和识别新图片。原有 frame_211 样例仅用于已有成果与通知链路验证。

下载者使用自己电脑的 CPU/GPU，安装、权重准备、离线预检和启动步骤见 `docs/local_inference.md`。本次不在交付电脑上安装推理环境。

协议参考：[SSE 与 EventSource](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)。
