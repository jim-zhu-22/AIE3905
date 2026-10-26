# 下载后使用自己的 CPU / GPU

本 demo 不需要云端推理账户。界面、队列和接口在下载者本机运行；分割和识别由下载者配置的模型环境使用本地算力执行。当前交付机器不安装推理环境，不包含已验证的真实模型性能数据。

## 环境要求

- 基线解释器：Python 3.12，Windows 或 Linux。
- 使用已有现代浏览器打开本机地址 http://127.0.0.1:8000/。
- 安装 API 依赖、匹配的 PyTorch/torchvision，以及 Transformers、Ultralytics 和 NumPy。
- CPU 可用于模型基线；GPU 可选，使用本程序的 CUDA 路径时需匹配 NVIDIA 驱动和 PyTorch CUDA 版本。
- 准备 Grounding DINO Tiny 固定修订和 SAM 2.1 Tiny 权重；材质识别另需组员训练好的 checkpoint。
- 图像支持 PNG/JPEG，单张最多 15 MiB、2000 万像素。速度、内存与显存要求需在目标电脑实测，页面及时刷新不代表模型能达到视频帧率。

API 的轻量依赖已冻结并验证；模型依赖沿用仓库声明的版本范围，尚未在本机安装验证。初次部署请先完成预检和一张真实图片试跑，再启用连续输入。Docker 不是直接本机运行的前提，现有 Docker 镜像只包含轻量 API 依赖。

## Windows CPU 安装示例

在解压后的项目根目录执行：

```powershell
py -3.12 -m venv .venv-model
.\.venv-model\Scripts\python.exe -m pip install --upgrade pip
.\.venv-model\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv-model\Scripts\python.exe -m pip install -r api_service\requirements-model.txt
```

GPU 用户将 PyTorch 安装命令替换为 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 中与系统、驱动匹配的命令，不将 CPU 安装命令当作 GPU 环境。Linux 使用 `python3.12 -m venv .venv-model` 和 `.venv-model/bin/python` 对应执行。

## 准备模型权重

1. SAM：将真实的 `sam2.1_t.pt` 放到 `week1_model_selection/sam2.1_t.pt`。通过 Git 克隆项目的用户可在确认 Git LFS 已配置后执行 `git lfs pull`；直接下载 demo ZIP 不包含这个大文件，需要按 [Ultralytics 官方模型说明](https://docs.ultralytics.com/models/sam-2/) 获取对应模型，或从组员处取得相同权重。LFS 指针文本不是权重。
2. DINO：联网准备电脑执行下面的显式下载命令，确保与应用缓存位置相同。

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.model_cache\huggingface'
.\.venv-model\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny', revision='a2bb814dd30d776dcf7e30523b00659f4f141c71')"
```

版本和 SHA-256 详见 `week1_model_selection/MODEL_MANIFEST.md`。首次下载需要网络，缓存与权重完整后可设置 `HF_HUB_OFFLINE=1` 使用本地文件。

## 预检与启动

```powershell
.\.venv-model\Scripts\python.exe -X utf8 scripts\check_model_environment.py --device cpu --verify-hashes
# 可选：在当前机器实际加载权重验证，不下载模型
.\.venv-model\Scripts\python.exe -X utf8 scripts\check_model_environment.py --device cpu --load-model
$env:AIE_DEVICE = 'cpu'
powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1 -Mode model -Python .\.venv-model\Scripts\python.exe
```

GPU 环境预检使用 `--device cuda`，启动前设置 `AIE_DEVICE=cuda`；`auto` 会在可用时使用 CUDA，否则选择 CPU。预检只检查本地依赖、权重、设备和可选的实际加载，不安装或下载任何东西。只有 ready=true 才表示所检查的环境条件满足；单图效果和运行速度仍需真实图片验证。

启动后打开“模块接入”页确认分割可用，上传一张真实原图检查结果，再使用“开始 / 继续连续处理”。如果分割加载失败，应用保留已有成果查询，新的图片输入会返回明确错误。

## 接入材质与其他识别程序

复制 `api_service/modules.default.json` 为 `api_service/modules.local.json`，在 material 项中设置 enabled=true 和 options.checkpoint 的实际权重路径，再设置：

```powershell
$env:AIE_MODULE_CONFIG = (Resolve-Path api_service\modules.local.json).Path
```

使用安装了材质程序依赖的同一个模型环境重启。独立环境中的同学程序也可以通过 command 或 HTTP 适配器接入，详细合同见 `docs/app_integration.md`。未配置的模块不会生成预测标签。

## 其他程序或目录发送器

模型已启用时，运行 `scripts/send_images.py` 逐张发送文件，或使用 `--watch` 接收本地文件夹中的新图片。也可直接使用 `examples/perception_client.py`。具体命令、重复提交保护和实时结果通知见 `docs/continuous_input.md`。

运行日志记录在启动终端，任务和结果保存在 `output/api/workspace/`。完成目标机部署后可另存 `pip freeze` 记录实际模型环境；不要把本机 API 依赖锁文件当作已验证的 GPU 模型环境锁文件。
