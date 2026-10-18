# 模型版本与交付权重

本交付采用 **Grounding DINO Tiny + SAM 2.1 Tiny**。GitHub 保存版本、哈希与获取方法；SAM 2.1 Tiny 权重由本目录的 Git LFS 管理。模型只处理 RGB；Depth 暂不输入模型。

| 用途 | 模型与固定版本 | 本机文件 | SHA-256 |
| --- | --- | --- | --- |
| 文本提示检测 | [`IDEA-Research/grounding-dino-tiny` 修订 `a2bb814dd30d776dcf7e30523b00659f4f141c71`](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71) | Hugging Face 缓存中的 `model.safetensors`，689,359,096 字节 | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |
| 框提示分割 | [Ultralytics SAM 2.1 Tiny `sam2.1_t.pt`](https://docs.ultralytics.com/models/sam-2/) | `week1_model_selection/sam2.1_t.pt`，由 Git LFS 管理，78,105,722 字节 | `3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3` |

交付代码默认从 Hugging Face 加载上述固定修订，并从本目录读取 SAM 权重。克隆仓库时请确保 Git LFS 已安装并执行 `git lfs pull`；联网机器可用下列命令下载 DINO。

```powershell
..\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny', revision='a2bb814dd30d776dcf7e30523b00659f4f141c71')"
```

在本目录校验本地权重：

```powershell
(Get-FileHash -Algorithm SHA256 .\sam2.1_t.pt).Hash
```

Hugging Face 缓存路径会因平台和环境变量变化。定位缓存中该修订的 `model.safetensors` 后，同样使用 `Get-FileHash -Algorithm SHA256 <实际路径>` 核对。若本机已设置 `HF_HOME`，加载器沿用该路径；否则默认使用项目根目录 `.model_cache/huggingface`。原始实验记录的 DINO 修订字段为 `unknown`，此固定修订用于交付后复跑，复跑指标须独立记录。
