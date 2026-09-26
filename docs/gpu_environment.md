# GPU 环境检查记录

检查日期：2026-09-26。机器为 RTX 4050 Laptop GPU，显存约 6 GB，NVIDIA 驱动 595.95。

## 已有可识别显卡的环境

| 环境 | Python | PyTorch | CUDA runtime |
| --- | --- | --- | --- |
| `GPU` | 3.12.12 | 2.9.1+cu128 | 12.8 |
| `dqdetr` | 3.9.25 | 2.8.0+cu128 | 12.8 |
| `molparser` | 已有环境 | 2.5.1+cu121 | 12.1 |

三个环境均能识别显卡。初期 `GPU` 环境的 32 × 32 矩阵运算成功；较大矩阵同步与 ResNet 测试曾阻塞。关闭 cuDNN、限制 CPU 线程以及切换 CUDA runtime 后仍观察到异常。

`nvidia-smi` 在训练前即显示持续 100% 利用率、SM 时钟 210 MHz，并报告 SW Power Cap 与 SW Thermal Slowdown。电脑接通电源，温度读数约 54～55°C。这些读数提示需要进一步检查显卡/驱动运行状态，不能据此断言某个应用或硬件故障。

15:00 左右重新进行限时检查，`GPU` 环境成功完成 ResNet-18 前向与反向传播，实际运算耗时 3.5857 秒，证据见 `docs/gpu_check.json`。32 张批量检查也通过，耗时 14.0993 秒，见 `docs/gpu_batch32_check.json`。连续测速 FP32 批量约 14 秒，混合精度约 9 秒；当前显卡运行效率明显偏低，恢复原因尚未确认。没有关闭用户应用、重置显卡或重启电脑。

已在 Python 3.12 CUDA 环境完成独立的小样本 GPU 训练：200 张训练、50 张验证，分类头 1 轮、全网络 1 轮；验证准确率 54.00%，Macro-F1 0.5235，证据在 `runs/week1_gpu_probe/`。完整数据基线改用 Python 3.11 CPU 环境，分类头 2 轮后微调整个网络 1 轮，证据在 `runs/week1_resnet18/`。此前 `runs/week1_gpu_resnet18/config.json` 只是中止的全量 GPU 尝试，不能作为完成证据。完整 GPU 训练需在显卡状态恢复后复核。

## 正式训练入口

在项目目录执行：

```powershell
& 'C:\Users\17619\miniconda3\envs\GPU\python.exe' -m material_recognition.train --device cuda --head-epochs 2 --full-epochs 3 --batch-size 32 --workers 2 --output-dir runs/week1_gpu_retry
```

训练前先运行实际前向和反向检查：

```powershell
& 'C:\Users\17619\miniconda3\envs\GPU\python.exe' scripts/check_gpu.py --timeout 60 --output docs/gpu_check.json
```

`--device cuda` 会拒绝静默回退 CPU。运行配置记录实际设备、PyTorch、CUDA runtime 与显卡名称；是否完成训练还应同时检查 `history.json` 与 `best.pt`，不能只看环境配置。
