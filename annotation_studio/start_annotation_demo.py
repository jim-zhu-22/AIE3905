"""Shared Windows/macOS launcher; no third-party imports before environment setup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent


def running(url):
    try:
        with urlopen(url + "api/status", timeout=2) as response:
            data = json.load(response)
        if Path(data.get("output", "")).resolve() == (ROOT / "06-实验结果" / "自动标注工作区").resolve():
            return True
        raise RuntimeError("该端口属于另一份标注工作区，请关闭另一份服务或更换端口。")
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--only-inbox", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    if not args.install_only and running(url):
        print("服务已启动：" + url, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return
    required = [
        ROOT / "05-模型权重" / "sam2.1_t.pt",
        ROOT / "05-模型权重" / "grounding-dino-tiny" / "model.safetensors",
        ROOT / "06-实验结果" / "训练记录" / "week1_resnet18" / "best.pt",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("模型文件不完整，请重新解压完整包：\n" + "\n".join(missing))
    requirements = ROOT / "requirements-demo.txt"
    signature = hashlib.sha256(requirements.read_bytes() + sys.version.encode()).hexdigest()
    marker = Path(sys.prefix) / ".annotation-dependencies.json"
    try:
        ready = json.loads(marker.read_text("utf-8"))["signature"] == signature
    except (OSError, ValueError, KeyError):
        ready = False
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    if not ready or args.repair:
        print("首次准备依赖（需要联网，可能需要几分钟；请勿关闭窗口）…", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--prefer-binary",
                        "-r", str(requirements)], check=True, env=env)
        subprocess.run([sys.executable, "-c",
                        "import torch,torchvision,transformers,ultralytics,numpy,PIL; print('依赖已就绪')"],
                       check=True, env=env)
        marker.write_text(json.dumps({"signature": signature}), encoding="utf-8")
    if args.install_only:
        print("依赖准备完成。之后双击启动文件即可。", flush=True)
        return
    command = [sys.executable, "-m", "rgbd_perception.annotation_service", "--port", str(args.port)]
    source = ROOT / "04-数据" / "映界_rgbd数据"
    if source.is_dir() and not args.only_inbox:
        command.extend(["--source", str(source)])
    if args.paused:
        command.append("--paused")
    print(f"正在启动：{url}\n请保持本窗口运行。关闭窗口会停止服务，已保存结果不会丢失。", flush=True)
    process = subprocess.Popen(command, cwd=ROOT / "02-代码", env=env)

    def show_when_ready():
        for _ in range(180):
            if process.poll() is not None:
                return
            if running(url):
                if not args.no_browser:
                    webbrowser.open(url)
                return
            time.sleep(1)
    threading.Thread(target=show_when_ready, daemon=True).start()
    try:
        code = process.wait()
    except KeyboardInterrupt:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
        return
    if code:
        raise RuntimeError(f"服务退出，错误码 {code}。请把此窗口截图发给项目负责人。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n启动失败：{exc}\n可重新双击重试；仍失败请把本窗口截图发给项目负责人。", flush=True)
        sys.exit(1)
