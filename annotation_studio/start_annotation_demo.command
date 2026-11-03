#!/bin/bash
# macOS Apple Silicon: bundled Python, isolated dependencies, then local website.
set -euo pipefail
cd -- "$(dirname -- "$0")"
finish() {
  code=$?
  if [ "$code" -ne 0 ]; then
    printf '\n启动未完成。请检查网络后重新双击；仍失败请将窗口截图发给项目负责人。\n'
  fi
  printf '\n按回车关闭此窗口…'
  read -r _unused || true
}
trap finish EXIT
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
if [ "$(uname -s)" != "Darwin" ]; then
  printf '此文件用于 macOS。Windows 请双击 start_annotation_demo.cmd。\n'
  exit 1
fi
if [ "$(uname -m)" != "arm64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" != "1" ]; then
  printf '此包的 Mac 运行时适用于 Apple 芯片（M 系列），不适用于 Intel Mac。\n'
  exit 1
fi
if [ ! -x ".runtime-macos/python/bin/python3" ]; then
  printf '正在解压包内 Python 运行时（不需要管理员权限）…\n'
  (cd runtime && shasum -a 256 -c python-macos-arm64.tar.gz.sha256)
  mkdir -p ".runtime-macos"
  tar -xzf "runtime/python-macos-arm64.tar.gz" -C ".runtime-macos"
fi
if [ ! -x ".venv-demo-macos/bin/python3" ]; then
  printf '正在创建本项目的独立环境…\n'
  ".runtime-macos/python/bin/python3" -m venv ".venv-demo-macos"
fi
".venv-demo-macos/bin/python3" -u "start_annotation_demo.py" "$@"
