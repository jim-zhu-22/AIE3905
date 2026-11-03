"""Portable offline annotation launchers; no Python needed by annotators."""
import stat
import zipfile
from pathlib import Path

MAC = '''#!/bin/bash
set -eu
PACKAGE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PAGE="$PACKAGE_DIR/demo/annotate.html"
if [ ! -f "$PAGE" ]; then
  printf '%s\\n' 'Please extract the entire ZIP before starting.'
  read -r -p 'Press Enter to close.' unused || true
  exit 1
fi
/usr/bin/open "$PAGE"
'''
WINDOWS = '@echo off\r\nsetlocal\r\nif not exist "%~dp0demo\\annotate.html" (\r\n  echo Please extract the entire ZIP before starting.\r\n  pause\r\n  exit /b 1\r\n)\r\nstart "" "%~dp0demo\\annotate.html"\r\n'
START = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta http-equiv="refresh" content="0;url=demo/annotate.html">
<title>打开商品包装标注</title>
<h1>商品包装标注</h1>
<p>如果没有自动进入，<a href="demo/annotate.html">点击打开标注页面</a>。</p>
<p>请先完整解压压缩包，并保持文件夹结构不变。</p></html>
'''
GUIDE = '''Windows / macOS 标注工具启动说明

1. 先将整个ZIP完整解压到固定文件夹。不要在压缩包里直接运行或只取出启动文件。
2. 推荐两种系统均用浏览器打开 START_HERE.html；也可打开 demo/annotate.html。
   如果双击进入编辑器，右键选择“打开方式”，选择 Safari、Chrome 或 Edge。
3. Windows也可双击“启动标注.cmd”。
4. Mac也可双击“启动标注_Mac.command”，终端会调用默认浏览器打开页面。
   如提示无执行权限或系统阻止脚本，请直接用浏览器打开 START_HERE.html，
   不必更改系统安全设置。无需 Python、CUDA、模型、管理员权限或联网。
5. 若浏览器限制本地文件访问，请换 Chrome/Edge；不要关闭浏览器安全保护。

保存与继续
每次结束前点击“导出标注文件”，确认JSON出现在下载目录。
迁移文件夹或更换浏览器前先导出；浏览器自动草稿不是可交付文件。
新包与旧包路径不同，旧草稿不会自动迁移。继续已有任务时，导入旧包导出的JSON。
导入会替换当前草稿，请先导出备份。把最新JSON及已检查图片范围发回负责人。

本包修复启动兼容性，原有数据和标注页面保留。它是离线人工标注工具，
不会运行自动识别模型；具体标注能力以本包页面为准。
企业数据仅限项目组内部交接，不上传公开仓库。
'''

def write_launchers(destination):
    destination = Path(destination)
    (destination / '启动标注.cmd').write_bytes(WINDOWS.encode('ascii'))
    mac = destination / '启动标注_Mac.command'
    mac.write_bytes(MAC.encode('utf-8'))  # LF, no BOM: required for the shebang.
    mac.chmod(0o755)
    (destination / 'START_HERE.html').write_text(START, encoding='utf-8')
    (destination / '启动说明_Windows_Mac.txt').write_text(GUIDE, encoding='utf-8-sig')

def write_archive(destination, archive):
    destination, archive = Path(destination), Path(archive)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(destination.rglob('*')):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo.from_file(path, path.relative_to(destination.parent).as_posix())
            info.create_system = 3
            mode = 0o755 if path.suffix == '.command' else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            z.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
