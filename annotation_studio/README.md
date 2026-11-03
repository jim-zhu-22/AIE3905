# 本地标注工作区

此目录保留独立标注工具的源码布局。主仓库的在线感知工作台和材质演示分别位于 API 服务的 `/` 和 `/demo`。

在 Python 3.12 虚拟环境中，从本目录安装 `python -m pip install -r requirements-demo.txt`，再进入 `02-代码` 执行 `python -m rgbd_perception.annotation_service --help`。从该目录执行 `python -m rgbd_perception.annotation_service --paused` 可启动本地审核服务。模型文件的准备和自动推理的操作见[标注说明](02-代码/rgbd_perception/ANNOTATION_DEMO.md)。首次使用请自行提供已授权的图像和模型。根目录的双击启动器用于带 Python 运行时与权重的本地交付包。

在 `02-代码` 下运行 `python -m unittest discover -s tests` 验证标注和数据处理，运行 `node --test tests/test_annotation_export.cjs tests/test_portable_annotation.cjs` 验证导出。
