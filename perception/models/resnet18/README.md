材质分类：model.py 网络；preprocess.py 预处理；labels.py 类别；backend.py 权重和推理。训练和评估入口仍在 material_recognition。

新一轮按帧调参入口：`python -m perception.models.resnet18.tune --help`。

该入口从未见现场样本的 MINC 权重开始每个验证折，比较分类头、layer4、全网络微调，并加入公开训练样本回放。模型选择使用三折验证；原最后一帧在选择完成后只做历史回归。自动候选不作为训练标签。完整参数、数据快照、权重哈希、每轮损失和选择依据均保存在输出目录。

训练需要 `scikit-learn`，可通过 `pip install -e '.[training]'` 安装。生产推理仍使用 backend.py，不依赖调参模块。
