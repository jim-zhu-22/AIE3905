"""Refresh the project summary using saved metrics, without invented results."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
metrics = json.loads((root / 'runs/week1_resnet18/evaluation/metrics.json').read_text())
accuracy = metrics['accuracy']
f1 = metrics['classification']['macro avg']['f1-score']
path = root / 'README.md'
text = path.read_text(encoding='utf-8')
start = text.index('已建立第一周的可运行基线：')
end = text.index('\n\n| 类别编号', start)
summary = (f'第一周已下载五类 MINC 全集 12,500 张，完成 ResNet-18 分类头 2 轮和全网络微调 1 轮。'
           f'625 张验证图上的准确率为 **{accuracy:.2%}**，Macro-F1 为 **{f1:.4f}**；验证集参与选模，'
           '保留的 1,250 张测试图尚未评估。完整基线使用 Python 3.11 CPU；已有 Python 3.12 CUDA 环境'
           '也完成了 200 张训练、50 张验证的独立 GPU 演示（准确率 54%，Macro-F1 0.5235）。'
           '显卡目前连续运算较慢，详见 `docs/gpu_environment.md`。企业数据尚未收到，现场基线与新 SKU 测试未完成。'
           '完整周报见 `docs/week1_report.md` 和 `deliverables/第一周周报.docx`。\n\n'
           '展示完整基线：\n\n```powershell\n'
           '.\\.venv\\Scripts\\python.exe -m material_recognition.demo --checkpoint runs/week1_resnet18/best.pt\n'
           '```\n\n'
           '复现同样训练预算：\n\n```powershell\n'
           '.\\.venv\\Scripts\\python.exe -m material_recognition.train --device cpu --head-epochs 2 --full-epochs 1 --output-dir runs/week1_reproduction\n'
           '```\n\n实际本次运行分为分类头训练与从最佳权重初始化的微调两步；历史和配置保留在 `runs/week1_cpu_resnet18` 与 `runs/week1_resnet18`。')
path.write_text(text[:start] + summary + text[end:], encoding='utf-8')
(root / 'docs/week1.md').write_text('# 第一周执行记录\n\n完整周报见 [week1_report.md](week1_report.md)。\n\n'
    f'公开数据基线验证准确率 {accuracy:.2%}，Macro-F1 {f1:.4f}，使用 CPU 完成全量训练。'
    '独立 GPU 小样本训练已完成；环境说明见 [gpu_environment.md](gpu_environment.md)。\n\n'
    '第一周公开线交付完成。企业图像与标注导出样例尚未收到，企业数据接入、SKU 测试与双模型切换未完成。\n', encoding='utf-8')
print('Updated README and first-week execution record')
