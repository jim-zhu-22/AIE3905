import argparse
from pathlib import Path

import gradio as gr
from PIL import Image

from .infer import Predictor


def launch(checkpoint: Path, port: int):
    predictor = Predictor(checkpoint)

    def classify(image: Image.Image):
        if image is None:
            raise gr.Error("请上传一张已裁剪的材质区域图像。")
        result = predictor.predict(image)
        summary = f"预测材质：{result['label']}  |  模型分数：{result['score']:.3f}  |  版本：{result['model_version']}"
        return summary, result["scores"]

    with gr.Blocks(title="表面材质识别 POC") as app:
        gr.Markdown("# 表面材质识别 POC\n上传单个接触面的 RGB ROI。当前仅识别五类，分数不等于真实正确概率。")
        image = gr.Image(type="pil", label="材质区域 ROI")
        button = gr.Button("识别")
        summary = gr.Textbox(label="预测结果")
        scores = gr.Label(label="五类模型分数", num_top_classes=5)
        button.click(classify, inputs=image, outputs=[summary, scores])
    app.launch(server_name="127.0.0.1", server_port=port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/minc_resnet18/best.pt"))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(args.checkpoint, args.port)


if __name__ == "__main__":
    main()
