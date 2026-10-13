"""Local image-understanding workbench built on Gradio and ollama-python."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / ".runtime" / "uploads")

import gradio as gr

from core import DEFAULT_MODEL, analyze_image, list_vision_models

PRESETS = {
    "图片问答": "请用中文简要描述这张图片：有哪些主要对象，它们在做什么？只说图中可见的内容。",
    "读取文字": "请按阅读顺序逐行抄录图片中清晰可见的文字，保留数字和单位。看不清的部分写[无法辨认]，不要补写。",
    "理解图表": "请说明图表的标题、坐标轴、单位、主要趋势和可见数值。区分直接读数与推测；无法看清的数值不要猜。",
    "结构化提取": "请用中文提取图片摘要、可见文字、主要对象以及无法确定的信息。",
}


def refresh_models():
    try:
        models = list_vision_models()
        selected = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
        message = f"已连接本机 Ollama · {len(models)} 个本地视觉模型" if models else "Ollama 已连接，尚无本地视觉模型。请运行：ollama pull qwen3-vl:2b，然后刷新。"
        return gr.Dropdown(choices=models, value=selected), message
    except Exception as exc:
        return gr.Dropdown(choices=[], value=None), f"无法连接：{exc}。请启动 Ollama 后刷新。"


def preset_question(task):
    return PRESETS[task]


def render_result(result):
    structured = result.get("structured")
    if structured:
        # Textbox avoids interpreting recognized text as executable HTML/Markdown.
        sections = [structured.get("summary", "")]
        for key, title in [("visible_text", "可见文字"), ("objects", "主要对象"), ("uncertainties", "不确定信息")]:
            sections.append(title + "：\n" + "\n".join("• " + str(x) for x in structured.get(key, [])))
        return "\n\n".join(sections)
    return result.get("answer") or result.get("raw_response", "")


def run_analysis(image_path, task, question, model, max_edge, max_tokens):
    if not image_path:
        raise gr.Error("请先上传一张图片。")
    if not model:
        raise gr.Error("请先下载视觉模型并点击“刷新模型”。")
    try:
        result = analyze_image(
            image_path, question, model=model,
            mode="extract" if task == "结构化提取" else "qa",
            max_edge=int(max_edge), max_tokens=int(max_tokens),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    status = result.get("status", "unknown")
    labels = {"ok": "完成", "truncated": "输出达到上限，结果不完整；可增加输出长度", "invalid_output": result.get("error") or "模型输出无效，请查看原始输出或重试"}
    elapsed = result.get("elapsed_seconds", 0)
    summary = f"{labels.get(status, status)} · {model} · {elapsed:.1f} 秒"
    output_dir = ROOT / ".runtime" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / uuid4().hex
    json_path = stem.with_suffix(".json")
    text_path = stem.with_suffix(".txt")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render_result(result)
    text_path.write_text(summary + "\n\n" + text, encoding="utf-8")
    return summary, text, result, [str(json_path), str(text_path)]


def build_app():
    with gr.Blocks(title="VisionForge · 本地图片理解", analytics_enabled=False) as demo:
        gr.Markdown("# VisionForge · 本地图片理解\n上传图片，用本机 Ollama 提问、读文字或提取结构化信息。")
        with gr.Row():
            with gr.Column(scale=1):
                picture = gr.Image(label="图片 / 截图", type="filepath", image_mode=None, format="png", sources=["upload", "clipboard"], height=350)
                task = gr.Radio(list(PRESETS), value="图片问答", label="任务")
                question = gr.Textbox(value=PRESETS["图片问答"], label="你的问题", lines=3, max_lines=6)
                with gr.Accordion("模型与速度", open=True):
                    model = gr.Dropdown(choices=[], label="本地视觉模型")
                    refresh = gr.Button("刷新模型", size="sm")
                    connection = gr.Textbox(label="连接状态", interactive=False)
                    edge = gr.Radio([448, 768, 1024], value=768, label="图片最长边（像素）", info="更小更快；小字和细节可能丢失。")
                    tokens = gr.Slider(64, 1024, value=384, step=64, label="最大输出 tokens")
                submit = gr.Button("开始理解", variant="primary")
                gr.Markdown("CPU 推理可能需要几十秒到数分钟。任务按顺序执行。")
            with gr.Column(scale=1):
                status = gr.Textbox(label="运行状态", interactive=False)
                answer = gr.Textbox(label="理解结果", lines=17, interactive=False)
                files = gr.File(label="下载结果（JSON / TXT）", file_count="multiple", interactive=False)
                with gr.Accordion("原始输出与运行记录", open=False):
                    record = gr.JSON(label="可复核记录")
        with gr.Accordion("试一试自制样例", open=False):
            gr.Markdown("这些图用于验证流程，不代表真实图片准确率。")
            examples_dir = ROOT / "examples"
            samples = sorted(examples_dir.glob("*.png"))
            if samples:
                gr.Examples([[str(path)] for path in samples], inputs=[picture])
        gr.Markdown(
            "模型可能读错小字、数量与图表，请核对原图。图片推理只调用本机地址，拒绝云模型。"
            "临时图片和导出结果存于项目 `.runtime`；若项目位于 OneDrive，文件仍受系统同步设置影响。\n\n"
            "复用 [Ollama 官方 SDK](https://github.com/ollama/ollama-python) · "
            "[Gradio](https://github.com/gradio-app/gradio) · "
            "[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)"
        )
        task.change(preset_question, [task], [question])
        refresh.click(refresh_models, outputs=[model, connection])
        submit.click(run_analysis, [picture, task, question, model, edge, tokens], [status, answer, record, files], concurrency_limit=1)
        demo.load(refresh_models, outputs=[model, connection])
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, share=False,
        inbrowser=False, max_file_size="20mb",
    )
