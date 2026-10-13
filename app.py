"""Local video evidence workbench. Inference runs in the persistent worker."""
from __future__ import annotations

import html
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from video_jobs import VideoEngine, default_data_dir

ROOT = Path(__file__).resolve().parent
DATA_ROOT = default_data_dir()
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TEMP_DIR"] = str(DATA_ROOT / "uploads")

import gradio as gr

from core import DEFAULT_MODEL, list_vision_models
from video_report import clock, effective_summary, export_job

STATUS = {
    "queued_prepare": "等待预处理", "preparing": "正在抽帧", "ready": "待开始分析",
    "queued_analysis": "等待分析", "running": "正在分析", "pause_requested": "本段结束后暂停",
    "paused": "已暂停", "completed": "分析完成", "partial": "部分失败，可继续重试",
    "failed": "失败", "pending": "待分析", "ok": "有结果",
}
REVIEWS = {"未复核": "unreviewed", "确认": "confirmed", "修正": "corrected", "排除": "rejected"}
REVERSE_REVIEWS = {value: key for key, value in REVIEWS.items()}
FOCUS = "记录画面中的主要对象与可见变化，重点关注车辆、行人及其在两张采样画面中的位置变化。无法确认同一对象时明确说明。"
SEEK_JS = """(seconds) => {
 const v = document.querySelector('#source-player video');
 if (v) { v.currentTime = Math.max(0, Number(seconds) || 0); v.pause();
 v.scrollIntoView({behavior:'smooth', block:'center'}); }
 return [];
}"""
CSS = """
.gradio-container {max-width:1480px!important; background:#f5f7fa;}
#hero {background:#102737;color:white;padding:26px 30px;border-radius:16px;margin-bottom:16px;}
#hero h1 {color:white;font-size:28px;margin:4px 0 10px;}
#hero p {color:#c2d7e2;margin:0;line-height:1.7;}
.eyebrow {color:#56d3b0;font:600 11px monospace;letter-spacing:2px;}
.metrics {display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px;}
.metric {background:white;border:1px solid #dce5eb;border-radius:12px;padding:15px;}
.metric b {display:block;font-size:21px;color:#153446;margin-bottom:5px;}
.metric span,.subtle {font-size:12px;color:#63798b;}
.coverage {display:flex;height:12px;gap:2px;border-radius:6px;overflow:hidden;margin:12px 0 6px;}
.coverage span {background:#d8e0e8;min-width:2px;}
.coverage span.selected {background:#199d84;}
.job-message {padding:10px 0;color:#365368;font-size:13px;overflow-wrap:anywhere;}
#timeline {font-size:13px;}
"""


def job_message(job):
    if job.get("error"):
        return job["error"]
    status = job["status"]
    if status in {"running", "pause_requested"}:
        segment = next((s for s in job.get("segments", []) if s["status"] == "running"), None)
        prefix = f"正在分析 {segment['id']} · 第 {segment['attempts']} 次尝试" if segment else "等待当前分析结束"
        elapsed = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(job["updated_at"])).total_seconds()))
        suffix = "；本段保存后暂停" if status == "pause_requested" else ""
        return f"{prefix} · 当前状态已持续 {elapsed} 秒{suffix}"
    if status == "preparing":
        stage = job.get("message", "").split(":", 1)[0]
        return {"probe": "核对视频文件与哈希", "normalize": "生成时间轴一致的无声预览", "scenes": "检测镜头变化并生成候选窗口", "frames": "提取带时间戳的证据帧"}.get(stage, "正在准备视频")
    return {"ready": "抽帧已完成。请检查采样范围，然后开始分析。", "queued_prepare": "等待预处理。",
            "queued_analysis": "等待分析，已完成片段将直接复用。", "paused": "已保存当前进度，可继续未完成片段。",
            "completed": "所有选定窗口均有模型结果，内容仍需结合证据复核。",
            "partial": "部分片段失败。原始响应已保存，可继续重试失败片段。"}.get(status, job.get("message", ""))


def refresh_models():
    try:
        models = list_vision_models()
        selected = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
        return gr.Dropdown(choices=models, value=selected), f"本机 Ollama · {len(models)} 个视觉模型"
    except Exception as exc:
        return gr.Dropdown(choices=[], value=None), f"Ollama 连接失败：{exc}"


def build_app(engine: VideoEngine | None = None):
    engine = engine or VideoEngine(DATA_ROOT)
    engine.start()

    def choices():
        return [(f"{j['name']} · {STATUS.get(j['status'], j['status'])} · {j['id'][:8]}", j["id"])
                for j in engine.list_jobs()]

    def initialize():
        options = choices()
        dropdown, connection = refresh_models()
        return gr.Dropdown(choices=options, value=options[0][1] if options else None), dropdown, connection

    def create(source, model, focus, budget, edge, tokens):
        if not source:
            raise gr.Error("请先选择一个视频。")
        try:
            engine.start()
            job_id = engine.create_job(source, model, focus, int(budget), int(edge), int(tokens))
            return gr.Dropdown(choices=choices(), value=job_id), "已创建作业。抽帧完成后检查采样范围，再开始分析。"
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def sample(model, focus, budget, edge, tokens):
        path = ROOT / "examples" / "video" / "street-traffic.webm"
        if not path.is_file():
            raise gr.Error(r"公开样片按需下载。请先在项目目录运行：.\.venv\Scripts\python.exe tools/fetch_video_sample.py")
        return create(str(path), model, focus, budget, edge, tokens)

    def act(job_id, action):
        if not job_id:
            raise gr.Error("请先选择作业。")
        try:
            engine.start()
            job = engine.analyze(job_id) if action == "analyze" else engine.pause(job_id)
            return job_message(job)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def view(job_id, query, old_preview):
        if not job_id:
            return "<p class='subtle'>从左侧导入视频或打开一个已有作业。</p>", "", [], gr.skip(), old_preview
        try:
            job = engine.get_job(job_id)
            video, selection = job.get("video") or {}, job.get("selection") or {}
            segments = job.get("segments") or []
            count = sum(s["status"] == "ok" for s in segments)
            stats = [(clock(video.get("duration_seconds", 0)), "视频时长"),
                     (f"{selection.get('selected_count', 0)} / {selection.get('candidate_count', 0)}", "已选 / 候选窗口"),
                     (f"{count} / {len(segments)}", "已完成片段"),
                     (STATUS.get(job["status"], job["status"]), "作业状态")]
            metrics = "<div class='metrics'>" + "".join(
                f"<div class='metric'><b>{html.escape(str(v))}</b><span>{k}</span></div>" for v, k in stats) + "</div>"
            metrics += "<div class='job-message'>" + html.escape(job_message(job)) + "</div>"
            windows = selection.get("all_windows", [])
            coverage = "<div class='coverage'>" + "".join(
                f"<span class='{'selected' if w['selected'] else ''}' style='flex:{max(.01, w['end_seconds']-w['start_seconds'])}' title='{clock(w['start_seconds'])}–{clock(w['end_seconds'])}'></span>"
                for w in windows) + "</div>"
            coverage += "<p class='subtle'>绿色：已选采样窗口；灰色：未选。每窗仅 2 帧，窗口边界不是事件起止时间。预览无音轨，本项目只分析画面。</p>"
            rows = []
            for segment in segments:
                result = segment.get("result") or {}
                searchable = [segment["id"], result.get("summary", ""), result.get("observations", []),
                              result.get("uncertainties", []), segment.get("review", {}).get("text", "")]
                if query and query.casefold() not in json.dumps(searchable, ensure_ascii=False).casefold():
                    continue
                rows.append([segment["id"], clock(segment["start_seconds"]), clock(segment["end_seconds"]),
                             STATUS.get(segment["status"], segment["status"]), effective_summary(segment),
                             REVERSE_REVIEWS.get(segment.get("review", {}).get("decision"), "未复核")])
            preview = video.get("preview_path")
            key = f"{job_id}|{preview}"
            player_update = gr.Video(value=preview or None) if key != old_preview else gr.skip()
            return metrics, coverage, rows, player_update, key
        except Exception as exc:
            return "<p>" + html.escape(str(exc)) + "</p>", "", [], gr.skip(), old_preview

    def show_segment(job_id, rows, evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        if not job_id or not isinstance(index, int) or not 0 <= index < len(rows):
            raise gr.Error("请重新选择片段。")
        segment_id = rows[index][0]
        job = engine.get_job(job_id)
        segment = next(s for s in job["segments"] if s["id"] == segment_id)
        frames = segment["frames"]
        gallery = [(f["path"], f"{f['id']} · {clock(f['time_seconds'])}") for f in frames]
        result = segment.get("result") or {}
        detail = [effective_summary(segment)]
        for item in result.get("observations", []):
            detail.append(f"[{item['category']}] {item['description']}\n证据：" + ", ".join(item["evidence_frame_ids"]))
        for uncertainty in result.get("uncertainties", []):
            detail.append("待核实：" + uncertainty)
        review = segment.get("review") or {}
        return (f"{job_id}:{segment_id}", gallery, "\n\n".join(detail), segment,
                frames[0]["time_seconds"] if frames else segment["start_seconds"],
                REVERSE_REVIEWS.get(review.get("decision"), "未复核"), review.get("text", ""))

    def frame_time(job_id, segment_id, evt: gr.SelectData):
        if not job_id or not segment_id:
            return 0
        selected_job, segment_id = segment_id.split(":", 1)
        if selected_job != job_id:
            raise gr.Error("作业已切换，请重新选择片段。")
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        segment = next(s for s in engine.get_job(job_id)["segments"] if s["id"] == segment_id)
        return segment["frames"][index]["time_seconds"]

    def review(job_id, segment_id, decision, note):
        if not job_id or not segment_id:
            raise gr.Error("先在时间线选择片段。")
        selected_job, segment_id = segment_id.split(":", 1)
        if selected_job != job_id:
            raise gr.Error("作业已切换，请重新选择片段。")
        try:
            engine.review_segment(job_id, segment_id, REVIEWS[decision], note)
            return "复核已保存。原始模型结果保留，修正意见单独记录。"
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def export(job_id):
        if not job_id:
            raise gr.Error("请先选择作业。")
        try:
            job = engine.get_job(job_id)
            attribution_path = ROOT / "examples" / "video" / "source.json"
            if attribution_path.exists():
                attribution = json.loads(attribution_path.read_text(encoding="utf-8-sig"))
                if attribution.get("original_file", {}).get("sha256") == (job.get("video") or {}).get("source_sha256"):
                    job["source_attribution"] = attribution
            return export_job(job, engine.root / "exports")
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="VisionForge · 视频证据工作台", analytics_enabled=False) as demo:
        gr.HTML("<div id='hero'><div class='eyebrow'>VISIONFORGE / LOCAL VIDEO INTELLIGENCE</div>"
                "<h1>从视频到可复核的事件时间线</h1>"
                "<p>镜头切分与抽帧 → 本地 VLM → 时间证据 → 人工复核。所有推理在这台电脑的 Ollama 上完成。</p></div>")
        selected_segment = gr.State("")
        preview_key = gr.State("")
        with gr.Row():
            with gr.Column(scale=1, min_width=285):
                job_picker = gr.Dropdown(choices=choices(), label="作业库", interactive=True)
                reload_jobs = gr.Button("刷新作业库", size="sm")
                with gr.Accordion("新作业 · 导入与分析设置", open=not bool(choices())):
                    upload = gr.File(label="本地视频 · 最大 1 GB / 30 分钟", file_types=[".mp4", ".webm", ".mov", ".mkv", ".avi"], type="filepath")
                    focus = gr.Textbox(label="观察目标", value=FOCUS, lines=3)
                    budget = gr.Slider(2, 12, value=4, step=1, label="最多分析窗口数", info="每窗 2 帧；窗口多时均匀选择，未选部分明确标灰。")
                    model = gr.Dropdown(choices=[], label="本地视觉模型")
                    refresh = gr.Button("刷新模型", size="sm")
                    connection = gr.Textbox(label="连接", interactive=False)
                    with gr.Accordion("CPU 预算", open=False):
                        edge = gr.Radio([320, 448, 768], value=448, label="送入模型的帧最长边")
                        tokens = gr.Slider(256, 768, value=512, step=128, label="每段最大输出 tokens")
                    prepare = gr.Button("1 · 导入并抽帧", variant="primary")
                    sample_button = gr.Button("载入公开街景样片 · 35 秒")
                    gr.Markdown("样片：Editor / [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Street_traffic.webm)")
                notice = gr.Textbox(label="操作反馈", lines=3, interactive=False)
                gr.Markdown("CPU 分析每段可能需要数分钟。关闭页面不影响后台作业；退出服务后可继续未完成的片段。")
            with gr.Column(scale=3, min_width=650):
                metrics = gr.HTML()
                player = gr.Video(label="视频预览 · 与证据时间一致", elem_id="source-player", interactive=False, height=360)
                coverage = gr.HTML()
                with gr.Row():
                    analyze = gr.Button("2 · 开始 / 继续分析", variant="primary")
                    pause = gr.Button("本段结束后暂停")
                    export_button = gr.Button("导出证据包")
                search = gr.Textbox(label="检索时间线", placeholder="按观察内容或人工复核文字筛选，如：车辆、行人", lines=1)
                table = gr.Dataframe(headers=["片段", "窗口起点", "窗口终点", "状态", "观察摘要", "复核"],
                                     datatype="str", type="array", interactive=False, wrap=True,
                                     max_height=320, elem_id="timeline", column_widths=[75, 105, 105, 90, 380, 70])
                gr.Markdown("**点击时间线的一行检查证据。点击证据帧可定位视频。**")
                gallery = gr.Gallery(label="实际采样帧 · 原始时间戳", columns=2, rows=1, height=270, interactive=False, allow_preview=False)
                with gr.Row():
                    seek = gr.Number(label="定位到视频秒数", value=0, precision=3)
                    jump = gr.Button("定位画面")
                observations = gr.Textbox(label="观察与证据引用", lines=6, interactive=False)
                with gr.Accordion("人工复核", open=False):
                    decision = gr.Radio(list(REVIEWS), value="未复核", label="结论")
                    note = gr.Textbox(label="修正内容 / 备注", lines=3)
                    save = gr.Button("保存复核")
                downloads = gr.File(label="下载 ZIP 证据包 / JSON / CSV", file_count="multiple", interactive=False)
                with gr.Accordion("原始响应、哈希与重试历史", open=False):
                    record = gr.JSON(label="片段记录")
        gr.Markdown("基于 [Ollama](https://github.com/ollama/ollama-python)、[PySceneDetect](https://github.com/Breakthrough/PySceneDetect)、"
                    "[FFmpeg](https://ffmpeg.org/) 与 [Gradio](https://github.com/gradio-app/gradio)。"
                    "数据默认保存在本机 LocalAppData/VisionForge/video-data，可通过 VISIONFORGE_DATA_DIR 设置。")
        refresh.click(refresh_models, outputs=[model, connection], api_name="refresh_models")
        reload_jobs.click(lambda: gr.Dropdown(choices=choices()), outputs=[job_picker], queue=False)
        prepare.click(create, [upload, model, focus, budget, edge, tokens], [job_picker, notice], api_name="create_video_job")
        sample_button.click(sample, [model, focus, budget, edge, tokens], [job_picker, notice], api_name="add_sample")
        analyze.click(lambda j: act(j, "analyze"), [job_picker], [notice], api_name="start_analysis")
        pause.click(lambda j: act(j, "pause"), [job_picker], [notice], api_name="pause_analysis")
        export_button.click(export, [job_picker], [downloads], api_name="export_report")
        for trigger in (job_picker.change, search.change):
            trigger(view, [job_picker, search, preview_key], [metrics, coverage, table, player, preview_key], queue=False)
        timer = gr.Timer(3)
        timer.tick(view, [job_picker, search, preview_key], [metrics, coverage, table, player, preview_key], queue=False, show_progress="hidden")
        table.select(show_segment, [job_picker, table], [selected_segment, gallery, observations, record, seek, decision, note]).then(
            fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        gallery.select(frame_time, [job_picker, selected_segment], [seek]).then(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        jump.click(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        save.click(review, [job_picker, selected_segment, decision, note], [notice], api_name="save_review")
        demo.load(initialize, outputs=[job_picker, model, connection])
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=32, default_concurrency_limit=2).launch(
        server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False,
        max_file_size="1gb", allowed_paths=[str(DATA_ROOT)], css=CSS,
    )

