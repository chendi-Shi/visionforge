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
from search_jobs import SearchEngine, DEFAULT_SEARCH_MODEL
from search_report import export_search
from core_grounding import COVERAGE_NOTE

STATUS = {
    "queued_prepare": "等待预处理", "preparing": "正在抽帧", "ready": "待开始分析",
    "queued_analysis": "等待分析", "running": "正在分析", "pause_requested": "本段结束后暂停",
    "paused": "已暂停", "completed": "分析完成", "partial": "部分失败，可继续重试",
    "failed": "失败", "pending": "待分析", "ok": "有结果",
}
REVIEWS = {"未复核": "unreviewed", "确认": "confirmed", "修正": "corrected", "排除": "rejected"}
REVERSE_REVIEWS = {value: key for key, value in REVIEWS.items()}
SEARCH_STATUS = {"queued": "等待处理", "running": "检索 / 核验中", "pause_requested": "保存当前帧后暂停",
                 "paused": "已暂停", "completed": "核验完成", "partial": "部分核验失败", "failed": "处理失败"}
VERDICT = {"visible": "本帧可见", "not_visible": "本帧未见", "uncertain": "无法确定", "failed": "核验失败", "pending": "待核验"}
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
        selected = next((name for name in (DEFAULT_SEARCH_MODEL, DEFAULT_MODEL) if name in models), models[0] if models else None)
        return gr.Dropdown(choices=models, value=selected), f"本机 Ollama · {len(models)} 个视觉模型"
    except Exception as exc:
        return gr.Dropdown(choices=[], value=None), f"Ollama 连接失败：{exc}"


def build_app(engine: VideoEngine | None = None, searches: SearchEngine | None = None):
    engine = engine or VideoEngine(DATA_ROOT)
    engine.start()
    searches = searches or SearchEngine(engine)
    searches.start()

    def choices():
        return [(f"{j['name']} · {STATUS.get(j['status'], j['status'])} · {j['id'][:8]}", j["id"])
                for j in engine.list_jobs()]

    def initialize():
        options = choices()
        dropdown, connection = refresh_models()
        selected_job = options[0][1] if options else None
        return gr.Dropdown(choices=options, value=selected_job), dropdown, connection, history(selected_job)

    def create(source, model, focus, budget, edge, tokens):
        if not source:
            raise gr.Error("请先选择一个视频。")
        try:
            engine.start()
            job_id = engine.create_job(source, model, focus, int(budget), int(edge), int(tokens))
            return gr.Dropdown(choices=choices(), value=job_id), "已创建视频作业。预处理完成后可直接搜索画面，无需先运行摘要。"
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
                     ("已准备" if video.get("preview_path") else STATUS.get(job["status"], job["status"]), "视频预处理"),
                     (len(searches.list_searches(job_id)), "该视频的检索任务"),
                     (f"{count} / {len(segments)}", "旧版摘要片段（可选）")]
            metrics = "<div class='metrics'>" + "".join(
                f"<div class='metric'><b>{html.escape(str(v))}</b><span>{k}</span></div>" for v, k in stats) + "</div>"
            video_message = "预览已准备，可直接输入目标检索画面。" if video.get("preview_path") else job_message(job)
            metrics += "<div class='job-message'>" + html.escape(video_message) + "</div>"
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

    def search_choices(job_id):
        return [(f"{s['target'][:32]} · {SEARCH_STATUS.get(s['status'], s['status'])} · {s['id'][:8]}", s["id"])
                for s in searches.list_searches(job_id)] if job_id else []

    def history(job_id):
        options = search_choices(job_id)
        return gr.Dropdown(choices=options, value=options[0][1] if options else None)

    def start_search(job_id, target, english, selected_model, top_k, radius, step, frame_budget, search_edge, search_tokens):
        if not job_id:
            raise gr.Error("先导入并完成视频预处理。")
        try:
            searches.start()
            search_id = searches.create_search(job_id, target, retrieval_query=english or "", model=selected_model,
                top_k=int(top_k), radius_seconds=float(radius), step_seconds=float(step), max_frames=int(frame_budget),
                max_edge=int(search_edge), max_tokens=int(search_tokens))
            return gr.Dropdown(choices=search_choices(job_id), value=search_id), "已排队：画面向量检索 → 候选附近抽帧 → 逐帧核验。"
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def search_action(search_id, action):
        if not search_id:
            raise gr.Error("先选择检索任务。")
        try:
            searches.start()
            data = searches.pause(search_id) if action == "pause" else searches.resume(search_id)
            return data.get("message", "")
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def search_view(job_id, search_id, old_key):
        blank = ("<p class='subtle'>输入要找的可见对象，例如“红牛涂装卡车”，从视频画面检索候选后逐帧核验。</p>",
                 [], {"id": "", "frames": []}, [], {"id": "", "frames": []}, [], [], "", {}, "")
        if not search_id or not job_id:
            return blank
        try:
            data = searches.get_search(search_id)
            if data["job_id"] != job_id:
                return blank
            key = f"{search_id}|{data.get('updated_at')}"
            if key == old_key:
                return tuple(gr.skip() for _ in blank)
            results = data.get("results") or []
            ok = sum(r.get("status") == "ok" for r in results)
            visible = sum((r.get("structured") or {}).get("verdict") == "visible" and r.get("status") == "ok" for r in results)
            unknown = sum((r.get("structured") or {}).get("verdict") == "uncertain" for r in results)
            metrics = [(SEARCH_STATUS.get(data['status'], data['status']), "检索状态"),
                       (f"{len(results)} / {len(data.get('frames') or [])}", "已核验 / 计划帧"),
                       (visible, "模型判定本帧可见"), (unknown, "无法确定")]
            summary = "<div class='metrics'>" + ''.join(f"<div class='metric'><b>{html.escape(str(value))}</b><span>{label}</span></div>" for value, label in metrics) + "</div>"
            summary += "<p class='job-message'>" + html.escape(data.get("error") or data.get("message") or "") + "</p>"
            summary += f"<p class='subtle'>结构合格 {ok} 帧；结构合格不代表判断正确。" + html.escape(COVERAGE_NOTE) + " 本流程不分析声音。</p>"
            candidates = data.get("candidates") or []
            candidate_gallery = [(f["path"], f"#{index+1} · {clock(f['time_seconds'])} · 相似度 {f.get('score',0):.3f}") for index, f in enumerate(candidates)]
            frames = data.get("frames") or []
            by_id = {r["frame"]["id"]: r for r in results}
            evidence_gallery, rows = [], []
            for frame in frames:
                result = by_id.get(frame["id"]) or {}
                structured = result.get("structured") or {}
                verdict = structured.get("verdict") if result.get("status") == "ok" else result.get("status", "pending")
                reason = structured.get("reason") or result.get("error") or ""
                evidence_gallery.append((frame["path"], f"{clock(frame['time_seconds'])} · {VERDICT.get(verdict, verdict)} · {frame['id']}"))
                rows.append([frame["id"], clock(frame["time_seconds"]), VERDICT.get(verdict, verdict), reason])
            intervals = data.get("intervals") or {}
            interval_rows = [["可见采样组（非连续事件）", clock(g["evidence_start_seconds"]), clock(g["evidence_end_seconds"]), ', '.join(g["frame_ids"])] for g in intervals.get("visible_groups", [])]
            for field, label in (("appearance_brackets", "未见 → 可见"), ("disappearance_brackets", "可见 → 未见")):
                interval_rows += [[label, clock(b["after_seconds"]), clock(b["by_seconds"]), b["negative_frame_id"] + ' / ' + b["positive_frame_id"]] for b in intervals.get(field, [])]
            translation = data.get("translation") or {}
            translated = f"实际检索词：{translation.get('query') or '等待生成'}\n逐帧核验目标：{translation.get('visual_target') or data.get('target') or '等待生成'}\n来源：{translation.get('source') or '尚未开始'}\n可修改上方目标或英文词后创建新检索；已有结果保留。"
            return summary, candidate_gallery, {"id": search_id, "frames": candidates}, evidence_gallery, {"id": search_id, "frames": frames}, rows, interval_rows, translated, data, key
        except Exception as exc:
            return ("<p>" + html.escape(str(exc)) + "</p>",) + blank[1:]

    def search_frame_time(search_id, frame_state, evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        if not search_id or frame_state.get("id") != search_id or not isinstance(index, int) or not 0 <= index < len(frame_state["frames"]):
            raise gr.Error("检索已切换，请重新选择证据。")
        frame = frame_state["frames"][index]
        data = searches.get_search(search_id)
        detail = next((r for r in data.get("results", []) if r.get("frame", {}).get("id") == frame["id"]), {"frame": frame, "note": "候选或待核验帧，不是已确认事件。"})
        return frame["time_seconds"], detail

    def interval_time(rows, evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        if not isinstance(index, int) or not 0 <= index < len(rows):
            return 0
        hours, minutes, seconds = rows[index][1].split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    def search_export(search_id):
        if not search_id:
            raise gr.Error("请先选择检索任务。")
        try:
            return export_search(searches.get_search(search_id), engine.root / "exports")
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="VisionForge · 视频证据工作台", analytics_enabled=False) as demo:
        gr.HTML("<div id='hero'><div class='eyebrow'>VISIONFORGE / LOCAL VISUAL SEARCH</div>"
                "<h1>先找画面，再逐帧核验</h1>"
                "<p>输入可见目标 → 画面向量检索 → 候选附近加密抽帧 → 本地 VLM 核验。时间来自实际帧，结论可以回到视频检查。</p></div>")
        selected_segment = gr.State("")
        preview_key = gr.State("")
        search_key = gr.State("")
        candidate_state = gr.State({"id": "", "frames": []})
        evidence_state = gr.State({"id": "", "frames": []})
        with gr.Row():
            with gr.Column(scale=1, min_width=285):
                job_picker = gr.Dropdown(choices=choices(), label="作业库", interactive=True)
                reload_jobs = gr.Button("刷新作业库", size="sm")
                model = gr.Dropdown(choices=[], label="本地视觉核验模型")
                refresh = gr.Button("刷新模型", size="sm")
                connection = gr.Textbox(label="连接", interactive=False)
                with gr.Accordion("导入视频", open=not bool(choices())):
                    upload = gr.File(label="本地视频 · 最大 1 GB / 30 分钟", file_types=[".mp4", ".webm", ".mov", ".mkv", ".avi"], type="filepath")
                    with gr.Accordion("旧版摘要设置（搜索无需运行摘要）", open=False):
                        focus = gr.Textbox(label="摘要观察目标", value=FOCUS, lines=3)
                        budget = gr.Slider(2, 12, value=4, step=1, label="旧版最多分析窗口数")
                        edge = gr.Radio([320, 448, 768], value=448, label="送入模型的帧最长边")
                        tokens = gr.Slider(256, 768, value=512, step=128, label="每段最大输出 tokens")
                    prepare = gr.Button("导入并准备预览", variant="primary")
                    sample_button = gr.Button("载入公开街景样片 · 35 秒")
                    gr.Markdown("样片：Editor / [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Street_traffic.webm)")
                notice = gr.Textbox(label="操作反馈", lines=3, interactive=False)
                gr.Markdown("CPU 核验逐帧排队，关闭页面不影响后台。首次画面检索会下载公开 CLIP 权重，后续复用本机索引。视频与核验请求保留在本机。")
            with gr.Column(scale=3, min_width=650):
                metrics = gr.HTML()
                player = gr.Video(label="视频预览 · 与证据时间一致", elem_id="source-player", interactive=False, height=360)
                target = gr.Textbox(label="想找什么可见目标？", placeholder="例如：红牛涂装卡车、骑自行车的人、红色公交车", value="红牛涂装卡车", lines=1)
                english = gr.Textbox(label="英文画面检索词（可选）", placeholder="例如：a Red Bull delivery truck。留空由本地模型翻译，实际词会显示在下方。", lines=1)
                with gr.Accordion("检索范围与 CPU 预算", open=False):
                    gr.Markdown("CLIP 每约 2 秒检索一帧，再在候选附近核验。相似度用于排序，不是检测置信度。")
                    with gr.Row():
                        top_k = gr.Slider(1, 4, value=2, step=1, label="候选位置数")
                        radius = gr.Slider(1, 6, value=2, step=1, label="候选前后范围（秒）")
                        step = gr.Radio([0.5, 1, 2], value=1, label="核验采样步长（秒）")
                    with gr.Row():
                        frame_budget = gr.Slider(3, 48, value=12, step=1, label="最多核验帧数")
                        search_edge = gr.Radio([320, 448, 640, 768], value=448, label="核验帧最长边")
                        search_tokens = gr.Slider(96, 512, value=160, step=32, label="单帧输出预算")
                with gr.Row():
                    search_start = gr.Button("检索并逐帧核验", variant="primary")
                    search_pause = gr.Button("当前帧后暂停")
                    search_resume = gr.Button("继续未完成核验")
                search_picker = gr.Dropdown(label="该视频的检索历史", interactive=True, choices=[])
                search_summary = gr.HTML()
                translation_text = gr.Textbox(label="实际使用的检索词与核验目标", lines=4, interactive=False)
                candidates_gallery = gr.Gallery(label="视觉相似度候选 · 排名不等于目标出现", columns=2, rows=1, height=230, interactive=False, allow_preview=False)
                evidence_gallery = gr.Gallery(label="逐帧核验 · 点击图片定位视频", columns=3, height=320, interactive=False, allow_preview=False)
                with gr.Row():
                    seek = gr.Number(label="定位到视频秒数", value=0, precision=3)
                    jump = gr.Button("定位画面")
                result_table = gr.Dataframe(headers=["证据帧", "真实采样时间", "本帧结论", "依据 / 失败原因"], datatype="str", type="array", interactive=False, wrap=True, max_height=300)
                interval_table = gr.Dataframe(headers=["证据类型", "前一 / 首个采样点", "后一 / 末个采样点", "证据帧 ID"], datatype="str", type="array", interactive=False, wrap=True, max_height=240)
                gr.Markdown("**时间证据只约束采样点。** 可见采样组不证明中间持续可见；出现 / 消失区间不是事件精确边界；不确定帧和较大空隙会阻断推导。未检索中的位置没有作出否定结论。不分析声音。")
                search_export_button = gr.Button("导出检索证据包 · 离线 HTML / JSON / CSV")
                search_downloads = gr.File(label="下载 ZIP / JSON / CSV（ZIP 解压后打开 report.html）", file_count="multiple", interactive=False)
                with gr.Accordion("单帧原始核验记录", open=False):
                    frame_record = gr.JSON(label="选中帧的响应、哈希与模型身份")
                with gr.Accordion("完整检索记录、翻译与重试历史", open=False):
                    search_record = gr.JSON(label="可审计记录")
                with gr.Accordion("旧版 · 稀疏窗口摘要与人工复核", open=False):
                    coverage = gr.HTML()
                    with gr.Row():
                        analyze = gr.Button("开始 / 继续旧版摘要")
                        pause = gr.Button("本段结束后暂停摘要")
                        export_button = gr.Button("导出摘要证据包")
                    search = gr.Textbox(label="按已生成摘要文字筛选（字符串匹配）", lines=1)
                    table = gr.Dataframe(headers=["片段", "窗口起点", "窗口终点", "状态", "观察摘要", "复核"], datatype="str", type="array", interactive=False, wrap=True, max_height=320, elem_id="timeline")
                    gallery = gr.Gallery(label="旧版采样帧", columns=2, rows=1, height=270, interactive=False, allow_preview=False)
                    observations = gr.Textbox(label="观察与证据引用", lines=6, interactive=False)
                    decision = gr.Radio(list(REVIEWS), value="未复核", label="人工复核结论")
                    note = gr.Textbox(label="修正内容 / 备注", lines=3)
                    save = gr.Button("保存复核")
                    downloads = gr.File(label="下载摘要 ZIP / JSON / CSV", file_count="multiple", interactive=False)
                    record = gr.JSON(label="旧版片段记录")
        gr.Markdown("基于 [Ollama](https://github.com/ollama/ollama-python)、[PySceneDetect](https://github.com/Breakthrough/PySceneDetect)、"
                    "[FFmpeg](https://ffmpeg.org/) 与 [Gradio](https://github.com/gradio-app/gradio)。"
                    "数据默认保存在本机 LocalAppData/VisionForge/video-data，可通过 VISIONFORGE_DATA_DIR 设置。")
        refresh.click(refresh_models, outputs=[model, connection], api_name="refresh_models")
        reload_jobs.click(lambda: gr.Dropdown(choices=choices()), outputs=[job_picker], queue=False)
        job_picker.change(history, [job_picker], [search_picker], queue=False)
        search_start.click(start_search, [job_picker, target, english, model, top_k, radius, step, frame_budget, search_edge, search_tokens],
                           [search_picker, notice], api_name="create_visual_search")
        search_pause.click(lambda s: search_action(s, "pause"), [search_picker], [notice], api_name="pause_visual_search")
        search_resume.click(lambda s: search_action(s, "resume"), [search_picker], [notice], api_name="resume_visual_search")
        search_export_button.click(search_export, [search_picker], [search_downloads], api_name="export_visual_search")
        search_inputs = [job_picker, search_picker, search_key]
        search_outputs = [search_summary, candidates_gallery, candidate_state, evidence_gallery, evidence_state, result_table, interval_table, translation_text, search_record, search_key]
        search_picker.change(search_view, search_inputs, search_outputs, queue=False)
        candidates_gallery.select(search_frame_time, [search_picker, candidate_state], [seek, frame_record]).then(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        evidence_gallery.select(search_frame_time, [search_picker, evidence_state], [seek, frame_record]).then(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        result_table.select(search_frame_time, [search_picker, evidence_state], [seek, frame_record]).then(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        interval_table.select(interval_time, [interval_table], [seek]).then(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        prepare.click(create, [upload, model, focus, budget, edge, tokens], [job_picker, notice], api_name="create_video_job")
        sample_button.click(sample, [model, focus, budget, edge, tokens], [job_picker, notice], api_name="add_sample")
        analyze.click(lambda j: act(j, "analyze"), [job_picker], [notice], api_name="start_analysis")
        pause.click(lambda j: act(j, "pause"), [job_picker], [notice], api_name="pause_analysis")
        export_button.click(export, [job_picker], [downloads], api_name="export_report")
        for trigger in (job_picker.change, search.change):
            trigger(view, [job_picker, search, preview_key], [metrics, coverage, table, player, preview_key], queue=False)
        timer = gr.Timer(3)
        timer.tick(view, [job_picker, search, preview_key], [metrics, coverage, table, player, preview_key], queue=False, show_progress="hidden")
        timer.tick(search_view, search_inputs, search_outputs, queue=False, show_progress="hidden")
        table.select(show_segment, [job_picker, table], [selected_segment, gallery, observations, record, seek, decision, note]).then(
            fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        gallery.select(frame_time, [job_picker, selected_segment], [seek]).then(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        jump.click(fn=None, inputs=[seek], outputs=[], js=SEEK_JS)
        save.click(review, [job_picker, selected_segment, decision, note], [notice], api_name="save_review")
        demo.load(initialize, outputs=[job_picker, model, connection, search_picker])
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=32, default_concurrency_limit=2).launch(
        server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False,
        max_file_size="1gb", allowed_paths=[str(DATA_ROOT)], css=CSS,
    )

