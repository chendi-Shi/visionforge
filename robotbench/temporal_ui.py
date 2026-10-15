"""Read-only continuous-trajectory experiment viewer for published artifacts."""
from __future__ import annotations

import json
import html
from functools import lru_cache
from pathlib import Path

import gradio as gr

from robotbench.temporal_plot import render_motion_plot

NAMES = {"uniform": "等间隔 12 帧", "state_event": "状态事件 + 均匀锚点 · 12 帧",
         "clock_midpoint": "时间中点 · 全报失败", "state_peak": "最大运动变化 · 全报失败"}
HEADERS = ["方法", "正确 / 全部", "失败召回", "成功误报", "拒答", "失败定位 ±2s", "有效定位覆盖", "条件 MAE (秒)"]
SEEK_JS = """(request, sample, method, binding) => {
 const id=__VIDEO_ID__, tokens=window.__visionforgeTemporalSeek ||= {};
 const token=tokens[id]=(tokens[id]||0)+1;
 if(!request || !binding || request.sample_id!==sample || request.method!==method ||
    binding.sample_id!==sample || binding.method!==method || request.video_name!==binding.video_name) return [];
 const t=Number(request.seconds); if(!Number.isFinite(t)||t<0) return [];
 const deadline=Date.now()+8000;
 const seek=()=>{
  if(tokens[id]!==token) return;
  const video=document.getElementById(id)?.querySelector('video');
  let source='';
  try{source=decodeURIComponent(new URL(video?.currentSrc||video?.src||'',window.location.href).pathname).replaceAll('\\\\','/');}catch{}
  if(!video || video.readyState<1 || !source.endsWith('/'+request.video_name)){
   if(Date.now()<deadline) setTimeout(seek,100); return;
  }
  video.currentTime=Math.min(t,Number.isFinite(video.duration)?video.duration:t);
  video.play().catch(()=>{});
 };
 seek();
 return [];
}"""
CANCEL_SEEK_JS = """() => {
 const id=__VIDEO_ID__, tokens=window.__visionforgeTemporalSeek ||= {};
 tokens[id]=(tokens[id]||0)+1;
 document.getElementById(id)?.querySelector('video')?.pause();
 return [];
}"""


def _path(root, name):
    root = Path(root).resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("实验资源路径超出公开目录")
    return candidate


def _read(root, name):
    path = _path(root, name)
    if path.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("公开实验文件超出大小限制")
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def _record(root, sample_id, method):
    if method not in ("uniform", "state_event"):
        raise ValueError("未知抽帧条件")
    record = _read(root, f"predictions/{sample_id}--{method}.json")
    if record["sample_id"] != sample_id or record["method"] != method:
        raise ValueError("预测记录身份不匹配")
    return record


def downloads(root):
    return [str(_path(root, name)) for name in ("report.html", "evaluation.json", "plan.json", "verify.py")
            if _path(root, name).is_file()] or None


def overview(root):
    try:
        report, samples, plan = (_read(root, name) for name in ("evaluation.json", "samples.json", "plan.json"))
        counts = []
        for method in NAMES:
            value = report["methods"][method]
            mae = value["conditional_onset_mae_seconds"]
            counts.append([NAMES[method], f"{value['correct']} / {value['n']}",
                *[_percent(value[key]) for key in ("failure_recall", "success_false_alarm_rate", "abstention_rate",
                  "hit_at_2s", "valid_onset_coverage")], "—" if mae is None else f"{mae:.2f}"])
        completed = sum(_path(root, f"predictions/{sample['sample_id']}--{method}.json").is_file()
                        for sample in samples for method in ("uniform", "state_event"))
        status = (f"**{len(samples)} 条连续机器人轨迹 · {len({s['task_id'] for s in samples})} 个任务 · "
                  f"{completed}/{len(samples) * 2} 次本地推理已保存**\n\n"
                  f"模型 `{plan['model']}` · 同一头部视角 · 每种方法 12 帧。作者人工标注失败起点，"
                  "成功轨迹没有虚构时间标签。成功/失败分层抽样的小规模测试，不代表原数据集的自然发生率。")
        return status, counts, [(f"{s['task_id']} · {s['sample_id']}", s["sample_id"]) for s in samples]
    except FileNotFoundError:
        return "连续轨迹实验尚未导出；此处不显示模拟成绩。", [], []
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return f"实验读取失败：{type(exc).__name__}: {exc}", [], []


def detail(root, sample_id, method):
    empty = ("选择一条轨迹查看原视频、模型输入和独立人工时间标签。", None, None, [], None, -1, {})
    if not sample_id:
        return empty
    try:
        sample = next(s for s in _read(root, "samples.json") if s["sample_id"] == sample_id)
        record = _record(root, sample_id, method)
        choices = [(f"帧 {i:02d} · {t:.3f} 秒", str(i)) for i, t in enumerate(record["selection"]["timestamps"])]
        frame = record.get("frame_id", -1)
        chosen = str(frame if type(frame) is int and 0 <= frame < len(choices) else 0)
        truth = sample.get("onset_seconds")
        prediction = record.get("onset_seconds")
        truth_text = "成功，无失败起点" if truth is None else f"失败起点 {truth:.3f} 秒"
        predicted_text = "无定位" if prediction is None else f"{prediction:.3f} 秒"
        text = (f"**任务：** {sample['task_description']}\n\n"
                f"**作者标注：** {sample['label']} · {truth_text}  \n"
                f"**模型输出：** {record['prediction']} · {predicted_text} · `{record['status']}`  \n"
                f"**视觉依据：** {record.get('reason') or record.get('error') or '无'}\n\n"
                "视频为便于复核生成的压缩副本；下方拼图保留实际发送给模型的字节。"
                "时间坐标来自原始记录。稀疏帧看不到的细节不能当作不存在。")
        return (text, str(_path(root, sample["video_file"])), str(_path(root, record["panel_file"])),
                choices, chosen, truth if truth is not None else -1, record)
    except (StopIteration, FileNotFoundError, KeyError, TypeError, ValueError, OSError) as exc:
        return (f"轨迹读取失败：{type(exc).__name__}: {exc}", *empty[1:])


def selected_time(root, sample_id, method, frame_id):
    if frame_id is None or not sample_id:
        return -1
    record = _record(root, sample_id, method)
    index = int(frame_id)
    if not 0 <= index < len(record["selection"]["timestamps"]):
        raise ValueError("帧编号超出范围")
    return record["selection"]["timestamps"][index]


def view_binding(values, sample_id, method):
    if not values[1] or not values[6]:
        return {}
    # Only the public filename goes to the browser; Gradio may change its cache
    # directory, but preserves this basename in the actual video source URL.
    return {"sample_id": sample_id, "method": method, "video_name": Path(values[1]).name}


def seek_request(root, sample_id, method, frame_id, binding, *, truth=False):
    if not binding or binding.get("sample_id") != sample_id or binding.get("method") != method:
        raise gr.Error("轨迹或抽帧条件正在切换，请等待画面加载后再播放。")
    sample = next(s for s in _read(root, "samples.json") if s["sample_id"] == sample_id)
    video_name = _path(root, sample["video_file"]).name
    if binding.get("video_name") != video_name:
        raise gr.Error("视频身份已改变，请刷新已导出实验。")
    seconds = sample.get("onset_seconds") if truth else selected_time(root, sample_id, method, frame_id)
    return {"sample_id": sample_id, "method": method, "video_name": video_name,
            "seconds": seconds if seconds is not None else -1}


@lru_cache(maxsize=2)
def _plot_evidence(root_name, modified_ns, size):
    # Retain only plot data, not the large state matrices. File identity keys
    # invalidate the cache when a refreshed export replaces the source JSON.
    rows = _read(root_name, "input-evidence.json")
    result = {row["sample_id"]: {key: row[key] for key in ("sample_id", "task_id", "source_sha256", "timestamp")}
              for row in rows}
    if len(result) != len(rows):
        raise ValueError("运动时间证据包含重复样本")
    return result


def trajectory_plot(root, sample_id, method):
    if not sample_id:
        return "<p>选择一条轨迹查看实测时间上的运动变化与采样帧。</p>"
    try:
        path = _path(root, "input-evidence.json")
        stat = path.stat()
        episode = _plot_evidence(str(Path(root).resolve()), stat.st_mtime_ns, stat.st_size)[sample_id]
        sample = next(row for row in _read(root, "samples.json") if row["sample_id"] == sample_id)
        return render_motion_plot(episode, sample, _record(root, sample_id, method))
    except FileNotFoundError:
        return "<p>运动时间证据尚未导出；不生成替代曲线。</p>"
    except (StopIteration, KeyError, TypeError, ValueError, OSError) as exc:
        return "<p>运动曲线读取失败：" + html.escape(f"{type(exc).__name__}: {exc}") + "</p>"


def build_temporal_tab(root, video_id="temporal-video"):
    root = Path(root).resolve()
    status_value, table_value, choices = overview(root)
    initial = choices[0][1] if choices else None
    values = detail(root, initial, "state_event")
    gr.Markdown("### 连续轨迹中的失败发生在哪一秒？\n"
                "FailTime 的 ALOHA / SO-101 真实视频与同步机器人状态；比较等间隔抽帧和状态事件抽帧。"
                "每个条件使用相同模型、相同相机和相同 12 帧预算。"
                "这是完整录像上的事后定位，不是实时告警或机器人控制。")
    status = gr.Markdown(status_value)
    table = gr.Dataframe(value=table_value, headers=HEADERS, datatype=["str"] * len(HEADERS), interactive=False, wrap=True)
    gr.Markdown("**±2s 命中率以全部真实失败为分母，漏检和拒答也计入。** 条件 MAE 只统计检出且有有效时间的失败，"
                "必须与定位覆盖率一起看。两个简单控制始终报告失败，误报代价在同表展示。")
    with gr.Row():
        picker = gr.Dropdown(choices=choices, value=initial, label="连续轨迹", interactive=True, scale=3, elem_id=video_id + "-sample")
        method = gr.Radio(choices=[(NAMES[m], m) for m in ("uniform", "state_event")], value="state_event", label="模型看到的抽帧条件", scale=3, elem_id=video_id + "-method")
    information = gr.Markdown(values[0])
    with gr.Row():
        video = gr.Video(value=values[1], label="完整轨迹 · 公开压缩复核副本", elem_id=video_id, interactive=False)
        panel = gr.Image(value=values[2], label="实际模型输入 · 12 帧与原始时间戳", interactive=False)
    curve = gr.HTML(value=trajectory_plot(root, initial, "state_event"), elem_id=video_id + "-motion")
    with gr.Row():
        frame = gr.Dropdown(choices=values[3], value=values[4], label="证据帧 / 时间", interactive=True, elem_id=video_id + "-frame")
        seek = gr.Button("播放所选证据时间", variant="primary", elem_id=video_id + "-seek")
        truth_seek = gr.Button("播放作者标注起点", elem_id=video_id + "-truth-seek")
    request = gr.JSON(value=None, visible=False)
    binding = gr.JSON(value=view_binding(values, initial, "state_event"), visible=False)
    truth_time = gr.Number(value=values[5], visible=False)
    with gr.Accordion("原始响应、抽帧分数与运行身份", open=False):
        raw = gr.JSON(value=values[6])
    refresh = gr.Button("刷新已导出实验")
    files = gr.File(value=downloads(root), label="公开报告与复算入口（完整资源请克隆仓库）", file_count="multiple", interactive=False)

    def load(sid, condition):
        result = detail(root, sid, condition)
        return (*result[:3], gr.Dropdown(choices=result[3], value=result[4]), *result[5:], view_binding(result, sid, condition),
                trajectory_plot(root, sid, condition))

    load_outputs = [information, video, panel, frame, truth_time, raw, binding, curve]
    queue_options = {"concurrency_id": "temporal-view:" + video_id, "concurrency_limit": 1}
    seek_js = SEEK_JS.replace("__VIDEO_ID__", json.dumps(video_id))
    cancel_js = CANCEL_SEEK_JS.replace("__VIDEO_ID__", json.dumps(video_id))
    for component in (picker, method):
        component.input(fn=None, inputs=[], outputs=[], js=cancel_js, queue=False)
        component.change(load, inputs=[picker, method], outputs=load_outputs, trigger_mode="always_last", **queue_options)
    seek.click(lambda sid, condition, fid, loaded: seek_request(root, sid, condition, fid, loaded),
        inputs=[picker, method, frame, binding], outputs=request, **queue_options).success(
        fn=None, inputs=[request, picker, method, binding], outputs=[], js=seek_js)
    truth_seek.click(lambda sid, condition, loaded: seek_request(root, sid, condition, None, loaded, truth=True),
        inputs=[picker, method, binding], outputs=request, **queue_options).success(
        fn=None, inputs=[request, picker, method, binding], outputs=[], js=seek_js)

    def reload(sid):
        status_text, rows, options = overview(root)
        sid = sid if sid in {value for _, value in options} else (options[0][1] if options else None)
        return status_text, rows, gr.Dropdown(choices=options, value=sid), downloads(root)

    refresh.click(reload, inputs=picker, outputs=[status, table, picker, files], **queue_options).then(
        load, inputs=[picker, method], outputs=load_outputs, **queue_options)
