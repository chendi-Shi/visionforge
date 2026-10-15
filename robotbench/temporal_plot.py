"""Dependency-free read-only plot of saved motion scores on measured time."""
from __future__ import annotations

import html
import math


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def render_motion_plot(episode, sample, record):
    identifiers = [row.get("sample_id") for row in (episode, sample, record)]
    if not identifiers[0] or len(set(identifiers)) != 1:
        raise ValueError("曲线、视频与预测的样本身份不一致")
    for key in ("task_id", "source_sha256"):
        if len({row[key] for row in (episode, sample, record)}) != 1:
            raise ValueError("曲线与当前视频的来源不一致")
    times = episode["timestamp"]
    selection = record["selection"]
    scores = selection["scores"]
    if not isinstance(times, list) or len(times) < 2 or not isinstance(scores, list) or len(times) != len(scores):
        raise ValueError("实测时间与运动分数长度不一致")
    if any(not _finite(t) for t in times) or times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("实测时间必须有限且严格递增")
    if any(not _finite(s) or s < 0 for s in scores):
        raise ValueError("运动分数必须为有限非负数")
    indices, selected_times = selection["indices"], selection["timestamps"]
    if (not isinstance(indices, list) or len(indices) != 12 or len(selected_times) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(times) for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError("需要已保存的12个互异且按时间排序的采样帧")
    if any(not _finite(t) or abs(t - times[i]) > 1e-6 for i, t in zip(indices, selected_times)):
        raise ValueError("选帧时间与原始实测时间不匹配")
    events = selection.get("event_indices", [])
    if not isinstance(events, list) or any(type(i) is not int or i not in indices for i in events) or len(set(events)) != len(events):
        raise ValueError("事件帧必须属于当前条件的已选帧")
    method = record.get("method")
    if method not in ("uniform", "state_event") or selection.get("method") != method or (method == "uniform" and events):
        raise ValueError("采样条件与事件标记不匹配")
    truth = sample.get("onset_seconds")
    if sample.get("label") not in ("success", "failure") or (truth is None) != (sample["label"] == "success"):
        raise ValueError("作者类别与失败起点不一致")
    prediction = record.get("onset_seconds") if record.get("status") == "ok" and record.get("prediction") == "failure" else None
    if prediction is not None:
        frame = record.get("frame_id")
        if type(frame) is not int or not 0 <= frame < len(indices) or prediction != selected_times[frame]:
            raise ValueError("模型起点与其引用的观察帧不一致")
    for value in (truth, prediction):
        if value is not None and (not _finite(value) or not times[0] <= value <= times[-1] + 1e-6):
            raise ValueError("起点不在该轨迹的实测时间范围内")

    left, top, width, height = 60, 42, 912, 180
    maximum = max(scores) or 1.0
    x = lambda t: left + width * (t - times[0]) / (times[-1] - times[0])
    y = lambda s: top + height * (1 - s / maximum)
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ['<div style="padding:12px 14px;border:1px solid #dbe4ea;border-radius:10px;background:#fff">',
             '<strong>运动变化与观察时刻</strong>',
             '<p style="margin:5px 0;color:#526575;font-size:13px">曲线是整段离线计算的运动变化分数，'
             '不是失败概率；横轴使用原始实测时间。标记来自当前采样条件，作者起点仅用于事后复核。</p>',
             '<svg viewBox="0 0 1000 290" role="img" aria-label="运动分数、12个观察帧与失败起点的实测时间图" '
             'style="display:block;width:100%;height:auto;font:12px sans-serif">',
             '<title>' + esc(identifiers[0]) + ' · ' + esc(method) + '</title>']
    for fraction in (0.0, 0.5, 1.0):
        position = y(maximum * fraction)
        parts += [f'<line x1="{left}" y1="{position:.2f}" x2="{left + width}" y2="{position:.2f}" stroke="#e2e8ee"/>',
                  f'<text x="{left - 10}" y="{position + 4:.2f}" text-anchor="end" fill="#526575">{maximum * fraction:.2f}</text>']
    for step in range(6):
        time = times[0] + (times[-1] - times[0]) * step / 5
        parts.append(f'<text x="{x(time):.2f}" y="246" text-anchor="middle" fill="#526575">{time:.1f}s</text>')
    points = " ".join(f"{x(t):.2f},{y(s):.2f}" for t, s in zip(times, scores))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#596e81" stroke-width="1.5"/>')
    event_set = set(events)
    for frame, index in enumerate(indices):
        event = index in event_set
        color = "#c06a14" if event else "#087f9b"
        kind = "状态事件帧" if event else ("等间隔帧" if method == "uniform" else "均匀锚点或间隙补齐帧")
        parts.append(f'<line x1="{x(times[index]):.2f}" y1="{top}" x2="{x(times[index]):.2f}" y2="{top + height}" '
                     f'stroke="{color}" stroke-opacity=".16" stroke-dasharray="2 3"/>')
        parts.append(f'<circle cx="{x(times[index]):.2f}" cy="{y(scores[index]):.2f}" r="4.5" fill="{color}" stroke="white">'
                     f'<title>帧 {frame:02d} · {times[index]:.3f}s · {kind}</title></circle>')
    for label, value, color, label_y in (("作者", truth, "#b42335", 15), ("模型", prediction, "#7845b8", 31)):
        if value is None:
            continue
        position = x(value)
        parts.append(f'<line x1="{position:.2f}" y1="{top}" x2="{position:.2f}" y2="{top + height}" '
                     f'stroke="{color}" stroke-width="2" stroke-dasharray="6 3"/>')
        anchor, offset = ("end", -4) if position > left + width * .8 else ("start", 4)
        parts.append(f'<text x="{position + offset:.2f}" y="{label_y}" text-anchor="{anchor}" fill="{color}">{label} {value:.3f}s</text>')
    parts.append('<text x="516" y="276" text-anchor="middle" fill="#526575">原始记录实测时间（秒）</text></svg>')
    non_event_name = "等间隔选帧" if method == "uniform" else "均匀锚点 / 间隙补齐帧"
    parts.append('<p style="margin:0;font-size:13px;color:#526575">'
                 f'<span style="color:#087f9b">● {non_event_name}</span> · '
                 f'<span style="color:#c06a14">● 状态事件帧</span> · 已选 {len(indices)} 帧 / 其中事件 {len(events)} 帧。 '
                 + ("作者：成功轨迹，无失败起点。 " if truth is None else "")
                 + ("模型：无有效失败定位。" if prediction is None else "") + '</p></div>')
    return "".join(parts)
