"""Portable, offline video evidence reports. No model calls or external assets."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any
import uuid
import zipfile


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def clock(seconds: float) -> str:
    """Format a media timestamp, carrying rounded milliseconds into the next unit."""
    try:
        number = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number) or number < 0:
        return "—"
    total = round(number * 1000)
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    whole, milliseconds = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{milliseconds:03d}"


def effective_summary(segment: dict) -> str:
    review = segment.get("review") or {}
    if review.get("decision") == "corrected" and isinstance(review.get("text"), str):
        return review["text"]
    result = segment.get("result") or {}
    if isinstance(result, dict) and result.get("summary"):
        return _text(result["summary"])
    error = segment.get("error") or (result.get("error") if isinstance(result, dict) else None)
    return _text(error) if error else "尚无模型结果"


def _safe_csv(value: Any) -> str:
    text = _text(value)
    if text.startswith(("\t", "\r", "\n")) or text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "source_path":
                continue
            converted = _portable(item, path_map)
            # Do not leak machine-specific paths from an unmapped optional metadata field.
            if key == "path" and isinstance(converted, str) and (Path(converted).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", converted)):
                converted = None
            result[key] = converted
        return result
    if isinstance(value, (list, tuple)):
        return [_portable(item, path_map) for item in value]
    if isinstance(value, (str, Path)):
        return path_map.get(str(value), str(value))
    return value


def _time_button(seconds: Any, label: str | None = None, body: str | None = None) -> str:
    try:
        number = float(seconds)
        valid = math.isfinite(number) and number >= 0
    except (TypeError, ValueError):
        valid, number = False, 0.0
    caption = escape(label if label is not None else clock(seconds))
    contents = body if body is not None else caption
    if not valid:
        return f'<span class="unknown-time">{contents}</span>'
    return f'<button class="seek" type="button" data-time="{number:.6f}" aria-label="定位到 {escape(clock(number), quote=True)}">{contents}</button>'


def _html_report(job: dict) -> str:
    sections = []
    for segment in job.get("segments", []):
        result = segment.get("result") or {}
        if not isinstance(result, dict):
            result = {"raw_result": result}
        review = segment.get("review") or {}
        frames = segment.get("frames") or []
        by_id = {str(frame.get("id")): frame for frame in frames}
        cards = []
        for frame in frames:
            frame_id = _text(frame.get("id"))
            label = f"{frame_id} · {clock(frame.get('time_seconds'))}"
            picture = f'<img src="{escape(_text(frame.get("path")), quote=True)}" alt="{escape(label, quote=True)}" loading="lazy"><span>{escape(label)}</span>'
            cards.append('<div class="frame">' + _time_button(frame.get("time_seconds"), label, picture) + '</div>')
        observations = []
        for observation in result.get("observations") or []:
            if not isinstance(observation, dict):
                observations.append(f"<li>{escape(_text(observation))}</li>")
                continue
            references = []
            for frame_id in observation.get("evidence_frame_ids") or []:
                frame = by_id.get(str(frame_id))
                references.append(_time_button(frame["time_seconds"], _text(frame_id)) if frame else escape(_text(frame_id)))
            observations.append(
                f'<li><strong>{escape(_text(observation.get("category")))}</strong> '
                f'{escape(_text(observation.get("description")))} '
                f'<span class="refs">证据帧：{" · ".join(references) if references else "未提供"}</span></li>'
            )
        uncertainties = result.get("uncertainties") or []
        uncertainty_html = "".join(f"<li>{escape(_text(item))}</li>" for item in uncertainties)
        raw = json.dumps(segment.get("result"), ensure_ascii=False, indent=2)
        original_summary = _text(result.get("summary"))
        sections.append(
            f'<section class="segment"><h2>{escape(_text(segment.get("id")))} '
            f'<span class="badge">{escape(_text(segment.get("status")))}</span></h2>'
            f'<p class="window">采样窗口：{_time_button(segment.get("start_seconds"))} — {_time_button(segment.get("end_seconds"))}</p>'
            f'<p class="summary">{escape(effective_summary(segment))}</p>'
            f'<p class="muted">原模型摘要：{escape(original_summary or "尚无模型摘要")}</p>'
            f'<div class="frames">{"".join(cards)}</div>'
            f'<h3>观察与引用</h3><ul>{"".join(observations) or "<li>尚无模型观察。</li>"}</ul>'
            f'<h3>不确定性</h3><ul>{uncertainty_html or "<li>未提供；这不代表结论已经确认。</li>"}</ul>'
            f'<p class="error">{escape(_text(segment.get("error")))}</p>'
            f'<div class="review"><strong>人工审阅：{escape(_text(review.get("decision", "unreviewed")))}</strong>'
            f'<p>{escape(_text(review.get("text")))}</p><small>人工修正仅影响展示摘要，原始模型结果保留在下方及 evidence.json 中。</small></div>'
            f'<details><summary>原始模型结果 JSON</summary><pre>{escape(raw)}</pre></details></section>'
        )
    exported = job.get("export") or {}
    warnings = "".join(f"<li>{escape(_text(item))}</li>" for item in exported.get("warnings", []))
    title = escape(_text(job.get("name") or "视频理解报告"))
    config, identity = job.get("config") or {}, job.get("identity") or {}
    attribution = job.get("source_attribution")
    attribution_html = (
        '<section class="segment"><h2>来源与许可</h2><pre>'
        + escape(json.dumps(attribution, ensure_ascii=False, indent=2)) + '</pre></section>'
        if attribution is not None else ''
    )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · VisionForge</title>
<style>
:root{{color-scheme:light;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#17243a;background:#f4f6fa}}
body{{margin:0}}main{{max-width:1100px;margin:auto;padding:28px 24px 80px}}h1{{font-size:28px}}h2{{font-size:21px}}h3{{font-size:16px}}
p,li{{line-height:1.7;overflow-wrap:anywhere}}.notice,.review{{padding:16px;background:#fff5dd;border-radius:10px}}.review{{background:#eef3ff;margin:18px 0}}
.segment{{background:white;border:1px solid #dbe2ec;border-radius:14px;padding:24px;margin:24px 0}}.badge{{font-size:13px;padding:4px 9px;border-radius:20px;background:#e8eef8}}
video{{display:block;width:100%;max-height:540px;background:#101723;border-radius:12px;margin:20px 0}}.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}
.frame img{{display:block;max-width:100%;width:100%;max-height:320px;object-fit:contain;background:#edf0f4}}.frame button{{display:block;width:100%;padding:0;overflow:hidden}}
.frame span{{display:block;padding:8px}}button{{cursor:pointer;border:1px solid #bfccdf;border-radius:7px;background:#f5f8ff;color:#183e78;padding:5px 9px}}button:hover{{background:#e2ebfc}}
.refs{{display:block;font-size:13px;margin-top:6px}}.muted,small{{color:#5e6b7f}}.summary{{font-size:18px;white-space:pre-wrap}}.review p{{white-space:pre-wrap}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;padding:16px;border-radius:8px;font-size:12px}}.error{{color:#a33326}}.window{{color:#53637a}}
</style></head><body><main>
<h1>{title}</h1><p>作业状态：<strong>{escape(_text(job.get("status")))}</strong> · ID：{escape(_text(job.get("id")))}</p>
<div class="notice"><strong>阅读边界</strong><p>本报告来自稀疏采样帧。所列时间段是采样窗口，绝非事件精确起止时间；未采样画面可能包含重要变化。本流程不分析声音。模型描述和未审阅结果都需要人工核验。</p></div>
<p>关注问题：{escape(_text(config.get("focus")))}</p><p class="muted">模型：{escape(_text(identity.get("model") or config.get("model")))} · digest：{escape(_text(identity.get("digest") or "尚未开始分析"))}</p>
<p class="error">{escape(_text(job.get("error")))}</p><p>{escape(_text(job.get("message")))}</p>
<video id="evidence-video" controls preload="metadata" playsinline src="video.mp4">当前浏览器无法播放此规范化预览；可使用本地播放器打开 video.mp4。</video>
<p class="muted">点击时间或帧定位到规范化预览，其时间基与证据帧一致。请保持 report.html、video.mp4 和 frames 文件夹在同一目录。</p>
{'<div class="notice"><strong>导出提示</strong><ul>' + warnings + '</ul></div>' if warnings else ''}
{''.join(sections) or '<p>此作业尚无已准备好的采样段。</p>'}
{attribution_html}
<details><summary>采样选择记录</summary><pre>{escape(json.dumps(job.get("selection"), ensure_ascii=False, indent=2))}</pre></details>
<p class="muted">导出时间（UTC）：{escape(_text(exported.get("created_at_utc")))}</p>
</main><script>
const video = document.getElementById('evidence-video');
document.querySelectorAll('button[data-time]').forEach(button => {{
  button.addEventListener('click', () => {{
    const target = Number(button.dataset.time);
    if (!Number.isFinite(target) || target < 0) return;
    const seek = () => {{ video.currentTime = Number.isFinite(video.duration) ? Math.min(target, video.duration) : target; video.pause(); }};
    if (video.readyState >= 1) seek(); else video.addEventListener('loadedmetadata', seek, {{once:true}});
    video.scrollIntoView({{behavior:'smooth', block:'center'}});
  }});
}});
</script></body></html>'''


def export_job(job: dict, output_root: str | Path) -> list[str]:
    """Return ZIP, JSON and CSV paths; each call writes a fresh UUID directory."""
    video = job.get("video") or {}
    preview_value = video.get("preview_path")
    if not isinstance(preview_value, (str, Path)) or not str(preview_value).strip():
        raise ValueError("Normalized MP4 preview is missing. Prepare the video before exporting; the original source cannot replace the evidence preview.")
    preview = Path(preview_value)
    if not preview.is_file():
        raise ValueError("Normalized MP4 preview file is missing. Prepare the video again before exporting.")
    expected_preview_hash = video.get("preview_sha256")
    if not isinstance(expected_preview_hash, str) or not expected_preview_hash:
        raise ValueError("Normalized preview SHA256 is missing; cannot verify the evidence timeline for export.")
    if _hash_file(preview) != expected_preview_hash:
        raise ValueError("Normalized preview SHA256 mismatch. The preview changed after preparation; create a new job before exporting.")
    folder = Path(output_root).expanduser().resolve() / uuid.uuid4().hex
    (folder / "frames").mkdir(parents=True)
    shutil.copy2(preview, folder / "video.mp4")
    if _hash_file(folder / "video.mp4") != expected_preview_hash:
        raise ValueError("Normalized preview changed during export; the evidence package was not created.")
    path_map = {str(preview): "video.mp4", str(preview.resolve()): "video.mp4"}
    assets, warnings = [], []
    for segment in job.get("segments", []):
        for frame in segment.get("frames") or []:
            original = Path(frame["path"])
            key = str(original)
            if key in path_map:
                continue
            if not original.is_file():
                raise ValueError(f"Evidence frame is missing: {frame.get('id', 'unknown')}")
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", _text(frame.get("id") or "frame"))[:64]
            suffix = original.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                suffix = ".img"
            relative = f"frames/{len(assets) + 1:04d}-{slug}{suffix}"
            target = folder / relative
            shutil.copy2(original, target)
            actual = _hash_file(target)
            expected = frame.get("sha256")
            if expected and expected != actual:
                warnings.append(f"证据帧 {frame.get('id', '')} 的当前文件与原记录 SHA256 不一致，不能视为原始证据；请检查源作业。")
            assets.append({"frame_id": frame.get("id"), "path": relative, "sha256": actual, "recorded_sha256": expected, "matches_recorded_sha256": actual == expected if expected else None})
            path_map[key] = relative
            path_map[str(original.resolve())] = relative
    portable = _portable(job, path_map)
    portable["export"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": "video.mp4", "report": "report.html", "frames": assets, "warnings": warnings,
        "video_role": "normalized H264 preview using the same timebase as evidence frames", "preview_sha256": expected_preview_hash,
        "scope": "Sparse sampled windows, not precise event boundaries. Audio is not analyzed. Human review is stored separately from original model results.",
    }
    json_path, csv_path, html_path = folder / "evidence.json", folder / "timeline.csv", folder / "report.html"
    json_path.write_text(json.dumps(portable, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["segment_id", "start_seconds", "end_seconds", "start_time", "end_time", "status", "effective_summary",
               "model_summary", "model_error", "review_decision", "review_text", "observations", "uncertainties", "frame_ids"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for segment in portable.get("segments", []):
            result, review = segment.get("result") or {}, segment.get("review") or {}
            row = [segment.get("id"), segment.get("start_seconds"), segment.get("end_seconds"), clock(segment.get("start_seconds")),
                   clock(segment.get("end_seconds")), segment.get("status"), effective_summary(segment), result.get("summary"),
                   segment.get("error") or result.get("error"), review.get("decision"), review.get("text"), result.get("observations"),
                   result.get("uncertainties"), [frame.get("id") for frame in segment.get("frames") or []]]
            writer.writerow([_safe_csv(value) for value in row])
    html_path.write_text(_html_report(portable), encoding="utf-8")
    archive = folder / "video-review.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path != archive:
                compress = zipfile.ZIP_DEFLATED if path.suffix in {".html", ".json", ".csv"} else zipfile.ZIP_STORED
                bundle.write(path, path.relative_to(folder).as_posix(), compress_type=compress)
    return [str(archive), str(json_path), str(csv_path)]
