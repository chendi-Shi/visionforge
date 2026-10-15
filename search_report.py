"""Portable reports for pixel retrieval and independently verified sample points."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import shutil
import uuid
import zipfile

from core_grounding import COVERAGE_NOTE
from video_report import clock, _hash_file, _safe_csv, _time_button


def _text(value):
    return "" if value is None else str(value)


def _portable(value, path_map, by_hash):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key == "source_path":
                continue
            if key in {"path", "preview_path"}:
                output[key] = path_map.get(str(item)) or by_hash.get(value.get("sha256"))
            elif key.endswith("_dir") or key == "vectors_file":
                output[key] = None
            else:
                output[key] = _portable(item, path_map, by_hash)
        return output
    if isinstance(value, (list, tuple)):
        return [_portable(item, path_map, by_hash) for item in value]
    if isinstance(value, Path):
        return path_map.get(str(value))
    if isinstance(value, str):
        return path_map.get(value, value)
    return value


def _cards(frames, results=None, effective=None, reviews=None):
    results = results or {}
    effective, reviews = effective or {}, reviews or {}
    cards = []
    for frame in frames:
        result = results.get(frame.get("id")) or {}
        structured = result.get("structured") or {}
        verdict = structured.get("verdict") if result.get("status") == "ok" else result.get("status", "pending")
        label = f"{frame.get('id', '')} · {clock(frame.get('time_seconds'))}"
        body = (f'<img src="{escape(_text(frame.get("path")), quote=True)}" alt="{escape(label, quote=True)}" loading="lazy">'
                f'<span>{escape(label)}</span>')
        score = f'<p>余弦相似度：{float(frame["score"]):.4f}（排序值，非概率）</p>' if "score" in frame else ""
        status = f'<p><b>{escape(_text(verdict))}</b> · {escape(_text(structured.get("reason") or result.get("error")))}</p>' if results else ""
        event = reviews.get(frame.get("id"))
        if event:
            final = effective.get(frame.get("id")) or result
            final_structured = final.get("structured") or {}
            source = {"user": "用户复核", "developer_assistant": "开发助手复核（未经用户审核）", "model": "原始模型", "invalid_evidence": "证据失效，未采用"}.get(final.get("judgment_source"), "原始模型")
            status = '<p>模型原判：' + escape(_text(verdict)) + ' · ' + escape(_text(structured.get("reason") or result.get("error"))) + '</p>'
            status += '<p><b>当前判读：' + escape(_text(final_structured.get("verdict") if final.get("status") == "ok" else final.get("status"))) + '</b> · ' + escape(source) + '</p>'
            status += '<p>复核版本 ' + str(int(event["revision"])) + ' · ' + escape(_text(final_structured.get("reason") or final.get("error"))) + '</p>'
            if event.get("state") in {"stale", "deferred"}:
                status += '<p><b>此复核未应用：</b>' + escape(event["state"]) + '</p>'
        raw = '<details><summary>原始核验记录</summary><pre>' + escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre></details>' if result else ''
        cards.append('<article>' + _time_button(frame.get("time_seconds"), label, body) + score + status + raw + '</article>')
    return ''.join(cards) or '<p>尚无已准备好的证据。</p>'


def _interval_rows(intervals):
    rows = []
    for group in intervals.get("visible_groups", []):
        rows.append('<li>可见采样组：' + _time_button(group["evidence_start_seconds"]) + ' — ' + _time_button(group["evidence_end_seconds"])
                    + ' · ' + escape(', '.join(group["frame_ids"])) + '（不证明中间持续可见）</li>')
    for field, label in (("appearance_brackets", "未见→可见的相邻采样区间"), ("disappearance_brackets", "可见→未见的相邻采样区间")):
        for bracket in intervals.get(field, []):
            rows.append('<li>' + label + '：' + _time_button(bracket["after_seconds"]) + ' — ' + _time_button(bracket["by_seconds"])
                        + ' · ' + escape(f"{bracket['negative_frame_id']} / {bracket['positive_frame_id']}") + '</li>')
    return ''.join(rows) or '<li>尚无可推导的可见采样组或相邻变化区间。</li>'


def _review_html(search):
    history = search.get("review_history") or []
    if not history:
        return ""
    intervals = search.get("reviewed_intervals") or {}
    summary = search.get("review_summary") or {}
    rejections = search.get("review_rejections") or []
    rejected = '<div class="notice"><b>未应用的复核</b><ul>' + ''.join(
        '<li>' + escape(_text(item.get("frame_id"))) + ' · ' + escape(_text(item.get("reason"))) + '</li>' for item in rejections
    ) + '</ul></div>' if rejections else ''
    if search.get("review_interval_warning"):
        rejected += '<div class="notice"><b>时间区间未推导：</b>' + escape(_text(search["review_interval_warning"])) + '</div>'
    rows = []
    for event in history:
        source = {"user": "用户", "developer_assistant": "开发助手（未经用户审核）"}.get(event.get("actor"), _text(event.get("actor")))
        rows.append('<tr>' + ''.join('<td>' + escape(_text(value)) + '</td>' for value in (
            event.get("frame_id"), event.get("revision"), source, event.get("verdict"), event.get("note"), event.get("created_at_utc"))) + '</tr>')
    return ('<section><h2>复核覆盖后的时间证据</h2><p>用户复核 ' + str(int(summary.get("active_user", 0)))
            + ' 帧；开发助手复核 ' + str(int(summary.get("active_developer_assistant", 0)))
            + ' 帧。未复核帧仍采用模型判断；这些区间不代表整段视频已经人工确认。复核不会修改模型原判。</p>' + rejected + '<ul>'
            + _interval_rows(intervals) + '</ul><details><summary>复核后的区间、间隔与不确定性</summary><pre>'
            + escape(json.dumps(intervals, ensure_ascii=False, indent=2)) + '</pre></details></section>'
            + '<section><h2>复核审计记录（追加保存）</h2><p>reset 表示恢复模型原判，历史判断仍保留。时间均为 UTC。</p><div style="overflow-x:auto"><table><thead><tr>'
            + ''.join('<th>' + label + '</th>' for label in ('帧', '版本', '复核人', '判读', '依据', '时间'))
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>')


def _html(search):
    intervals = search.get("intervals") or {}
    translation = search.get("translation") or {}
    attribution = search.get("source_attribution")
    attributed = '<section><h2>来源与许可</h2><pre>' + escape(json.dumps(attribution, ensure_ascii=False, indent=2)) + '</pre></section>' if attribution else ''
    review = search.get("assistant_review")
    reviewed = '<section><h2>开发助手复核 · 未经用户审核</h2><pre>' + escape(json.dumps(review, ensure_ascii=False, indent=2)) + '</pre></section>' if review else ''
    results = {r.get("frame", {}).get("id"): r for r in search.get("results", [])}
    effective = {r.get("frame", {}).get("id"): r for r in search.get("effective_results", [])}
    title = escape(_text(search.get("job_name") or "视频视觉检索"))
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · VisionForge</title><style>
body{{margin:0;background:#f3f6f8;color:#173143;font:15px system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:30px 24px 80px}}
h1{{font-size:28px}}h2{{font-size:21px}}p,li{{line-height:1.7;overflow-wrap:anywhere}}section,article{{background:white;padding:20px;border:1px solid #dbe5eb;border-radius:12px;margin:16px 0}}
.notice{{background:#fff1d2;padding:16px;border-radius:10px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:15px}}.cards article{{min-width:0;margin:0}}
video{{width:100%;max-height:550px;background:#102737;border-radius:12px}}img{{width:100%;max-height:280px;object-fit:contain}}button{{cursor:pointer;border:1px solid #c0d3df;border-radius:7px;padding:5px 9px;background:#f2f8fa;color:#193e50}}
article button{{padding:0;width:100%;overflow:hidden}}article span{{display:block;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f7;padding:14px;font-size:12px}}small{{color:#587283}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #dbe5eb;vertical-align:top;overflow-wrap:anywhere}}
</style></head><body><main><h1>{title} · 视觉检索证据</h1>
<p>状态：<b>{escape(_text(search.get('status')))}</b> · 检索 ID：{escape(_text(search.get('id')))}</p>
<p>原问题：{escape(_text(search.get('target')))}</p><p>逐帧核验目标：{escape(_text(translation.get('visual_target') or search.get('target')))}</p><p>实际英文检索词：{escape(_text(translation.get('query')))} · 来源：{escape(_text(translation.get('source')))}</p>
<div class="notice">{escape(COVERAGE_NOTE)} 本流程不分析声音。CLIP 候选只表示画面相关性排序；核验结论仍需人工查看原帧。</div>
<p>{escape(_text(search.get('message')))}</p><p>{escape(_text(search.get('error')))}</p>
{reviewed}
<video id="evidence-video" controls playsinline preload="metadata" src="video.mp4"></video>
<p><small>点击帧或时间定位。请保持 report.html、video.mp4 和 frames 文件夹在同一目录。</small></p>
{_review_html(search)}
<section><h2>模型原判推导的时间证据 · 请复核原帧</h2><p>任何一帧误判都会使区间错误。以下区间不是已人工确认的事件边界。</p><ul>{_interval_rows(intervals)}</ul>
<details><summary>完整区间、间隔与不确定性</summary><pre>{escape(json.dumps(intervals, ensure_ascii=False, indent=2))}</pre></details></section>
<section><h2>视觉相似度候选</h2><p>候选由视频画面向量检索得到，不依赖旧版摘要；排名靠前不等于目标出现。</p><div class="cards">{_cards(search.get('candidates', []))}</div></section>
<section><h2>逐帧核验与复核</h2><div class="cards">{_cards(search.get('frames', []), results, effective, search.get('review_current'))}</div></section>
{attributed}<details><summary>配置、模型身份与检索词翻译原文</summary><pre>{escape(json.dumps({k: search.get(k) for k in ('config', 'identity', 'translation', 'export')}, ensure_ascii=False, indent=2))}</pre></details>
<p><small>完整原始结果与每次尝试保存在 evidence.json；schema 合格率不能作为准确率。</small></p></main><script>
const video=document.getElementById('evidence-video');
document.querySelectorAll('button[data-time]').forEach(button=>button.addEventListener('click',()=>{{
 const target=Number(button.dataset.time);if(!Number.isFinite(target)||target<0)return;
 const seek=()=>{{video.currentTime=Number.isFinite(video.duration)?Math.min(target,video.duration):target;video.pause();}};
 if(video.readyState>=1)seek();else video.addEventListener('loadedmetadata',seek,{{once:true}});
 video.scrollIntoView({{behavior:'smooth',block:'center'}});
}}));</script></body></html>'''


def export_search(search: dict, outputdir: str | Path) -> list[str]:
    """Create a new verified portable package; return ZIP / JSON / CSV paths."""
    video = search.get("video") or {}
    preview = Path(video.get("preview_path") or "")
    expected = video.get("preview_sha256")
    if not preview.is_file() or not expected or _hash_file(preview) != expected:
        raise ValueError("规范化预览缺失或 SHA256 不匹配，请重新准备视频。")
    folder = Path(outputdir).expanduser().resolve() / uuid.uuid4().hex
    (folder / "frames").mkdir(parents=True)
    shutil.copy2(preview, folder / "video.mp4")
    if _hash_file(folder / "video.mp4") != expected:
        raise ValueError("预览在导出期间发生变化，未生成证据包。")
    path_map = {str(preview): "video.mp4", str(preview.resolve()): "video.mp4"}
    by_hash, assets = {}, []
    frames = list(search.get("candidates") or []) + list(search.get("frames") or [])
    frames += [r["frame"] for r in list(search.get("results") or []) + list(search.get("attempts") or []) if isinstance(r.get("frame"), dict)]
    for frame in frames:
        original = Path(frame.get("path") or "")
        expected_frame = frame.get("sha256")
        if not original.is_file() or not expected_frame or _hash_file(original) != expected_frame:
            raise ValueError(f"证据帧 {frame.get('id', '')} 缺失或 SHA256 不匹配，拒绝导出改变的证据。")
        if str(original) in path_map:
            continue
        if expected_frame in by_hash:
            relative = by_hash[expected_frame]
        else:
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(frame.get("id") or "frame"))[:64]
            suffix = original.suffix.lower() if original.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".img"
            relative = f"frames/{len(assets)+1:04d}-{slug}{suffix}"
            shutil.copy2(original, folder / relative)
            if _hash_file(folder / relative) != expected_frame:
                raise ValueError("证据帧在导出期间发生变化，未生成证据包。")
            by_hash[expected_frame] = relative
            assets.append({"path": relative, "sha256": expected_frame})
        path_map[str(original)] = relative
        path_map[str(original.resolve())] = relative
    portable = _portable(search, path_map, by_hash)
    attribution_path = Path(__file__).resolve().parent / "examples" / "video" / "source.json"
    if "source_attribution" not in portable and attribution_path.is_file():
        attribution = json.loads(attribution_path.read_text(encoding="utf-8-sig"))
        if attribution.get("original_file", {}).get("sha256") == video.get("source_sha256"):
            portable["source_attribution"] = attribution
    portable["export"] = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "video": "video.mp4",
                          "preview_sha256": expected, "frames": assets, "scope": COVERAGE_NOTE,
                          "index_assets": "Only selected candidates and verification frames are packaged; other index image paths and vector paths are null. Identity and hashes are retained."}
    json_path, csv_path = folder / "evidence.json", folder / "timeline.csv"
    json_path.write_text(json.dumps(portable, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["search_id", "target", "frame_id", "time_seconds", "time", "status", "verdict", "reason", "error", "model", "digest", "sha256",
                         "effective_status", "effective_verdict", "effective_reason", "judgment_source", "review_revision", "review_actor", "review_note", "review_state"])
        effective = {r.get("frame", {}).get("id"): r for r in portable.get("effective_results", [])}
        for result in portable.get("results", []):
            frame, structured = result.get("frame") or {}, result.get("structured") or {}
            final = effective.get(frame.get("id")) or result
            final_structured = final.get("structured") or {}
            event = (portable.get("review_current") or {}).get(frame.get("id")) or {}
            writer.writerow([_safe_csv(value) for value in [portable.get("id"), portable.get("target"), frame.get("id"),
                frame.get("time_seconds"), clock(frame.get("time_seconds")), result.get("status"), structured.get("verdict"),
                structured.get("reason"), result.get("error"), result.get("model"), result.get("digest"), frame.get("sha256"),
                final.get("status"), final_structured.get("verdict"), final_structured.get("reason"), final.get("judgment_source", "model"),
                event.get("revision"), event.get("actor"), event.get("note"), event.get("state")]])
    (folder / "report.html").write_text(_html(portable), encoding="utf-8")
    archive = folder / "visual-search.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path != archive:
                compression = zipfile.ZIP_DEFLATED if path.suffix in {".html", ".json", ".csv"} else zipfile.ZIP_STORED
                bundle.write(path, path.relative_to(folder).as_posix(), compress_type=compression)
    return [str(archive), str(json_path), str(csv_path)]
