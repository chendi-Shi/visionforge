"""Presentation helpers shared by the library UI and portable case reports."""
from __future__ import annotations

from html import escape

from video_report import clock

STATUS = {"queued_prepare": "等待预处理", "preparing": "正在抽帧", "ready": "可搜索",
          "queued_analysis": "等待摘要", "queued": "排队中", "running": "处理中",
          "pause_requested": "正在暂停", "paused": "已暂停", "completed": "已完成",
          "partial": "部分完成", "failed": "失败", "ok": "已核验"}
VERDICT = {"visible": "可见", "not_visible": "未见", "uncertain": "不确定",
           "failed": "失败", "pending": "待核验", "reset": "恢复模型判断"}
ACTOR = {"user": "用户复核", "developer_assistant": "开发助手复核"}


def verdict(result):
    return ((result.get("structured") or {}).get("verdict", "uncertain")
            if result.get("status") == "ok" else result.get("status", "pending"))


def library_rows(videos, query="", include_archived=False):
    query = (query or "").strip().casefold()
    filtered = [v for v in videos if (include_archived or not v.get("archived")) and
                (not query or query in " ".join([v.get("title", ""), v.get("name", ""),
                                                 *v.get("tags", [])]).casefold())]
    rows = [[v["id"], v.get("title") or v["name"], ", ".join(v.get("tags", [])),
             clock((v.get("video") or {}).get("duration_seconds", 0)),
             "已归档" if v.get("archived") else ("可搜索" if v.get("ready_for_search") else STATUS.get(v["status"], v["status"]))]
            for v in filtered]
    options = [(v.get("title") or v["name"], v["id"]) for v in filtered
               if v.get("ready_for_search") and not v.get("archived")]
    return rows, options


def metrics(items):
    return '<div class="metrics">' + ''.join(
        '<div class="metric"><b>' + escape(str(value)) + '</b><span>' + escape(label) + '</span></div>'
        for value, label in items) + '</div>'


def interval_rows(intervals):
    rows = [["可见采样组", clock(g["evidence_start_seconds"]), clock(g["evidence_end_seconds"]),
             ", ".join(g["frame_ids"])] for g in intervals.get("visible_groups", [])]
    for field, label in (("appearance_brackets", "未见 → 可见"), ("disappearance_brackets", "可见 → 未见")):
        rows += [[label, clock(b["after_seconds"]), clock(b["by_seconds"]),
                  b["negative_frame_id"] + " / " + b["positive_frame_id"]]
                 for b in intervals.get(field, [])]
    return rows


def evidence_rows(search, only="全部"):
    raw = {r["frame"]["id"]: r for r in search.get("results", [])}
    effective = {r["frame"]["id"]: r for r in search.get("effective_results", search.get("results", []))}
    current = search.get("review_current", {})
    rows = []
    for frame in search.get("frames", []):
        fid = frame["id"]
        result = effective.get(fid, {})
        review = current.get(fid, {})
        is_reviewed = review.get("state") == "active"
        if only == "待复核" and is_reviewed:
            continue
        if only == "可见" and verdict(result) != "visible":
            continue
        if only == "不确定 / 失败" and verdict(result) not in {"uncertain", "failed", "pending"}:
            continue
        rows.append([fid, clock(frame["time_seconds"]), VERDICT.get(verdict(raw.get(fid, {})), verdict(raw.get(fid, {}))),
                     VERDICT.get(verdict(result), verdict(result)), ACTOR.get(review.get("actor"), "模型 · 待复核") if is_reviewed else "模型 · 待复核"])
    return rows


def cross_video_rows(children, reviews=None, only_visible=False):
    """Never join equal frame IDs from different videos or compare CLIP as confidence."""
    rows = []
    for child in children:
        search = child["search"]
        if reviews is not None:
            search = reviews.apply(search)
        current = search.get("review_current", {})
        for result in search.get("effective_results", search.get("results", [])):
            value = verdict(result)
            if only_visible and value != "visible":
                continue
            frame = result["frame"]
            review = current.get(frame["id"], {})
            reviewed = review.get("state") == "active"
            rows.append([child["search_id"], child["job_id"], frame["id"], child["title"], clock(frame["time_seconds"]),
                         VERDICT.get(value, value), ACTOR.get(review.get("actor"), "模型") if reviewed else "模型 · 待复核"])
    return rows
