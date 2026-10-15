"""Append-only review decisions, bound to immutable search and frame evidence.

Reviews are an overlay: the original model results and attempts are never rewritten.
An actor records who made a decision; ``developer_assistant`` is never counted as
user review. Callers must fetch a fresh search snapshot before saving a decision.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from core_grounding import derive_intervals


REVIEWABLE_STATUSES = frozenset({"completed", "paused", "partial", "failed"})
VERDICTS = frozenset({"visible", "not_visible", "uncertain", "reset"})
ACTORS = frozenset({"user", "developer_assistant"})


class ReviewConflict(ValueError):
    """Another decision was saved after the caller loaded this frame."""


class StaleReview(ValueError):
    """A saved review no longer refers to the supplied immutable evidence."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_id(frame: dict) -> str:
    """Use the stable extracted ID, with an identity-based fallback for imports."""
    existing = frame.get("id")
    if isinstance(existing, str) and existing.strip():
        return existing
    digest = frame.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("证据帧缺少稳定 ID 和 SHA256。")
    position = frame.get("frame_index", frame.get("time_seconds"))
    if position is None:
        raise ValueError("证据帧缺少帧序号或时间。")
    return "frame-" + hashlib.sha256(_json([digest, position]).encode("utf-8")).hexdigest()[:24]


def _position(frame):
    stamp = frame.get("time_seconds")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0:
        raise ValueError("证据帧时间无效，不能保存复核。")
    return {"id": frame_id(frame), "sha256": frame.get("sha256"),
            "frame_index": frame.get("frame_index"), "time_seconds": stamp}


def _result(search, selected):
    found = [item for item in search.get("results", []) if frame_id(item.get("frame") or {}) == selected]
    if len(found) != 1:
        raise StaleReview("复核对应的原始结果缺失或 ID 重复。")
    return found[0]


def _binding(search, selected, *, verify_file=True):
    result = _result(search, selected)
    frame = result.get("frame") or {}
    point = _position(frame)
    if not isinstance(point["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", point["sha256"]):
        raise StaleReview("复核证据缺少有效 SHA256。")
    prepared = [item for item in search.get("frames", []) if frame_id(item) == selected]
    if len(prepared) != 1 or _position(prepared[0]) != point:
        raise StaleReview("准备的证据帧与原始结果不一致。")
    if verify_file:
        for item in (frame, prepared[0]):
            path = Path(item.get("path") or "")
            if not path.is_file() or _hash(path) != point["sha256"]:
                raise StaleReview("复核证据帧缺失或 SHA256 改变，拒绝应用旧复核。")
    identity = search.get("identity") or {}
    target = search.get("target")
    model = result.get("model") or identity.get("model")
    digest = result.get("digest") or identity.get("digest")
    if not isinstance(target, str) or not target.strip() or not model or not digest:
        raise StaleReview("复核需要明确的检索目标及模型身份。")
    # Exclude machine-specific asset paths, but bind the review to the exact
    # original answer as well as the model/video/target and extracted pixel hash.
    original = {key: result.get(key) for key in ("status", "target", "model", "digest", "structured", "raw_response", "error")}
    return {"search_id": search["id"], "target": target,
            "visual_target": (search.get("translation") or {}).get("visual_target"),
            "video_sha256": (search.get("video") or {}).get("preview_sha256"),
            "search_identity": identity, "model": model, "digest": digest, "frame": point,
            "original_result_sha256": hashlib.sha256(_json(original).encode("utf-8")).hexdigest()}


def _max_gap(search):
    step = (search.get("config") or {}).get("step_seconds", 1)
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
        raise ValueError("复核时间线需要有效采样间隔。")
    return step + 1 / 12


class ReviewStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "search-reviews.sqlite3"
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS reviews (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id TEXT NOT NULL, frame_id TEXT NOT NULL, revision INTEGER NOT NULL,
                event TEXT NOT NULL, UNIQUE(search_id, frame_id, revision))""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS reviews_no_update BEFORE UPDATE ON reviews
                BEGIN SELECT RAISE(ABORT, 'Review history is append-only'); END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS reviews_no_delete BEFORE DELETE ON reviews
                BEGIN SELECT RAISE(ABORT, 'Review history is append-only'); END""")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.database, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _stable(search):
        if search.get("status") not in REVIEWABLE_STATUSES:
            raise ValueError("请先暂停或等待检索完成，再保存或应用复核。")
        if not isinstance(search.get("id"), str) or not search["id"].strip():
            raise ValueError("复核缺少检索 ID。")

    def save(self, search: dict, frame_id: str, verdict: str, note: str,
             actor: str = "user", expected_revision: int = 0) -> dict:
        self._stable(search)
        if verdict not in VERDICTS:
            raise ValueError("复核结论必须为 visible、not_visible、uncertain 或 reset。")
        if actor not in ACTORS:
            raise ValueError("复核人只能为 user 或 developer_assistant。")
        if not isinstance(note, str) or len(note.strip()) > 2000 or (verdict != "reset" and not note.strip()):
            raise ValueError("复核判断必须填写依据，最多 2000 字。")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("复核版本无效。")
        binding = _binding(search, frame_id)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT event FROM reviews WHERE search_id=? AND frame_id=? ORDER BY revision DESC LIMIT 1",
                             (search["id"], frame_id)).fetchone()
            current = json.loads(row[0]) if row else None
            revision = current["revision"] if current else 0
            if revision != expected_revision:
                raise ReviewConflict(f"复核已更新至版本 {revision}，请刷新后再保存。")
            if current and current["binding"] != binding and verdict != "reset":
                raise StaleReview("目标、模型或原始证据改变，已有复核不能套用到当前结果。")
            if current and current["binding"] != binding and not note.strip():
                raise ValueError("撤销已失效的复核时，请填写原因；随后可以对新证据重新复核。")
            event = {"search_id": search["id"], "frame_id": frame_id, "revision": revision + 1,
                     "verdict": verdict, "note": note.strip(), "actor": actor,
                     "created_at_utc": datetime.now(timezone.utc).isoformat(), "binding": binding}
            db.execute("INSERT INTO reviews(search_id, frame_id, revision, event) VALUES(?,?,?,?)",
                       (search["id"], frame_id, revision + 1, _json(event)))
        return event

    def apply(self, search: dict) -> dict:
        """Return a reviewed view; retain but never apply stale/in-flight reviews.

        Rejections are explicit metadata so resuming a partially reviewed job
        cannot make its entire case unreadable. A changed or missing image is
        also a failed point in the effective timeline, never trusted evidence.
        """
        output = deepcopy(search)
        with self._db() as db:
            history = [json.loads(row[0]) for row in db.execute(
                "SELECT event FROM reviews WHERE search_id=? ORDER BY sequence", (search.get("id"),))]
        current = {event["frame_id"]: deepcopy(event) for event in history}
        rejections, invalid_evidence = [], set()
        for selected, event in current.items():
            event["state"] = "reset" if event["verdict"] == "reset" else "active"
            if search.get("status") not in REVIEWABLE_STATUSES:
                event["state"] = "deferred"
                rejections.append({"frame_id": selected, "revision": event["revision"], "state": "deferred",
                                   "reason": "检索正在执行；复核覆盖将在暂停或完成后重新验证。"})
                continue
            try:
                binding = _binding(search, selected)
            except ValueError as exc:
                invalid_evidence.add(selected)
                event["state"] = "stale"
                rejections.append({"frame_id": selected, "revision": event["revision"], "state": "stale", "reason": str(exc)})
                continue
            if event["binding"] != binding:
                event["state"] = "stale"
                rejections.append({"frame_id": selected, "revision": event["revision"], "state": "stale",
                                   "reason": "目标、模型或原始结果已改变，旧复核未应用。填写原因并 reset 后可重新复核。"})
        effective = deepcopy(search.get("results", []))
        for result in effective:
            selected = frame_id(result.get("frame") or {})
            result["frame"]["id"] = selected
            event = current.get(selected)
            result["judgment_source"] = "model"
            if selected in invalid_evidence:
                result.update(status="failed", structured=None, error="原始证据无法验证；旧复核未应用。", judgment_source="invalid_evidence")
            elif event and event["state"] == "active":
                result["status"] = "ok"
                result["structured"] = {"verdict": event["verdict"], "reason": event["note"]}
                result["error"] = None
                result["model"] = result.get("model") or event["binding"]["model"]
                result["digest"] = result.get("digest") or event["binding"]["digest"]
                if result.get("target") is None:
                    known_targets = {r.get("target") for r in search.get("results", []) if r.get("status") == "ok" and r.get("target") is not None}
                    if len(known_targets) == 1:
                        result["target"] = next(iter(known_targets))
                result["judgment_source"] = event["actor"]
                result["review_revision"] = event["revision"]
        interval_warning = None
        try:
            intervals = derive_intervals(effective, max_gap_seconds=_max_gap(search))
        except ValueError as exc:
            if str(exc) != "Cannot combine different targets or successful model identities":
                raise
            # A failed inference can retain the digest of weights changed during
            # the call. A person can still inspect its intact pixels, but that
            # must not rewrite the recorded model identity or combine unlike
            # inference identities into one temporal claim.
            interval_warning = "当前判读包含不同目标或模型身份；逐帧复核仍保留，但不合并推导时间区间。请新建使用一致模型身份的检索后重新核验。"
            intervals = derive_intervals([], max_gap_seconds=_max_gap(search))
            intervals["derivation_error"] = interval_warning
            intervals["uncertain_or_failed"] = [
                {"frame_id": (result.get("frame") or {}).get("id"),
                 "time_seconds": (result.get("frame") or {}).get("time_seconds"),
                 "status": "identity_conflict", "reason": interval_warning}
                for result in effective]
        output.update(review_history=history, review_current=current, review_rejections=rejections, effective_results=effective,
                      reviewed_intervals=intervals, review_interval_warning=interval_warning,
                      review_summary={"revisions": len(history), "active_user": sum(e["actor"] == "user" and e["state"] == "active" for e in current.values()),
                                      "active_developer_assistant": sum(e["actor"] == "developer_assistant" and e["state"] == "active" for e in current.values()),
                                      "rejected_or_deferred": len(rejections)})
        return output
