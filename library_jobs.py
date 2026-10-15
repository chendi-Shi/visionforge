"""Local video library and durable, budgeted searches across selected videos.

The parent, children and their mapping share SearchEngine's SQLite transaction.
There is no dispatch worker or retry window that can create duplicate children.
Model work remains serialized by the existing SearchEngine worker.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import uuid

from search_jobs import DEFAULT_SEARCH_MODEL, SearchEngine
from video_jobs import VideoEngine, _json, _now


class LibraryEngine:
    def __init__(self, videos: VideoEngine | None = None, searches: SearchEngine | None = None):
        self.videos = videos or (searches.videos if searches else VideoEngine())
        self.searches = searches or SearchEngine(self.videos)
        if self.videos.root.resolve() != self.searches.root.resolve():
            raise ValueError("视频库与检索必须使用同一个数据目录。")
        self.root = self.videos.root
        self.database = self.searches.database
        with self.searches._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS library_videos (
                    job_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]', archived INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS library_searches (
                    id TEXT PRIMARY KEY, target TEXT NOT NULL, retrieval_query TEXT NOT NULL,
                    model TEXT NOT NULL, config TEXT NOT NULL, control_state TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS library_search_members (
                    run_id TEXT NOT NULL, job_id TEXT NOT NULL, search_id TEXT NOT NULL UNIQUE,
                    ordinal INTEGER NOT NULL, title TEXT NOT NULL,
                    PRIMARY KEY(run_id,job_id), UNIQUE(run_id,ordinal));
                CREATE INDEX IF NOT EXISTS library_members_run ON library_search_members(run_id);
            """)

    @property
    def worker_alive(self):
        return self.searches.worker_alive

    def start(self):
        return self.searches.start()

    def stop(self):
        self.searches.stop()

    @staticmethod
    def _video(job, metadata=None):
        metadata = dict(metadata or {})
        video = job.get("video") or {}
        return {**job, "title": metadata.get("title") or job["name"],
                "tags": json.loads(metadata.get("tags", "[]")),
                "archived": bool(metadata.get("archived", False)),
                "ready_for_search": bool(video.get("preview_path") and video.get("preview_sha256")),
                "library_updated_at": metadata.get("updated_at")}

    def list_videos(self, include_archived: bool = False) -> list[dict]:
        with self.searches._db() as db:
            metadata = {row["job_id"]: row for row in db.execute("SELECT * FROM library_videos")}
        rows = [self._video(job, metadata.get(job["id"])) for job in self.videos.list_jobs()]
        return rows if include_archived else [row for row in rows if not row["archived"]]

    def update_video(self, job_id: str, title: str | None = None,
                     tags: list[str] | None = None, archived: bool | None = None) -> dict:
        job = self.videos.get_job(job_id)
        if title is not None and (not isinstance(title, str) or not 1 <= len(title.strip()) <= 200):
            raise ValueError("视频名称须为1–200字。")
        if tags is not None:
            if not isinstance(tags, (list, tuple)) or len(tags) > 20:
                raise ValueError("每段视频最多20个标签，请使用标签列表。")
            cleaned, seen = [], set()
            for tag in tags:
                if not isinstance(tag, str) or not 1 <= len(tag.strip()) <= 40:
                    raise ValueError("每个标签须为1–40字。")
                tag = tag.strip()
                if tag.casefold() not in seen:
                    cleaned.append(tag)
                    seen.add(tag.casefold())
            tags = cleaned
        if archived is not None and not isinstance(archived, bool):
            raise ValueError("归档状态必须是布尔值。")
        with self.searches._db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT * FROM library_videos WHERE job_id=?", (job_id,)).fetchone()
            current = self._video(job, prior)
            values = (job_id, title.strip() if title is not None else current["title"],
                      _json(tags if tags is not None else current["tags"]),
                      int(archived if archived is not None else current["archived"]), _now())
            db.execute("INSERT INTO library_videos VALUES(?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET "
                       "title=excluded.title,tags=excluded.tags,archived=excluded.archived,updated_at=excluded.updated_at", values)
            row = db.execute("SELECT * FROM library_videos WHERE job_id=?", (job_id,)).fetchone()
        return self._video(job, row)

    def import_videos(self, paths, model: str = DEFAULT_SEARCH_MODEL) -> dict:
        """Queue preprocessing only; return each accepted file and each error."""
        if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 50:
            raise ValueError("每批请选择1–50段视频。")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("请选择本地视觉模型。")
        items, job_ids, seen = [], [], set()
        for source in paths:
            item = {"path": str(source), "job_id": None, "status": "failed", "error": None}
            try:
                path = Path(source).expanduser().resolve()
                if path in seen:
                    raise ValueError("同一批次中重复选择了这个文件。")
                seen.add(path)
                job_id = self.videos.create_job(path, model.strip(),
                    "预处理供视频库检索使用；摘要分析按需启动。")
                job_ids.append(job_id)
                item.update(job_id=job_id, status="queued_prepare")
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            items.append(item)
        return {"job_ids": job_ids, "items": items, "succeeded": len(job_ids),
                "failed": len(items) - len(job_ids)}

    def create_search(self, job_ids: list[str], target: str, retrieval_query: str = "",
                      model: str = DEFAULT_SEARCH_MODEL, max_total_frames: int = 48, **config) -> str:
        if not isinstance(job_ids, (list, tuple)) or not 1 <= len(job_ids) <= 50:
            raise ValueError("请选中1–50段已预处理的视频。")
        if any(not isinstance(job_id, str) or not job_id for job_id in job_ids):
            raise ValueError("视频编号无效。")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("同一批检索中不能重复选择视频。")
        if isinstance(max_total_frames, bool) or not isinstance(max_total_frames, int) or not 3 <= max_total_frames <= 480:
            raise ValueError("整批核验预算须为3–480帧。")
        # Validate the caller's per-video budget before allocating a global cap.
        # prepare_search is pure validation/snapshotting: no model or DB writes.
        prepared = [self.searches.prepare_search(job_id, target, retrieval_query, model, **config)
                    for job_id in job_ids]
        requested_config = dict(prepared[0]["config"])
        minimum = max(3, prepared[0]["config"]["top_k"])
        if max_total_frames < len(job_ids) * minimum:
            raise ValueError(f"整批预算至少需要 {len(job_ids) * minimum} 帧（每段至少 {minimum} 帧）。")
        cap, remainder = divmod(max_total_frames, len(job_ids))
        for ordinal, payload in enumerate(prepared):
            payload["config"]["max_frames"] = min(payload["config"]["max_frames"], cap + (ordinal < remainder))
        run_id, now = uuid.uuid4().hex, _now()
        run_config = {**requested_config, "max_total_frames": max_total_frames,
                      "allocated_frame_budget": sum(p["config"]["max_frames"] for p in prepared)}
        with self.searches._db() as db:
            db.execute("BEGIN IMMEDIATE")
            metadata = {row["job_id"]: row for row in db.execute("SELECT * FROM library_videos")}
            for job_id in job_ids:
                if metadata.get(job_id) and metadata[job_id]["archived"]:
                    raise ValueError("所选视频已归档，请先恢复视频再检索。")
            db.execute("INSERT INTO library_searches VALUES(?,?,?,?,?,?,?,?)",
                       (run_id, target.strip(), retrieval_query.strip(), model.strip(), _json(run_config), "running", now, now))
            for ordinal, (job_id, payload) in enumerate(zip(job_ids, prepared)):
                search_id = uuid.uuid4().hex
                self.searches.insert_prepared(db, search_id, job_id, payload)
                title = metadata[job_id]["title"] if job_id in metadata else payload["job_name"]
                db.execute("INSERT INTO library_search_members VALUES(?,?,?,?,?)",
                           (run_id, job_id, search_id, ordinal, title))
        self.searches._wake.set()
        return run_id

    @staticmethod
    def _status(statuses, control_state):
        if not statuses:
            return "failed"
        if all(status == "completed" for status in statuses):
            return "completed"
        if any(status in {"running", "pause_requested"} for status in statuses):
            return "pause_requested" if control_state == "paused" else "running"
        if any(status == "queued" for status in statuses):
            return "queued" if all(status == "queued" for status in statuses) else "running"
        if any(status == "paused" for status in statuses):
            return "paused"
        if all(status == "failed" for status in statuses):
            return "failed"
        return "partial"

    def _get_search(self, db, run_id: str) -> dict:
        row = db.execute("SELECT * FROM library_searches WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("未知视频库检索。")
        run = dict(row)
        run["config"] = json.loads(run["config"])
        members = db.execute("SELECT * FROM library_search_members WHERE run_id=? ORDER BY ordinal", (run_id,)).fetchall()
        children, results, intervals, metadata_updates = [], [], [], []
        for member in members:
            child_row = db.execute("SELECT * FROM searches WHERE id=?", (member["search_id"],)).fetchone()
            child = self.searches._decode(child_row) if child_row else {
                "id": member["search_id"], "status": "failed", "results": [], "frames": [],
                "error": "子检索记录缺失，已有映射已保留；请检查数据文件。", "message": "子检索记录缺失。"}
            title_row = db.execute("SELECT title,updated_at FROM library_videos WHERE job_id=?", (member["job_id"],)).fetchone()
            title = title_row["title"] if title_row else member["title"]
            if title_row:
                metadata_updates.append(title_row["updated_at"])
            context = {"job_id": member["job_id"], "search_id": member["search_id"], "title": title,
                       "source_sha256": (child.get("video") or {}).get("preview_sha256")}
            verified = sum(result.get("status") == "ok" for result in child.get("results", []))
            planned = len(child.get("frames", []))
            children.append({**context, "status": child["status"], "message": child.get("message"),
                             "error": child.get("error"), "verified_frames": verified,
                             "planned_frames": planned, "search": child})
            for ordinal, result in enumerate(child.get("results", [])):
                frame_id = (result.get("frame") or {}).get("id")
                results.append({**result, **context, "frame_id": frame_id,
                    "key": f"{member['job_id']}:{member['search_id']}:{frame_id or 'record-' + str(ordinal)}"})
            intervals.append({**context, "intervals": child.get("intervals") or {}})
        statuses = [child["status"] for child in children]
        counts = Counter(statuses)
        run.update(status=self._status(statuses, run["control_state"]), job_ids=[m["job_id"] for m in members],
                   children=children, results=results, intervals=intervals,
                   progress={"videos_total": len(children), "videos_completed": counts["completed"],
                             "videos_failed": counts["failed"], "videos_partial": counts["partial"],
                             "videos_paused": counts["paused"],
                             "frames_verified": sum(c["verified_frames"] for c in children),
                             "frames_planned": sum(c["planned_frames"] for c in children),
                             "frame_budget": run["config"]["allocated_frame_budget"]})
        run["updated_at"] = max([run["updated_at"]] + metadata_updates + [c["search"].get("updated_at", "") for c in children])
        run["errors"] = [{"job_id": c["job_id"], "search_id": c["search_id"], "title": c["title"],
                          "error": c["error"]} for c in children if c["error"]]
        return run

    def get_search(self, run_id: str) -> dict:
        with self.searches._db() as db:
            db.execute("BEGIN")
            return self._get_search(db, run_id)

    def list_searches(self) -> list[dict]:
        with self.searches._db() as db:
            db.execute("BEGIN")
            ids = [row["id"] for row in db.execute("SELECT id FROM library_searches ORDER BY created_at DESC,rowid DESC")]
            return [self._get_search(db, run_id) for run_id in ids]

    def pause(self, run_id: str) -> dict:
        with self.searches._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM library_searches WHERE id=?", (run_id,)).fetchone():
                raise KeyError("未知视频库检索。")
            now = _now()
            db.execute("UPDATE library_searches SET control_state='paused',updated_at=? WHERE id=?", (now, run_id))
            db.execute("UPDATE searches SET status=CASE WHEN status='queued' THEN 'paused' ELSE 'pause_requested' END,"
                       "message=?,updated_at=? WHERE id IN (SELECT search_id FROM library_search_members WHERE run_id=?) "
                       "AND status IN ('queued','running')", ("视频库检索已请求暂停；当前核验结束后保存。", now, run_id))
        self.searches._wake.set()
        return self.get_search(run_id)

    def resume(self, run_id: str) -> dict:
        with self.searches._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM library_searches WHERE id=?", (run_id,)).fetchone():
                raise KeyError("未知视频库检索。")
            now = _now()
            db.execute("UPDATE library_searches SET control_state='running',updated_at=? WHERE id=?", (now, run_id))
            db.execute("UPDATE searches SET status='queued',error=NULL,message=?,updated_at=? "
                       "WHERE id IN (SELECT search_id FROM library_search_members WHERE run_id=?) "
                       "AND status IN ('paused','partial','failed')",
                       ("等待继续；保留已成功核验的帧。", now, run_id))
        self.searches._wake.set()
        return self.get_search(run_id)
