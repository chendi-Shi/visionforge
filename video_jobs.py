"""Persistent local video jobs with one cross-process inference worker."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
from typing import Any
import uuid

from filelock import FileLock, Timeout


def default_data_dir() -> Path:
    override = os.environ.get("VISIONFORGE_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share")))
    return local / "VisionForge" / "video-data"


def _json(value: Any) -> str:
    def paths(item: Any) -> str:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=paths)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model: str) -> dict[str, str]:
    import core
    import core_video
    client = core._create_client()
    try:
        resolved, digest = core._check_model(client, model)
    except ValueError:
        raise
    except Exception as exc:
        raise core._call_error(exc) from None
    finally:
        client.close()
    if not digest:
        raise ValueError("Local model has no digest; cannot safely resume this job.")
    return {
        "model": resolved,
        "digest": digest,
        "core_sha256": _file_sha256(Path(core.__file__)),
        "core_video_sha256": _file_sha256(Path(core_video.__file__)),
    }


def _prepare_video(*args: Any, **kwargs: Any) -> dict:
    from video_media import prepare_video
    return prepare_video(*args, **kwargs)


def _analyze_segment(*args: Any, **kwargs: Any) -> dict:
    from core_video import analyze_segment
    return analyze_segment(*args, **kwargs)


class VideoEngine:
    """SQLite is the source of truth; only the process holding worker.lock works."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or default_data_dir()).expanduser().resolve()
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.database = self.root / "jobs.sqlite3"
        self._lock = FileLock(str(self.root / "worker.lock"), timeout=0, thread_local=False)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = threading.Lock()
        with self._db() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, source_path TEXT NOT NULL,
                    status TEXT NOT NULL, error TEXT, message TEXT NOT NULL DEFAULT '',
                    config TEXT NOT NULL, identity TEXT, video TEXT, selection TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segments (
                    job_id TEXT NOT NULL REFERENCES jobs(id), id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, start_seconds REAL NOT NULL, end_seconds REAL NOT NULL,
                    frames TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', result TEXT,
                    error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    review_decision TEXT NOT NULL DEFAULT 'unreviewed', review_text TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(job_id, id)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    job_id TEXT NOT NULL, segment_id TEXT NOT NULL, attempt INTEGER NOT NULL,
                    status TEXT NOT NULL, result TEXT, error TEXT, finished_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, segment_id, attempt)
                );
            """)

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @property
    def worker_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._lock.is_locked)

    def start(self) -> bool:
        """Return False for a submitting/reading client when another owner exists."""
        with self._lifecycle:
            if self.worker_alive:
                return True
            try:
                self._lock.acquire(timeout=0)
            except Timeout:
                return False
            try:
                with self._db() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("UPDATE segments SET status='pending' WHERE status='running'")
                    connection.execute(
                        "UPDATE jobs SET status='paused',message=?,updated_at=? WHERE status IN ('running','pause_requested')",
                        ("Worker interrupted; click Continue to resume unfinished segments.", _now()),
                    )
                    connection.execute(
                        "UPDATE jobs SET status='queued_prepare',message=?,updated_at=? WHERE status='preparing'",
                        ("Resuming interrupted video preparation.", _now()),
                    )
                self._stop.clear()
                self._wake.clear()
                self._thread = threading.Thread(target=self._work, name="VisionForge-video-worker", daemon=True)
                self._thread.start()
                return True
            except BaseException:
                self._lock.release()
                raise

    def stop(self) -> None:
        """Finish the in-flight segment, persist it, and release ownership."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join()

    def create_job(self, source_path: str | Path, model: str, focus: str,
                   max_segments: int = 4, max_edge: int = 448, max_tokens: int = 384) -> str:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Video source does not exist or is not a regular file.")
        if source.stat().st_size > 1024 ** 3:
            raise ValueError("Video source must not exceed 1 GB.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A local model is required.")
        if not isinstance(focus, str) or not 1 <= len(focus.strip()) <= 2000:
            raise ValueError("Focus must contain 1 to 2000 characters.")
        for name, value, lower, upper in (("max_segments", max_segments, 2, 12), ("max_edge", max_edge, 224, 768), ("max_tokens", max_tokens, 128, 1024)):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer from {lower} to {upper}")
        job_id = uuid.uuid4().hex
        folder = self.root / "jobs" / job_id
        folder.mkdir()
        target = folder / ("source" + source.suffix.lower())
        try:
            shutil.copy2(source, target)
            config = {"model": model.strip(), "focus": focus.strip(), "max_segments": max_segments,
                      "frames_per_segment": 2, "max_edge": max_edge, "max_tokens": max_tokens}
            with self._db() as connection:
                connection.execute(
                    "INSERT INTO jobs(id,name,source_path,status,config,created_at,updated_at,message) VALUES(?,?,?,?,?,?,?,?)",
                    (job_id, source.name, str(target), "queued_prepare", _json(config), _now(), _now(), "Queued for video preparation."),
                )
        except BaseException:
            target.unlink(missing_ok=True)
            folder.rmdir()
            raise
        self._wake.set()
        return job_id

    def list_jobs(self) -> list[dict]:
        with self._db() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._decode_job(row) for row in rows]

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict:
        job = dict(row)
        for field in ("config", "identity", "video", "selection"):
            job[field] = json.loads(job[field]) if job[field] else None
        return job

    def get_job(self, job_id: str) -> dict:
        with self._db() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown video job: {job_id}")
            job = self._decode_job(row)
            segments = connection.execute("SELECT * FROM segments WHERE job_id=? ORDER BY ordinal", (job_id,)).fetchall()
            history = connection.execute("SELECT * FROM attempts WHERE job_id=? ORDER BY attempt", (job_id,)).fetchall()
        job["segments"] = []
        for row in segments:
            segment = dict(row)
            segment["frames"] = json.loads(segment["frames"])
            segment["result"] = json.loads(segment["result"]) if segment["result"] else None
            segment["review"] = {"decision": segment.pop("review_decision"), "text": segment.pop("review_text")}
            segment["attempt_history"] = [
                {**dict(item), "result": json.loads(item["result"]) if item["result"] else None}
                for item in history if item["segment_id"] == segment["id"]
            ]
            job["segments"].append(segment)
        return job

    def analyze(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job["status"] in {"completed", "queued_analysis", "running", "pause_requested"}:
            return job
        if job["status"] == "failed" and not job["segments"]:
            with self._db() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE jobs SET status='queued_prepare',error=NULL,message=?,updated_at=? "
                    "WHERE id=? AND status='failed' AND NOT EXISTS (SELECT 1 FROM segments WHERE job_id=?)",
                    ("Retrying preparation only. When frames are ready, click Analyze again to start the model.", _now(), job_id, job_id),
                )
            self._wake.set()
            return self.get_job(job_id)
        if job["status"] not in {"ready", "paused", "partial", "failed"} or not job["segments"]:
            raise ValueError("Prepare the video first; analysis requires a ready job with sampled segments.")
        for segment in job["segments"]:
            if segment["status"] != "ok":
                continue
            for frame in segment["frames"]:
                try:
                    if not frame.get("sha256") or _file_sha256(Path(frame["path"])) != frame["sha256"]:
                        raise ValueError("Evidence frame content has changed.")
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Successful segment evidence is missing or changed. Create a new job; old results cannot be reused.") from exc
        identity = _model_identity(job["config"]["model"])
        identity["config_sha256"] = hashlib.sha256(_json(job["config"]).encode("utf-8")).hexdigest()
        if job["identity"] and job["identity"] != identity:
            raise ValueError("Model digest, generation configuration, or core_video code changed. Create a new job; old and new results cannot be mixed.")
        with self._db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT status,config,identity FROM jobs WHERE id=?", (job_id,)).fetchone()
            if current["status"] not in {"ready", "paused", "partial", "failed"}:
                return self.get_job(job_id)
            if current["config"] != _json(job["config"]) or (current["identity"] and json.loads(current["identity"]) != identity):
                raise ValueError("Job configuration changed; create a new job.")
            connection.execute("UPDATE segments SET status='pending' WHERE job_id=? AND status='failed'", (job_id,))
            connection.execute(
                "UPDATE jobs SET status='queued_analysis',identity=?,error=NULL,message=?,updated_at=? WHERE id=?",
                (_json(identity), "Queued for analysis; completed segments will be reused.", _now(), job_id),
            )
        self._wake.set()
        return self.get_job(job_id)

    def pause(self, job_id: str) -> dict:
        with self._db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(f"Unknown video job: {job_id}")
            if row["status"] in {"queued_analysis", "running", "pause_requested"}:
                state = "paused" if row["status"] == "queued_analysis" else "pause_requested"
                connection.execute("UPDATE jobs SET status=?,message=?,updated_at=? WHERE id=?",
                                   (state, "Pause requested; the current segment will finish first.", _now(), job_id))
        self._wake.set()
        return self.get_job(job_id)

    def review_segment(self, job_id: str, segment_id: str, decision: str, text: str) -> dict:
        if not isinstance(decision, str) or decision not in {"unreviewed", "confirmed", "corrected", "rejected"} or not isinstance(text, str):
            raise ValueError("Review decision must be unreviewed, confirmed, corrected or rejected; text must be a string.")
        if decision == "corrected" and not text.strip():
            raise ValueError("A corrected review requires non-empty correction text.")
        with self._db() as connection:
            cursor = connection.execute("UPDATE segments SET review_decision=?,review_text=? WHERE job_id=? AND id=?",
                                        (decision.strip(), text, job_id, segment_id))
            if not cursor.rowcount:
                raise KeyError(f"Unknown segment: {segment_id}")
            connection.execute("UPDATE jobs SET updated_at=? WHERE id=?", (_now(), job_id))
        return self.get_job(job_id)

    def _message(self, job_id: str, stage: str, message: str) -> None:
        with self._db() as connection:
            connection.execute("UPDATE jobs SET message=?,updated_at=? WHERE id=?", (f"{stage}: {message}", _now(), job_id))

    def _prepare(self, job: dict) -> None:
        job_id = job["id"]
        with self._db() as connection:
            connection.execute("UPDATE jobs SET status='preparing',error=NULL,updated_at=? WHERE id=?", (_now(), job_id))
        config = job["config"]
        prepared = _prepare_video(Path(job["source_path"]), self.root / "jobs" / job_id,
                                  max_segments=config["max_segments"], frames_per_segment=config["frames_per_segment"],
                                  progress_callback=lambda stage, message: self._message(job_id, stage, message))
        segments = prepared.get("segments", [])
        if not isinstance(segments, list) or not segments:
            raise ValueError("Video preparation returned no sampled segments.")
        with self._db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for index, segment in enumerate(segments):
                if not segment.get("frames"):
                    raise ValueError("A prepared segment has no evidence frames.")
                connection.execute(
                    "INSERT INTO segments(job_id,id,ordinal,start_seconds,end_seconds,frames) VALUES(?,?,?,?,?,?)",
                    (job_id, segment["id"], index, segment["start_seconds"], segment["end_seconds"], _json(segment["frames"])),
                )
            connection.execute("UPDATE jobs SET status='ready',video=?,selection=?,message=?,updated_at=? WHERE id=?",
                               (_json(prepared.get("video", {})), _json(prepared.get("selection", {})),
                                "Frames are ready. Review the sampling and click Analyze to start the model.", _now(), job_id))

    def _finish_state(self, connection: sqlite3.Connection, job_id: str, pause: bool = False) -> None:
        rows = connection.execute("SELECT status FROM segments WHERE job_id=?", (job_id,)).fetchall()
        statuses = [row["status"] for row in rows]
        if statuses and all(status == "ok" for status in statuses):
            status, message = "completed", "All sampled segments completed. Model claims still need human review."
        elif pause and "pending" in statuses:
            status, message = "paused", "Paused. Click Continue to run unfinished segments."
        elif "pending" in statuses:
            status, message = "running", "Analyzing sampled segments."
        else:
            status = "partial" if "ok" in statuses else "failed"
            message = "Some segments failed. Inspect their raw results and retry unfinished segments."
        connection.execute("UPDATE jobs SET status=?,message=?,updated_at=? WHERE id=?", (status, message, _now(), job_id))

    def _analyze_one(self, job: dict) -> None:
        job_id = job["id"]
        with self._db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()["status"]
            if state not in {"queued_analysis", "running", "pause_requested"}:
                return
            if state == "pause_requested" or self._stop.is_set():
                self._finish_state(connection, job_id, pause=True)
                return
            segment = connection.execute("SELECT * FROM segments WHERE job_id=? AND status='pending' ORDER BY ordinal LIMIT 1", (job_id,)).fetchone()
            if segment is None:
                self._finish_state(connection, job_id)
                return
            attempt = segment["attempts"] + 1
            connection.execute("UPDATE segments SET status='running',attempts=?,error=NULL WHERE job_id=? AND id=?", (attempt, job_id, segment["id"]))
            connection.execute("UPDATE jobs SET status='running',message=?,updated_at=? WHERE id=?", (f"Analyzing {segment['id']} (attempt {attempt}).", _now(), job_id))
        error = None
        try:
            config = job["config"]
            result = _analyze_segment(json.loads(segment["frames"]), focus=config["focus"], model=job["identity"]["model"],
                                      max_edge=config["max_edge"], max_tokens=config["max_tokens"])
            if not isinstance(result, dict):
                raise ValueError("Segment analyzer returned a non-object result.")
            _json(result)
            if result.get("digest") != job["identity"]["digest"]:
                error = "Model digest differs from this job's fixed identity. Create a new job."
            elif result.get("status") != "ok":
                error = f"Model result status: {result.get('status', 'missing')}"
            elif not isinstance(result.get("summary"), str) or not result["summary"].strip():
                error = "Model returned no summary."
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = {"status": "error", "error": error}
        segment_status = "failed" if error else "ok"
        with self._db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE segments SET status=?,result=?,error=? WHERE job_id=? AND id=?", (segment_status, _json(result), error, job_id, segment["id"]))
            connection.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?)", (job_id, segment["id"], attempt, segment_status, _json(result), error, _now()))
            state = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()["status"]
            if error:
                connection.execute("UPDATE jobs SET error=? WHERE id=?", (error, job_id))
            self._finish_state(connection, job_id, pause=state == "pause_requested" or self._stop.is_set())

    def _work(self) -> None:
        try:
            while not self._stop.is_set():
                with self._db() as connection:
                    row = connection.execute("SELECT * FROM jobs WHERE status IN ('queued_prepare','queued_analysis','running','pause_requested') ORDER BY created_at LIMIT 1").fetchone()
                if row is None:
                    self._wake.wait(0.3)
                    self._wake.clear()
                    continue
                job = self._decode_job(row)
                try:
                    self._prepare(job) if job["status"] == "queued_prepare" else self._analyze_one(job)
                except Exception as exc:
                    with self._db() as connection:
                        connection.execute("UPDATE segments SET status='pending' WHERE job_id=? AND status='running'", (job["id"],))
                        connection.execute("UPDATE jobs SET status='failed',error=?,message=?,updated_at=? WHERE id=?",
                                           (f"{type(exc).__name__}: {exc}", "Worker could not finish this job; inspect the error.", _now(), job["id"]))
        finally:
            try:
                with self._db() as connection:
                    connection.execute("UPDATE jobs SET status='paused',message=?,updated_at=? WHERE status IN ('running','pause_requested')",
                                       ("Worker stopped; click Continue to resume.", _now()))
            finally:
                self._lock.release()
