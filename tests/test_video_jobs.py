"""Offline state-machine tests: fake media and model; no Ollama or decoding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_jobs import VideoEngine, _json


class VideoJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "clip.mp4"
        self.source.write_bytes(b"fake-video")
        self.engines = []
        self.calls = []
        self.identity = {"model": "fake:local", "digest": "weights-a", "core_sha256": "core-a", "core_video_sha256": "code-a"}
        self.patches = [patch("video_jobs._prepare_video", side_effect=self.prepare),
                        patch("video_jobs._model_identity", side_effect=lambda model: dict(self.identity)),
                        patch("video_jobs._analyze_segment", side_effect=self.analyzer)]
        for mock in self.patches:
            mock.start()

    def tearDown(self):
        for engine in self.engines:
            engine.stop()
        for mock in reversed(self.patches):
            mock.stop()
        self.temporary.cleanup()

    def engine(self):
        engine = VideoEngine(self.base / "data")
        self.engines.append(engine)
        return engine

    def prepare(self, source, folder, **kwargs):
        kwargs["progress_callback"]("sampling", "fake frames")
        for i in (1, 2):
            (folder / f"frame{i}.jpg").write_bytes(f"fake-frame-{i}".encode("ascii"))
        return {"video": {"duration_seconds": 20}, "selection": {"candidate": 2, "selected": 2, "unselected": 0, "all_windows": []},
                "segments": [{"id": f"S000{i}", "start_seconds": (i - 1) * 10, "end_seconds": i * 10,
                              "frames": [{"id": f"S000{i}-F01", "time_seconds": (i - 1) * 10, "path": str(folder / f"frame{i}.jpg"),
                                          "sha256": hashlib.sha256((folder / f"frame{i}.jpg").read_bytes()).hexdigest()}]} for i in (1, 2)]}

    def analyzer(self, frames, **kwargs):
        self.calls.append(frames[0]["id"])
        return {"status": "ok", "summary": "Original model summary", "observations": [], "digest": self.identity["digest"]}

    def wait_status(self, engine, job_id, expected):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = engine.get_job(job_id)
            if job["status"] in expected:
                return job
            time.sleep(0.015)
        self.fail(f"Job failed to reach {expected}: {engine.get_job(job_id)}")

    def new_ready_job(self, engine):
        job_id = engine.create_job(self.source, "fake:local", "visible events")
        engine.start()
        self.wait_status(engine, job_id, {"ready"})
        return job_id

    def test_single_worker_ready_gate_and_separate_human_review(self):
        first, second = self.engine(), self.engine()
        self.assertTrue(first.start())
        self.assertFalse(second.start())
        for override in ({"focus": " "}, {"focus": "x" * 2001}, {"max_segments": 1}, {"max_edge": 100}, {"max_tokens": 1025}):
            with self.assertRaises(ValueError), patch("video_jobs.shutil.copy2") as copy:
                second.create_job(self.source, **{"model": "fake:local", "focus": "visible events", **override})
            copy.assert_not_called()
        job_id = second.create_job(self.source, "fake:local", "visible events")
        self.wait_status(first, job_id, {"ready"})
        self.assertEqual(self.calls, [])
        second.analyze(job_id)
        job = self.wait_status(second, job_id, {"completed"})
        self.assertEqual(len(self.calls), 2)
        original = job["segments"][0]["result"]
        with self.assertRaises(ValueError):
            second.review_segment(job_id, "S0001", "unknown", "text")
        with self.assertRaises(ValueError):
            second.review_segment(job_id, "S0001", "corrected", " ")
        corrected = second.review_segment(job_id, "S0001", "corrected", "Human correction")
        self.assertEqual(corrected["segments"][0]["result"], original)
        self.assertEqual(corrected["segments"][0]["review"]["text"], "Human correction")
        second.analyze(job_id)
        self.assertEqual(len(self.calls), 2)

    def test_pause_finishes_current_segment_and_resume_skips_success(self):
        engine = self.engine()
        job_id = self.new_ready_job(engine)
        entered, release = threading.Event(), threading.Event()

        def slow(frames, **kwargs):
            entered.set()
            release.wait(5)
            return self.analyzer(frames, **kwargs)

        with patch("video_jobs._analyze_segment", side_effect=slow):
            engine.analyze(job_id)
            self.assertTrue(entered.wait(3))
            self.assertEqual(engine.pause(job_id)["status"], "pause_requested")
            release.set()
            job = self.wait_status(engine, job_id, {"paused"})
        self.assertEqual([s["status"] for s in job["segments"]], ["ok", "pending"])
        engine.analyze(job_id)
        self.wait_status(engine, job_id, {"completed"})
        self.assertEqual(self.calls, ["S0001-F01", "S0002-F01"])

    def test_failed_outputs_and_digest_changes_are_not_mixed(self):
        engine = self.engine()
        job_id = self.new_ready_job(engine)

        def incomplete(frames, **kwargs):
            result = self.analyzer(frames, **kwargs)
            if "S0002" in frames[0]["id"]:
                result["status"] = "truncated"
                result["raw_response"] = '{"summary":"incomplete'
            return result

        with patch("video_jobs._analyze_segment", side_effect=incomplete):
            engine.analyze(job_id)
            job = self.wait_status(engine, job_id, {"partial"})
        self.assertEqual(job["segments"][1]["status"], "failed")
        self.assertEqual(job["segments"][1]["result"]["status"], "truncated")
        self.identity["digest"] = "weights-b"
        with self.assertRaisesRegex(ValueError, "Create a new job"):
            engine.analyze(job_id)
        self.identity["digest"] = "weights-a"
        self.identity["core_video_sha256"] = "code-b"
        with self.assertRaisesRegex(ValueError, "Create a new job"):
            engine.analyze(job_id)
        self.identity["core_video_sha256"] = "code-a"
        self.identity["core_sha256"] = "core-b"
        with self.assertRaisesRegex(ValueError, "Create a new job"):
            engine.analyze(job_id)
        self.identity["core_sha256"] = "core-a"
        evidence_path = Path(job["segments"][0]["frames"][0]["path"])
        original_bytes = evidence_path.read_bytes()
        evidence_path.write_bytes(b"altered-frame")
        with self.assertRaisesRegex(ValueError, "Create a new job"):
            engine.analyze(job_id)
        evidence_path.write_bytes(original_bytes)
        original_config = engine.get_job(job_id)["config"]
        with engine._db() as db:
            db.execute("UPDATE jobs SET config=? WHERE id=?", (_json({**original_config, "max_tokens": 512}), job_id))
        with self.assertRaisesRegex(ValueError, "Create a new job"):
            engine.analyze(job_id)
        with engine._db() as db:
            db.execute("UPDATE jobs SET config=? WHERE id=?", (_json(original_config), job_id))

        def wrong_digest(frames, **kwargs):
            return {**self.analyzer(frames, **kwargs), "digest": "unexpected-weights"}

        with patch("video_jobs._analyze_segment", side_effect=wrong_digest):
            engine.analyze(job_id)
            mismatched = self.wait_status(engine, job_id, {"partial"})
        self.assertIn("digest differs", mismatched["segments"][1]["error"])
        self.assertEqual(mismatched["segments"][1]["result"]["digest"], "unexpected-weights")
        engine.analyze(job_id)
        resumed = self.wait_status(engine, job_id, {"completed"})
        self.assertEqual(self.calls.count("S0001-F01"), 1)
        self.assertEqual(resumed["segments"][1]["attempts"], 3)
        self.assertEqual(resumed["segments"][1]["attempt_history"][0]["result"]["status"], "truncated")

    def test_restart_pauses_interrupted_segment_until_explicit_continue(self):
        engine = self.engine()
        job_id = self.new_ready_job(engine)
        engine.stop()
        config_hash = hashlib.sha256(_json(engine.get_job(job_id)["config"]).encode("utf-8")).hexdigest()
        identity = {**self.identity, "config_sha256": config_hash}
        with engine._db() as db:
            db.execute("UPDATE jobs SET status='running',identity=? WHERE id=?", (json.dumps(identity), job_id))
            db.execute("UPDATE segments SET status='running',attempts=1 WHERE job_id=? AND id='S0001'", (job_id,))
        restarted = self.engine()
        self.assertTrue(restarted.start())
        job = restarted.get_job(job_id)
        self.assertEqual(job["status"], "paused")
        self.assertEqual(job["segments"][0]["status"], "pending")
        self.assertEqual(self.calls, [])
        restarted.analyze(job_id)
        completed = self.wait_status(restarted, job_id, {"completed"})
        self.assertEqual(completed["segments"][0]["attempts"], 2)
        self.assertEqual(len(self.calls), 2)

    def test_failed_preparation_retries_to_ready_without_model_analysis(self):
        engine = self.engine()
        with patch("video_jobs._prepare_video", side_effect=ValueError("temporary preparation failure")):
            job_id = engine.create_job(self.source, "fake:local", "visible events")
            engine.start()
            failed = self.wait_status(engine, job_id, {"failed"})
        self.assertEqual(failed["segments"], [])
        with patch("video_jobs._model_identity", side_effect=AssertionError("preparation must not inspect or start a model")):
            engine.analyze(job_id)
            ready = self.wait_status(engine, job_id, {"ready"})
        self.assertEqual(len(ready["segments"]), 2)
        self.assertIsNone(ready["identity"])
        self.assertIsNone(ready["error"])
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
