"""Durable corpus orchestration; no video decoder, embedding or model calls."""
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from library_jobs import LibraryEngine
from search_jobs import SearchEngine


class LibraryJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.jobs = {f"v{i}": {"id": f"v{i}", "name": f"video-{i}.mp4", "status": "ready",
                    "video": {"preview_path": str(self.root / f"video-{i}.mp4"),
                              "preview_sha256": str(i) * 64, "duration_seconds": 12}}
                     for i in range(1, 5)}
        self.jobs["pending"] = {"id": "pending", "name": "new.mp4", "status": "preparing", "video": None}
        self.videos = types.SimpleNamespace(root=self.root, get_job=self.jobs.__getitem__,
            list_jobs=lambda: list(self.jobs.values()), create_job=self.import_one)
        self.searches = SearchEngine(self.videos)
        self.engine = LibraryEngine(self.videos, self.searches)
        self.imported = []

    def tearDown(self):
        self.searches.stop()
        self.temp.cleanup()

    def import_one(self, path, model, focus):
        if not path.is_file():
            raise ValueError("missing input")
        self.imported.append(str(path))
        return "import-" + str(len(self.imported))

    def create(self, ids=None, **kwargs):
        return self.engine.create_search(ids or ["v1", "v2"], "红色车辆", "red vehicle",
                                         model="fixture:local", **kwargs)

    def counts(self):
        with self.searches._db() as db:
            return tuple(db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                         for table in ("library_searches", "library_search_members", "searches"))

    def test_metadata_filter_restore_persist_and_no_original_mutation(self):
        value = self.engine.update_video("v1", title="入口摄像头", tags=["门口", " Traffic ", "traffic"], archived=True)
        self.assertEqual(value["tags"], ["门口", "Traffic"])
        self.assertTrue(value["archived"])
        self.assertEqual(self.jobs["v1"]["name"], "video-1.mp4")
        fresh = LibraryEngine(self.videos, SearchEngine(self.videos))
        self.assertNotIn("v1", [row["id"] for row in fresh.list_videos()])
        self.assertEqual(len(fresh.list_videos(include_archived=True)), 5)
        restored = fresh.update_video("v1", archived=False)
        self.assertEqual(restored["title"], "入口摄像头")
        self.assertEqual(restored["tags"], ["门口", "Traffic"])
        self.assertTrue(restored["ready_for_search"])
        self.assertFalse(next(row for row in fresh.list_videos() if row["id"] == "pending")["ready_for_search"])

    def test_metadata_validation_does_not_erase_prior_values(self):
        self.engine.update_video("v1", title="named", tags=["label"])
        for kwargs in ({"title": " "}, {"tags": "text"}, {"tags": [""]}, {"archived": 1}):
            with self.assertRaises(ValueError):
                self.engine.update_video("v1", **kwargs)
        row = next(row for row in self.engine.list_videos() if row["id"] == "v1")
        self.assertEqual((row["title"], row["tags"], row["archived"]), ("named", ["label"], False))

    def test_batch_import_reports_each_failure_keeps_success_and_no_model_call(self):
        source = self.root / "input.mp4"
        source.write_bytes(b"fixture")
        report = self.engine.import_videos([source, self.root / "missing.mp4", source])
        self.assertEqual((report["succeeded"], report["failed"]), (1, 2))
        self.assertEqual(report["job_ids"], ["import-1"])
        self.assertEqual(report["items"][0]["status"], "queued_prepare")
        self.assertIn("missing", report["items"][1]["error"])
        self.assertIn("重复", report["items"][2]["error"])

    def test_invalid_or_unprepared_video_rolls_back_entire_submission(self):
        for ids in (["v1", "pending"], ["v1", "missing"], ["v1", "v1"]):
            with self.assertRaises((ValueError, KeyError)):
                self.create(ids)
            self.assertEqual(self.counts(), (0, 0, 0))
        self.engine.update_video("v2", archived=True)
        with self.assertRaisesRegex(ValueError, "归档"):
            self.create()
        self.assertEqual(self.counts(), (0, 0, 0))

    def test_insert_failure_rolls_back_parent_children_and_mapping(self):
        original = self.searches.insert_prepared
        inserted = 0
        def fail_second(*args):
            nonlocal inserted
            inserted += 1
            original(*args)
            if inserted == 2:
                raise sqlite3.OperationalError("simulated write failure")
        with patch.object(self.searches, "insert_prepared", side_effect=fail_second):
            with self.assertRaises(sqlite3.OperationalError):
                self.create()
        self.assertEqual(self.counts(), (0, 0, 0))

    def test_reader_never_sees_parent_without_children_during_commit(self):
        entered, release = threading.Event(), threading.Event()
        original = self.searches.insert_prepared
        outcome = []
        def hold(*args):
            original(*args)
            entered.set()
            self.assertTrue(release.wait(30))
        def submit():
            try:
                outcome.append(self.create())
            except BaseException as exc:
                outcome.append(exc)
        with patch.object(self.searches, "insert_prepared", side_effect=hold):
            thread = threading.Thread(target=submit)
            thread.start()
            try:
                self.assertTrue(entered.wait(30))
                self.assertEqual(self.counts(), (0, 0, 0))
                self.assertEqual(self.engine.list_searches(), [])
            finally:
                release.set()
                thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], str)
        self.assertEqual(self.counts(), (1, 2, 2))

    def test_global_budget_allocation_is_persisted_and_bounded(self):
        run = self.engine.get_search(self.create(["v1", "v2", "v3"], max_total_frames=10, max_frames=12))
        budgets = [child["search"]["config"]["max_frames"] for child in run["children"]]
        self.assertEqual(budgets, [4, 3, 3])
        self.assertEqual(run["config"]["allocated_frame_budget"], 10)
        self.assertEqual(run["progress"]["frame_budget"], 10)
        with self.assertRaises(ValueError):
            self.create(["v1", "v2", "v3", "v4"], max_total_frames=11)
        with self.assertRaises(ValueError):
            self.create(max_total_frames=6, top_k=4)
        self.assertEqual(self.counts(), (1, 3, 3))

    def test_concurrent_submissions_keep_disjoint_child_ownership(self):
        barrier = threading.Barrier(3)
        outcomes = []
        def submit():
            barrier.wait(30)
            try:
                outcomes.append(self.create())
            except BaseException as exc:
                outcomes.append(exc)
        threads = [threading.Thread(target=submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait(30)
        for thread in threads:
            thread.join(30)
            self.assertFalse(thread.is_alive())
        self.assertTrue(all(isinstance(value, str) for value in outcomes), outcomes)
        self.assertEqual(len(set(outcomes)), 2)
        child_ids = [child["search_id"] for run_id in outcomes
                     for child in self.engine.get_search(run_id)["children"]]
        self.assertEqual(len(set(child_ids)), 4)
        self.assertEqual(self.counts(), (2, 4, 4))

    def test_rename_invalidates_aggregate_cache_timestamp(self):
        run_id = self.create(["v1"])
        with patch("library_jobs._now", return_value="2099-01-01T00:00:00+00:00"):
            self.engine.update_video("v1", title="新名称")
        run = self.engine.get_search(run_id)
        self.assertEqual(run["updated_at"], "2099-01-01T00:00:00+00:00")
        self.assertEqual(run["children"][0]["title"], "新名称")

    def test_restart_reuses_mapping_and_pause_resume_preserves_completed(self):
        run_id = self.create()
        initial = self.engine.get_search(run_id)
        ids = [child["search_id"] for child in initial["children"]]
        self.searches._update(ids[0], status="completed", results=[{"status": "ok", "frame": {"id": "F1"}}])
        self.assertEqual(self.engine.pause(run_id)["status"], "paused")
        fresh = LibraryEngine(self.videos, SearchEngine(self.videos))
        self.assertEqual([child["search_id"] for child in fresh.get_search(run_id)["children"]], ids)
        resumed = fresh.resume(run_id)
        self.assertEqual([child["status"] for child in resumed["children"]], ["completed", "queued"])
        self.assertEqual(len(resumed["results"]), 1)
        self.assertEqual(self.counts(), (1, 2, 2))

    def test_pause_inflight_child_requests_pause_and_keeps_raw_results(self):
        run_id = self.create()
        child = self.engine.get_search(run_id)["children"][0]
        self.searches._update(child["search_id"], status="running")
        paused = self.engine.pause(run_id)
        self.assertEqual(paused["status"], "pause_requested")
        self.assertEqual([child["status"] for child in paused["children"]], ["pause_requested", "paused"])

    def test_aggregate_keeps_video_identity_intervals_and_partial_errors(self):
        run_id = self.create()
        initial = self.engine.get_search(run_id)
        for index, child in enumerate(initial["children"]):
            result = {"status": "ok", "frame": {"id": "QF00000012", "time_seconds": index + 1},
                      "structured": {"verdict": "visible", "reason": "fixture"}}
            self.searches._update(child["search_id"], status="completed" if index == 0 else "partial",
                error=None if index == 0 else "one frame failed", results=[result], frames=[result["frame"]],
                intervals={"visible_groups": [{"start_seconds": index + 1}]})
        run = self.engine.get_search(run_id)
        self.assertEqual(run["status"], "partial")
        self.assertEqual(len({result["key"] for result in run["results"]}), 2)
        self.assertEqual({result["source_sha256"] for result in run["results"]}, {"1" * 64, "2" * 64})
        self.assertEqual(len(run["intervals"]), 2)
        self.assertEqual(run["errors"][0]["job_id"], "v2")
        self.assertEqual(run["progress"]["frames_verified"], 2)
        self.assertEqual(run["progress"]["videos_completed"], 1)
        self.assertEqual(run["progress"]["videos_partial"], 1)

    def test_missing_child_is_visible_failure_and_never_resubmitted(self):
        run_id = self.create(["v1"])
        child_id = self.engine.get_search(run_id)["children"][0]["search_id"]
        with self.searches._db() as db:
            db.execute("DELETE FROM searches WHERE id=?", (child_id,))
        run = self.engine.get_search(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertIn("缺失", run["errors"][0]["error"])
        self.engine.resume(run_id)
        self.assertEqual(self.counts(), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
