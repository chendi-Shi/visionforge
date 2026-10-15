"""Review overlays preserve raw evidence and recompute only supported bounds."""
from copy import deepcopy
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from core_grounding import derive_intervals
from search_reviews import ReviewConflict, ReviewStore, StaleReview, frame_id


class SearchReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ReviewStore(self.root)
        frames, results = [], []
        for stamp in range(30, 35):
            path = self.root / f"frame-{stamp}.png"
            path.write_bytes(f"immutable pixels {stamp}".encode())
            frame = {"id": f"F{stamp}", "frame_index": stamp * 12, "time_seconds": stamp,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path)}
            frames.append(frame)
            results.append({"frame": deepcopy(frame), "status": "ok", "target": "branded truck",
                            "model": "local:vision", "digest": "model-weights", "raw_response": f"original {stamp}",
                            "structured": {"verdict": "visible" if stamp == 31 else "not_visible", "reason": f"model reason {stamp}"}})
        self.search = {"id": "search-one", "status": "completed", "target": "When is the branded truck visible?",
                       "translation": {"visual_target": "branded truck"}, "identity": {"model": "local:vision", "digest": "model-weights"},
                       "video": {"preview_sha256": "v" * 64}, "config": {"step_seconds": 1},
                       "frames": frames, "results": results, "attempts": deepcopy(results),
                       "intervals": derive_intervals(results, max_gap_seconds=1 + 1 / 12)}

    def test_restart_recomputes_bound_without_rewriting_originals(self):
        before = deepcopy(self.search)
        saved = self.store.save(self.search, "F32", "visible", "Logo and vehicle remain visible behind the pole.")
        self.assertEqual(saved["revision"], 1)
        reviewed = ReviewStore(self.root).apply(self.search)
        self.assertEqual(self.search, before)
        for key in ("results", "attempts", "intervals"):
            self.assertEqual(reviewed[key], before[key])
        self.assertEqual(reviewed["intervals"]["disappearance_brackets"][0]["after_seconds"], 31)
        self.assertEqual(reviewed["reviewed_intervals"]["disappearance_brackets"][0]["after_seconds"], 32)
        self.assertEqual(reviewed["reviewed_intervals"]["disappearance_brackets"][0]["by_seconds"], 33)
        self.assertEqual(reviewed["effective_results"][2]["structured"]["verdict"], "visible")
        self.assertEqual(reviewed["effective_results"][2]["judgment_source"], "user")
        self.assertEqual(reviewed["review_summary"], {"revisions": 1, "active_user": 1, "active_developer_assistant": 0, "rejected_or_deferred": 0})

    def test_optimistic_conflict_reset_and_assistant_attribution(self):
        self.store.save(self.search, "F32", "visible", "Assistant inspection only", actor="developer_assistant")
        with self.assertRaises(ReviewConflict):
            ReviewStore(self.root).save(self.search, "F32", "uncertain", "stale screen", expected_revision=0)
        assistant = self.store.apply(self.search)
        self.assertEqual(assistant["review_summary"]["active_user"], 0)
        self.assertEqual(assistant["review_summary"]["active_developer_assistant"], 1)
        reset = self.store.save(self.search, "F32", "reset", "", expected_revision=1)
        self.assertEqual(reset["revision"], 2)
        reviewed = self.store.apply(self.search)
        self.assertEqual(reviewed["reviewed_intervals"], self.search["intervals"])
        self.assertEqual(reviewed["effective_results"][2]["judgment_source"], "model")
        self.assertEqual([e["verdict"] for e in reviewed["review_history"]], ["visible", "reset"])
        self.assertEqual(reviewed["review_current"]["F32"]["revision"], 2)

    def test_database_history_is_append_only(self):
        self.store.save(self.search, "F32", "visible", "visible logo")
        with closing(sqlite3.connect(self.store.database)) as db:
            for statement in ("UPDATE reviews SET revision=7", "DELETE FROM reviews"):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                    db.execute(statement)
        self.assertEqual(len(self.store.apply(self.search)["review_history"]), 1)

    def test_changed_target_model_result_or_frame_reject_stale_review(self):
        self.store.save(self.search, "F32", "visible", "visible logo")
        mutations = [
            lambda s: s.update(target="a different target"),
            lambda s: s["translation"].update(visual_target="different visual target"),
            lambda s: s["identity"].update(digest="new-weights"),
            lambda s: s["video"].update(preview_sha256="different-video"),
            lambda s: s["results"][2].update(raw_response="rewritten original"),
            lambda s: s["results"][2]["frame"].update(sha256=None),
            lambda s: s["frames"][2].update(frame_index=999),
            lambda s: s["results"].pop(2),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.search)
                mutation(changed)
                reviewed = self.store.apply(changed)
                self.assertEqual(reviewed["review_current"]["F32"]["state"], "stale")
                self.assertEqual(len(reviewed["review_rejections"]), 1)
                self.assertEqual(reviewed["review_summary"]["active_user"], 0)
        Path(self.search["frames"][2]["path"]).write_bytes(b"changed pixels")
        reviewed = self.store.apply(self.search)
        self.assertIn("SHA256", reviewed["review_rejections"][0]["reason"])
        self.assertEqual(reviewed["effective_results"][2]["status"], "failed")
        self.assertEqual(reviewed["reviewed_intervals"]["disappearance_brackets"], [])
        with self.assertRaises(StaleReview):
            self.store.save(self.search, "F32", "uncertain", "cannot verify changed pixels", expected_revision=1)

    def test_missing_files_bad_decisions_and_running_searches_are_rejected(self):
        for status in ("queued", "running", "pause_requested"):
            with self.subTest(status=status):
                changed = {**self.search, "status": status}
                with self.assertRaisesRegex(ValueError, "暂停"):
                    self.store.save(changed, "F32", "visible", "logo")
        for options in ({"note": " "}, {"actor": "assistant"}, {"verdict": "yes"}, {"expected_revision": True}):
            arguments = dict(frame_id="F32", verdict="visible", note="logo") | options
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.store.save(self.search, **arguments)
        Path(self.search["frames"][2]["path"]).unlink()
        with self.assertRaises(StaleReview):
            self.store.save(self.search, "F32", "visible", "logo")
        self.assertEqual(self.store.apply({**self.search, "results": []})["review_history"], [])

    def test_uncertain_review_and_large_gaps_do_not_form_false_bounds(self):
        self.store.save(self.search, "F32", "uncertain", "Occlusion prevents confident judgment")
        reviewed = self.store.apply(self.search)
        self.assertEqual(reviewed["reviewed_intervals"]["disappearance_brackets"], [])
        self.assertEqual(reviewed["reviewed_intervals"]["uncertain_or_failed"][0]["frame_id"], "F32")
        spaced = deepcopy(self.search)
        spaced["id"] = "spaced-search"
        for frame in (spaced["frames"][2], spaced["results"][2]["frame"]):
            frame.update(time_seconds=40, frame_index=480)
        self.store.save(spaced, "F32", "visible", "Logo visible at isolated later sample")
        intervals = self.store.apply(spaced)["reviewed_intervals"]
        self.assertEqual(intervals["appearance_brackets"], [{"after_seconds": 30, "by_seconds": 31, "negative_frame_id": "F30", "positive_frame_id": "F31"}])
        self.assertTrue(intervals["gaps"])
        self.assertEqual(intervals["visible_groups"][-1]["frame_ids"], ["F32"])

    def test_failed_model_call_can_be_reviewed_with_explicit_source(self):
        failed = self.search["results"][2]
        failed.update(status="failed", digest=None, structured=None, error="generation timeout")
        self.search["status"] = "partial"
        self.store.save(self.search, "F32", "visible", "Manually inspected intact original frame")
        reviewed = self.store.apply(self.search)
        self.assertEqual(reviewed["results"][2]["status"], "failed")
        self.assertEqual(reviewed["effective_results"][2]["status"], "ok")
        self.assertEqual(reviewed["effective_results"][2]["judgment_source"], "user")
        self.assertEqual(reviewed["effective_results"][2]["digest"], "model-weights")

    def test_resume_defers_review_and_changed_retry_requires_explicit_reset(self):
        self.search["status"] = "paused"
        self.store.save(self.search, "F32", "visible", "Logo visible")
        running = self.store.apply({**self.search, "status": "running"})
        self.assertEqual(running["review_current"]["F32"]["state"], "deferred")
        self.assertEqual(running["effective_results"][2]["structured"]["verdict"], "not_visible")
        retried = deepcopy(self.search)
        retried["results"][2]["raw_response"] = "new inference after failed retry"
        stale = self.store.apply(retried)
        self.assertEqual(stale["review_current"]["F32"]["state"], "stale")
        with self.assertRaises(StaleReview):
            self.store.save(retried, "F32", "visible", "review new result", expected_revision=1)
        self.store.save(retried, "F32", "reset", "Retire old review after model retry", expected_revision=1)
        restored = self.store.apply(retried)
        self.assertEqual(restored["review_rejections"], [])
        self.assertEqual(restored["review_current"]["F32"]["state"], "reset")
        self.store.save(retried, "F32", "visible", "Rechecked latest result against original pixels", expected_revision=2)
        fresh = self.store.apply(retried)
        self.assertEqual(len(fresh["review_history"]), 3)
        self.assertEqual(fresh["review_current"]["F32"]["state"], "active")

    def test_reviewed_failed_frame_with_changed_digest_preserves_identity_and_disables_bounds(self):
        failed = self.search["results"][2]
        failed.update(status="failed", digest="unexpected-new-weights", structured=None,
                      error="Model digest changed during inference")
        self.search["status"] = "partial"
        before = deepcopy(self.search)
        self.store.save(self.search, "F32", "visible", "Original pixels show the truck despite inference failure")
        reviewed = self.store.apply(self.search)
        self.assertEqual(self.search, before)
        self.assertEqual(reviewed["results"][2], before["results"][2])
        self.assertEqual(reviewed["effective_results"][2]["digest"], "unexpected-new-weights")
        self.assertEqual(reviewed["effective_results"][2]["structured"]["verdict"], "visible")
        self.assertEqual(reviewed["effective_results"][2]["judgment_source"], "user")
        self.assertIn("模型身份", reviewed["review_interval_warning"])
        self.assertEqual(reviewed["reviewed_intervals"]["derivation_error"], reviewed["review_interval_warning"])
        for field in ("visible_groups", "appearance_brackets", "disappearance_brackets"):
            self.assertEqual(reviewed["reviewed_intervals"][field], [])

    def test_fallback_frame_identity_is_stable_and_pixel_bound(self):
        frame = deepcopy(self.search["frames"][0])
        self.assertEqual(frame_id(frame), "F30")
        frame.pop("id")
        identifier = frame_id(frame)
        self.assertEqual(frame_id(deepcopy(frame)), identifier)
        frame["sha256"] = "0" * 64
        self.assertNotEqual(frame_id(frame), identifier)


if __name__ == "__main__":
    unittest.main()
