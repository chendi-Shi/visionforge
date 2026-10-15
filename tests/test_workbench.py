"""User-facing library filters and cross-video evidence identity, without models."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from search_reviews import ReviewStore
from workbench import cross_video_rows, evidence_rows, library_rows


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reviews = ReviewStore(self.root)

    def search(self, suffix, verdict="not_visible"):
        image = self.root / f"evidence-{suffix}.png"
        image.write_bytes(f"pixels for video {suffix}".encode())
        frame = {"id": "QF00000012", "frame_index": 12, "time_seconds": 1,
                 "path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
        result = {"status": "ok", "frame": deepcopy(frame), "model": "local:vision", "digest": "weights",
                  "target": "truck", "raw_response": f"original {suffix}",
                  "structured": {"verdict": verdict, "reason": "original model judgment"}}
        return {"id": "search-" + suffix, "job_id": "video-" + suffix, "job_name": "Video " + suffix,
                "status": "completed", "target": "truck", "identity": {"model": "local:vision", "digest": "weights"},
                "video": {"preview_sha256": hashlib.sha256(suffix.encode()).hexdigest()},
                "config": {"step_seconds": 1}, "frames": [frame], "results": [result], "attempts": [deepcopy(result)]}

    @staticmethod
    def child(search):
        return {"search_id": search["id"], "job_id": search["job_id"], "title": search["job_name"], "search": search}

    def test_library_filters_keep_archived_and_unready_out_of_search_scope(self):
        videos = [
            {"id": "a", "name": "a.mp4", "title": "North gate", "tags": ["Traffic"], "status": "ready", "ready_for_search": True},
            {"id": "b", "name": "b.mp4", "title": "South gate", "tags": ["traffic"], "status": "completed", "ready_for_search": True, "archived": True},
            {"id": "c", "name": "c.mp4", "title": "Gate upload", "tags": ["Traffic"], "status": "preparing", "ready_for_search": False},
        ]
        rows, options = library_rows(videos, "TRAFFIC")
        self.assertEqual([row[0] for row in rows], ["a", "c"])
        self.assertEqual(options, [("North gate", "a")])
        rows, options = library_rows(videos, "gate", include_archived=True)
        self.assertEqual([row[0] for row in rows], ["a", "b", "c"])
        self.assertEqual(rows[1][-1], "已归档")
        self.assertEqual(options, [("North gate", "a")])
        self.assertEqual(library_rows(videos, "b.mp4", True)[1], [])

    def test_duplicate_frame_ids_route_to_separate_video_searches_and_reviews(self):
        first, second = self.search("a"), self.search("b")
        self.reviews.save(second, "QF00000012", "visible", "Truck in video B only")
        rows = cross_video_rows([self.child(first), self.child(second)], self.reviews)
        self.assertEqual([row[:3] for row in rows], [
            ["search-a", "video-a", "QF00000012"], ["search-b", "video-b", "QF00000012"]])
        self.assertEqual([row[5:] for row in rows], [["未见", "模型 · 待复核"], ["可见", "用户复核"]])
        visible = cross_video_rows([self.child(first), self.child(second)], self.reviews, only_visible=True)
        self.assertEqual([row[0] for row in visible], ["search-b"])
        self.assertEqual(first["results"][0]["structured"]["verdict"], "not_visible")

    def test_stale_and_deferred_reviews_are_not_presented_as_human_approval(self):
        search = self.search("a", "visible")
        self.reviews.save(search, "QF00000012", "not_visible", "Correction based on pixels")
        active = self.reviews.apply(search)
        self.assertEqual(evidence_rows(active, "待复核"), [])
        self.assertEqual(evidence_rows(active)[0][-1], "用户复核")
        for changed in ({**search, "status": "running"}, deepcopy(search)):
            if changed["status"] == "completed":
                changed["results"][0]["raw_response"] = "new model attempt"
            reviewed = self.reviews.apply(changed)
            self.assertTrue(reviewed["review_rejections"])
            row = evidence_rows(reviewed, "待复核")[0]
            self.assertEqual(row[2:], ["可见", "可见", "模型 · 待复核"])
            cross = cross_video_rows([self.child(changed)], self.reviews, only_visible=True)
            self.assertEqual(cross[0][-1], "模型 · 待复核")

    def test_uncertain_and_pending_filters_keep_work_that_needs_attention(self):
        search = self.search("a", "uncertain")
        pending = {**search["frames"][0], "id": "QF00000024", "time_seconds": 2}
        search["frames"].append(pending)
        self.assertEqual(len(evidence_rows(search, "不确定 / 失败")), 2)
        self.assertEqual(evidence_rows(search, "可见"), [])
        self.assertEqual(cross_video_rows([{"search_id": "missing", "job_id": "video", "title": "Missing", "search": {"results": []}}], self.reviews), [])


if __name__ == "__main__":
    unittest.main()
