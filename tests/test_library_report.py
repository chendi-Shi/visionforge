"""Portable multi-video dossiers preserve separate clocks, sources and review audit."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from library_report import export_library_search
from search_reviews import ReviewStore


class LibraryReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reviews = ReviewStore(self.root)
        self.children = [self.child("a"), self.child("b")]
        self.run = {"id": "case-one", "target": "<script>alert('target')</script>", "retrieval_query": "truck",
                    "model": "local:vision", "config": {"max_total_frames": 6}, "status": "completed",
                    "progress": {"videos_total": 2}, "children": self.children}

    def child(self, suffix):
        preview = self.root / f"video-{suffix}.mp4"
        preview.write_bytes(f"normalized video {suffix}".encode())
        image = self.root / f"frame-{suffix}.png"
        image.write_bytes(f"different evidence pixels {suffix}".encode())
        frame = {"id": "QF00000012", "time_seconds": 1, "frame_index": 12,
                 "path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
        result = {"status": "ok", "frame": deepcopy(frame), "target": "truck", "model": "local:vision", "digest": "weights",
                  "raw_response": f"untouched raw response {suffix}",
                  "structured": {"verdict": "not_visible", "reason": "Original judgment"}}
        video = {"preview_path": str(preview), "preview_sha256": hashlib.sha256(preview.read_bytes()).hexdigest()}
        search = {"id": "search-" + suffix, "job_id": "video-" + suffix, "job_name": "=video-" + suffix,
                  "status": "completed", "target": "truck", "video": video,
                  "identity": {"model": "local:vision", "digest": "weights"}, "translation": {"query": "truck"},
                  "config": {"step_seconds": 1}, "candidates": [], "frames": [frame], "results": [result],
                  "attempts": [deepcopy(result)], "intervals": {}}
        return {"job_id": search["job_id"], "search_id": search["id"], "title": "=video-" + suffix,
                "source_sha256": video["preview_sha256"], "status": "completed", "message": "done", "error": None, "search": search}

    def test_archive_is_portable_and_duplicate_frame_ids_keep_their_video(self):
        self.reviews.save(self.children[1]["search"], "QF00000012", "visible", "Visible in the second video", actor="developer_assistant")
        before = deepcopy(self.run)
        outputs = export_library_search(self.run, self.reviews, self.root / "exports")
        self.assertEqual(self.run, before)
        manifest = json.loads(Path(outputs[2]).read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["videos"]), 2)
        self.assertEqual(len({item["report"] for item in manifest["videos"]}), 2)
        with zipfile.ZipFile(outputs[0]) as archive:
            names = archive.namelist()
            self.assertIn("index.html", names)
            self.assertFalse(any(name.endswith(".zip") for name in names))
            self.assertFalse(any(Path(name).is_absolute() or ".." in Path(name).parts for name in names))
            for index, item in enumerate(manifest["videos"]):
                self.assertIn(item["report"], names)
                evidence = json.loads(archive.read(item["evidence"]))
                self.assertEqual(evidence["id"], self.children[index]["search_id"])
                self.assertEqual(evidence["results"][0]["raw_response"], before["children"][index]["search"]["results"][0]["raw_response"])
                prefix = Path(item["evidence"]).parent
                self.assertEqual(archive.read((prefix / evidence["video"]["preview_path"]).as_posix()), Path(self.children[index]["search"]["video"]["preview_path"]).read_bytes())
                self.assertEqual(archive.read((prefix / evidence["frames"][0]["path"]).as_posix()), Path(self.children[index]["search"]["frames"][0]["path"]).read_bytes())
                self.assertNotIn(str(self.root), json.dumps(evidence))
                if index == 1:
                    self.assertEqual(evidence["effective_results"][0]["judgment_source"], "developer_assistant")
                    self.assertEqual(evidence["review_summary"]["active_user"], 0)
            self.assertNotIn("<script>alert(", archive.read("index.html").decode())
        with Path(outputs[1]).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["search_id"] for row in rows], ["search-a", "search-b"])
        self.assertEqual([row["frame_id"] for row in rows], ["QF00000012"] * 2)
        self.assertEqual([row["video"] for row in rows], ["'=video-a", "'=video-b"])
        self.assertNotEqual(rows[0]["preview_sha256"], rows[1]["preview_sha256"])
        self.assertEqual(rows[1]["model_verdict"], "not_visible")
        self.assertEqual(rows[1]["effective_verdict"], "visible")
        self.assertEqual(rows[1]["judgment_source"], "developer_assistant")

    def test_missing_child_remains_explicit_failure_with_successful_reports(self):
        self.run["status"] = "partial"
        self.run["children"].append({"job_id": "gone-video", "search_id": "gone-search", "title": "missing",
            "status": "failed", "message": "missing child record", "error": "Missing child",
            "search": {"id": "gone-search", "status": "failed", "results": [], "frames": []}})
        outputs = export_library_search(self.run, self.reviews, self.root / "exports")
        manifest = json.loads(Path(outputs[2]).read_text(encoding="utf-8"))
        missing = manifest["videos"][2]
        self.assertEqual(missing["status"], "failed")
        self.assertEqual(missing["error"], "Missing child")
        self.assertIsNone(missing["report"])
        self.assertIn("未生成", missing["export_note"])
        self.assertTrue(manifest["videos"][0]["report"])

    def test_corrupted_video_or_frame_aborts_bundle_instead_of_dropping_child(self):
        for asset in ("preview", "frame"):
            with self.subTest(asset=asset):
                run = deepcopy(self.run)
                child = run["children"][1]["search"]
                if asset == "preview":
                    child["video"]["preview_sha256"] = "0" * 64
                else:
                    child["frames"][0]["sha256"] = "0" * 64
                output = self.root / ("bad-" + asset)
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    export_library_search(run, self.reviews, output)
                self.assertEqual(list(output.rglob("video-library-evidence.zip")), [])

    def test_stale_review_is_exported_as_audit_without_false_reviewed_source(self):
        search = self.children[1]["search"]
        self.reviews.save(search, "QF00000012", "visible", "Earlier correction")
        search["results"][0]["raw_response"] = "new model attempt"
        outputs = export_library_search(self.run, self.reviews, self.root / "exports")
        with Path(outputs[1]).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[1]["effective_verdict"], "not_visible")
        self.assertEqual(rows[1]["judgment_source"], "model")
        with zipfile.ZipFile(outputs[0]) as archive:
            manifest = json.loads(archive.read("case.json"))
            evidence = json.loads(archive.read(manifest["videos"][1]["evidence"]))
            self.assertEqual(evidence["review_current"]["QF00000012"]["state"], "stale")
            self.assertTrue(evidence["review_history"])
            self.assertTrue(evidence["review_rejections"])


if __name__ == "__main__":
    unittest.main()
