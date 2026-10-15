"""Offline evidence packaging: verified assets, escaped text and original responses."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from search_report import export_search
from search_reviews import ReviewStore


class SearchReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.preview = self.root / "preview.mp4"
        self.preview.write_bytes(b"normalized-h264-test-video")
        self.frame = self.root / "candidate.png"
        self.frame.write_bytes(b"fixed-evidence-image")
        frame = {"id": "F1", "time_seconds": 3.25, "path": str(self.frame), "sha256": hashlib.sha256(self.frame.read_bytes()).hexdigest()}
        result = {"status": "ok", "frame": frame, "model": "local:vision", "digest": "weights", "raw_response": '<script>alert("raw")</script>',
                  "structured": {"verdict": "visible", "reason": '=HYPERLINK("bad") <script>alert(1)</script>'}}
        self.search = {"id": "search-1", "job_name": "<img src=x onerror=alert(1)>", "target": "=target", "status": "partial",
            "source_path": str(self.root / "original.webm"), "video": {"preview_path": str(self.preview), "preview_sha256": hashlib.sha256(self.preview.read_bytes()).hexdigest()},
            "translation": {"query": 'red truck <script>alert(2)</script>', "source": "user"},
            "config": {}, "identity": {}, "candidates": [{**frame, "score": .51}], "frames": [frame], "results": [result], "attempts": [result],
            "index": {"preview_path": str(self.preview), "vectors_file": "generations/test/vectors.npz", "frames": [{**frame, "path": "generations/test/frame.png"}, {"id": "unused", "path": "generations/test/unused.png", "sha256": "unpackaged"}]},
            "source_attribution": {"author": "Editor <script>bad</script>", "license": "CC BY 3.0"}, "intervals": {}}

    def test_package_is_portable_and_preserves_raw_records(self):
        before = deepcopy(self.search)
        outputs = export_search(self.search, self.root / "exports")
        self.assertEqual(self.search, before)
        data = json.loads(Path(outputs[1]).read_text(encoding="utf-8"))
        self.assertEqual(data["video"]["preview_path"], "video.mp4")
        self.assertNotIn("source_path", data)
        self.assertNotIn(str(self.root), json.dumps(data))
        self.assertIsNone(data["index"]["vectors_file"])
        self.assertIsNone(data["index"]["frames"][1]["path"])
        self.assertEqual(data["source_attribution"], before["source_attribution"])
        self.assertEqual(data["results"][0]["raw_response"], before["results"][0]["raw_response"])
        self.assertEqual(data["attempts"][0]["structured"], before["results"][0]["structured"])
        self.assertEqual(data["results"][0]["frame"]["path"], data["candidates"][0]["path"])
        with zipfile.ZipFile(outputs[0]) as bundle:
            self.assertEqual(bundle.read("video.mp4"), self.preview.read_bytes())
            self.assertEqual(bundle.read(data["results"][0]["frame"]["path"]), self.frame.read_bytes())
            self.assertIn("report.html", bundle.namelist())

    def test_html_is_escaped_csv_cannot_execute_formulas_and_exports_are_unique(self):
        outputs = export_search(self.search, self.root / "exports")
        report = (Path(outputs[0]).parent / "report.html").read_text(encoding="utf-8")
        self.assertNotIn('<script>alert(', report)
        self.assertNotIn('<img src=x onerror=', report)
        self.assertIn('&lt;script&gt;', report)
        self.assertIn('data-time="3.250000"', report)
        self.assertIn('partial', report)
        self.assertTrue(Path(outputs[2]).read_bytes().startswith(b'\xef\xbb\xbf'))
        with Path(outputs[2]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[1][1], "'=target")
        self.assertTrue(rows[1][7].startswith("'=HYPERLINK"))
        self.assertNotEqual(outputs[0], export_search(self.search, self.root / "exports")[0])

    def test_changed_or_missing_preview_and_evidence_refuse_export(self):
        bad = deepcopy(self.search)
        bad["video"]["preview_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            export_search(bad, self.root / "exports")
        bad["video"] = {}
        with self.assertRaises(ValueError):
            export_search(bad, self.root / "exports")
        self.frame.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            export_search(self.search, self.root / "exports")

    def test_reviewed_export_includes_raw_effective_and_audit_without_claiming_user_approval(self):
        store = ReviewStore(self.root)
        store.save(self.search, "F1", "uncertain", '=CHECK() <script>alert("review")</script>', actor="developer_assistant")
        reviewed = store.apply(self.search)
        outputs = export_search(reviewed, self.root / "exports")
        data = json.loads(Path(outputs[1]).read_text(encoding="utf-8"))
        self.assertEqual(data["results"][0]["structured"]["verdict"], "visible")
        self.assertEqual(data["effective_results"][0]["structured"]["verdict"], "uncertain")
        self.assertEqual(data["review_history"][0]["actor"], "developer_assistant")
        self.assertEqual(data["review_summary"]["active_user"], 0)
        self.assertNotIn(str(self.root), json.dumps(data))
        report = (Path(outputs[0]).parent / "report.html").read_text(encoding="utf-8")
        self.assertIn("复核覆盖后的时间证据", report)
        self.assertIn("模型原判", report)
        self.assertIn("开发助手复核（未经用户审核）", report)
        self.assertNotIn('<script>alert(', report)
        with Path(outputs[2]).open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["verdict"], "visible")
        self.assertEqual(row["effective_verdict"], "uncertain")
        self.assertEqual(row["judgment_source"], "developer_assistant")
        self.assertEqual(row["review_revision"], "1")
        self.assertTrue(row["review_note"].startswith("'=CHECK"))
        with zipfile.ZipFile(outputs[0]) as bundle:
            packed = json.loads(bundle.read("evidence.json"))
            self.assertEqual(packed["review_history"], data["review_history"])


if __name__ == "__main__":
    unittest.main()
