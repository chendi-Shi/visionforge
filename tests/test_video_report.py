"""Offline portable-export and text-injection checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_report import clock, effective_summary, export_job


class VideoReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        source, preview, frame = self.base / "source.webm", self.base / "normalized-preview.mp4", self.base / "frame.png"
        source.write_bytes(b"original-webm-video-bytes")
        preview.write_bytes(b"normalized-h264-preview-bytes")
        frame.write_bytes(b"local-frame-bytes")
        frame_info = {"id": "S0001-F01", "time_seconds": 1.25, "path": str(frame), "sha256": hashlib.sha256(frame.read_bytes()).hexdigest()}
        self.job = {
            "id": "job-1", "name": "Demo", "source_path": str(source), "status": "partial", "error": "second segment failed", "message": "Review needed",
            "video": {"preview_path": str(preview), "preview_sha256": hashlib.sha256(preview.read_bytes()).hexdigest(), "timebase": "normalized preview"},
            "config": {"focus": "visible events", "model": "local:vision"}, "identity": {"model": "local:vision", "digest": "abc"}, "selection": {"selected": 1, "candidate": 2},
            "segments": [{"id": "S0001", "status": "ok", "start_seconds": 0, "end_seconds": 5, "frames": [frame_info],
                          "result": {"status": "ok", "summary": "Original model words", "frames": [dict(frame_info)], "raw_response": '{"summary":"Original model words"}',
                                     "observations": [{"category": "状态", "description": "A visible object", "evidence_frame_ids": ["S0001-F01"]}], "uncertainties": ["Unsampled interval"]},
                          "review": {"decision": "corrected", "text": "Human correction"}, "attempt_history": []}],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_portable_assets_and_review_preserve_original_result(self):
        attribution = {"author": "<b>Jane</b>", "license": "CC BY 3.0", "source": "https://example.invalid/source-video"}
        self.job["source_attribution"] = attribution
        files = export_job(self.job, self.base / "exports")
        self.assertEqual(len(files), 3)
        archive, json_path, csv_path = map(Path, files)
        portable = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertNotIn("source_path", portable)
        self.assertEqual(portable["video"]["preview_path"], "video.mp4")
        self.assertNotIn(str(self.base), json_path.read_text(encoding="utf-8"))
        segment = portable["segments"][0]
        relative_frame = segment["frames"][0]["path"]
        self.assertTrue(relative_frame.startswith("frames/"))
        self.assertEqual(segment["result"]["frames"][0]["path"], relative_frame)
        self.assertEqual(segment["result"]["summary"], "Original model words")
        self.assertEqual(segment["review"]["text"], "Human correction")
        self.assertEqual(portable["status"], "partial")
        self.assertEqual(portable["source_attribution"], attribution)
        report = (archive.parent / "report.html").read_text(encoding="utf-8")
        self.assertIn("CC BY 3.0", report)
        self.assertIn("&lt;b&gt;Jane&lt;/b&gt;", report)
        self.assertNotIn('<a href="https://example.invalid', report)
        self.assertEqual(self.job["segments"][0]["frames"][0]["path"], str(self.base / "frame.png"))
        with zipfile.ZipFile(archive) as bundle:
            self.assertTrue({"report.html", "evidence.json", "timeline.csv", "video.mp4", relative_frame}.issubset(bundle.namelist()))
            self.assertEqual(bundle.read("video.mp4"), b"normalized-h264-preview-bytes")
            self.assertNotEqual(bundle.read("video.mp4"), (self.base / "source.webm").read_bytes())
            self.assertEqual(bundle.read(relative_frame), b"local-frame-bytes")
        self.assertTrue(csv_path.read_bytes().startswith(b"\xef\xbb\xbf"))
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["effective_summary"], "Human correction")
        self.assertEqual(row["model_summary"], "Original model words")
        another = export_job(self.job, self.base / "exports")
        self.assertNotEqual(Path(another[0]).parent, archive.parent)
        self.assertTrue(archive.exists())

    def test_escape_untrusted_html_and_csv_formulas(self):
        attack = '<script>alert("model")</script>'
        self.job["name"] = attack
        self.job["config"]["focus"] = '<img src=x onerror="alert(1)">'
        segment = self.job["segments"][0]
        segment["result"]["summary"] = attack
        segment["result"]["observations"][0]["description"] = attack
        segment["result"]["uncertainties"] = [attack]
        segment["result"]["raw_response"] = attack
        segment["review"]["text"] = '=HYPERLINK("https://example.invalid","click")'
        files = export_job(self.job, self.base / "exports")
        report = (Path(files[0]).parent / "report.html").read_text(encoding="utf-8")
        self.assertNotIn(attack, report)
        self.assertNotIn('<img src=x onerror=', report)
        self.assertIn("&lt;script&gt;", report)
        self.assertIn('data-time="1.250000"', report)
        self.assertIn('src="video.mp4"', report)
        self.assertIn("绝非事件精确起止时间", report)
        self.assertIn("不分析声音", report)
        with Path(files[2]).open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertTrue(row["effective_summary"].startswith("'=HYPERLINK"))
        self.assertTrue(row["review_text"].startswith("'=HYPERLINK"))

    def test_clock_fallback_and_changed_evidence_are_explicit(self):
        self.assertEqual(clock(3661.125), "01:01:01.125")
        self.assertEqual(clock(59.9999), "00:01:00.000")
        self.assertEqual(clock(float("nan")), "—")
        self.assertEqual(effective_summary({"result": None}), "尚无模型结果")
        self.assertEqual(effective_summary({"error": "Model failed"}), "Model failed")
        (self.base / "frame.png").write_bytes(b"changed-evidence")
        files = export_job(self.job, self.base / "exports")
        data = json.loads(Path(files[1]).read_text(encoding="utf-8"))
        self.assertFalse(data["export"]["frames"][0]["matches_recorded_sha256"])
        self.assertTrue(data["export"]["warnings"])
        self.assertEqual(data["segments"][0]["frames"][0]["sha256"], self.job["segments"][0]["frames"][0]["sha256"])

    def test_missing_or_changed_normalized_preview_is_rejected(self):
        metadata = self.job["video"]
        self.job["video"] = {}
        with self.assertRaisesRegex(ValueError, "Normalized MP4 preview is missing"):
            export_job(self.job, self.base / "exports")
        self.job["video"] = {"preview_path": metadata["preview_path"]}
        with self.assertRaisesRegex(ValueError, "SHA256 is missing"):
            export_job(self.job, self.base / "exports")
        self.job["video"] = metadata
        preview = Path(metadata["preview_path"])
        preview.write_bytes(b"modified-preview")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            export_job(self.job, self.base / "exports")
        preview.unlink()
        with self.assertRaisesRegex(ValueError, "preview file is missing"):
            export_job(self.job, self.base / "exports")


if __name__ == "__main__":
    unittest.main()
