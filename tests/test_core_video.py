import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import core_video


class VideoCoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "frame.png"
        Image.new("RGB", (100, 60), "white").save(self.path)
        self.frames = [{"id": "F1", "time_seconds": 1.0, "path": str(self.path)}, {"id": "F2", "time_seconds": 2.0, "path": str(self.path)}]
        self.data = {"summary": "车辆位置不同", "observations": [{"category": "变化", "description": "两帧中车辆位置发生变化", "evidence_frame_ids": ["F1", "F2"]}], "uncertainties": []}
        self.client = MagicMock()
        self.client.list.return_value = {"models": [{"model": "local:2b", "digest": "digest"}]}
        self.client.show.return_value = {"capabilities": ["vision"], "details": {"family": "other"}}
        self.client.chat.return_value = {"message": {"content": json.dumps(self.data)}, "done_reason": "stop"}

    def test_pair_is_sent_in_order_and_timestamps_remain_application_evidence(self):
        with patch.object(core_video.core, "_create_client", return_value=self.client):
            result = core_video.analyze_segment(self.frames, "车辆变化", "local:2b")
        self.assertEqual(result["status"], "ok")
        self.assertEqual([x["time_seconds"] for x in result["frames"]], [1.0, 2.0])
        self.assertEqual(len(self.client.chat.call_args.kwargs["messages"][1]["images"]), 2)
        self.assertEqual(result["digest"], "digest")

    def test_unknown_evidence_or_single_frame_motion_is_rejected(self):
        for references in (["F1", "FAKE"], ["F1"], ["F1", "F1"]):
            with self.subTest(references=references):
                self.data["observations"][0]["evidence_frame_ids"] = references
                self.client.chat.return_value["message"]["content"] = json.dumps(self.data)
                with patch.object(core_video.core, "_create_client", return_value=self.client):
                    result = core_video.analyze_segment(self.frames, "车辆变化", "local:2b")
                self.assertEqual(result["status"], "invalid_output")
                self.assertTrue(result["raw_response"])

    def test_changed_frame_or_invalid_time_is_rejected_before_model_call(self):
        for value in (float("nan"), -1, True):
            self.frames[0]["time_seconds"] = value
            with self.assertRaises(ValueError):
                core_video.analyze_segment(self.frames, "车辆变化", "local:2b")
        self.frames[0]["time_seconds"] = 1
        self.frames[0]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            core_video.analyze_segment(self.frames, "车辆变化", "local:2b")

    def test_schema_valid_but_truncated_is_not_success(self):
        self.client.chat.return_value["done_reason"] = "length"
        with patch.object(core_video.core, "_create_client", return_value=self.client):
            result = core_video.analyze_segment(self.frames, "车辆变化", "local:2b")
        self.assertEqual(result["status"], "truncated")


if __name__ == "__main__":
    unittest.main()
