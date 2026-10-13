"""Offline single-frame grounding and conservative temporal derivation checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import core_grounding as grounding


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "frame.png"
        Image.new("RGB", (720, 480), "white").save(self.path)
        self.frame = {"id": "F1", "time_seconds": 3.25, "path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}
        self.client = MagicMock()
        self.client.list.return_value = {"models": [{"model": "local:vision", "digest": "weights-a"}]}
        self.client.show.return_value = {"capabilities": ["vision"], "details": {"family": "other"}}
        self.client.chat.return_value = {"message": {"content": json.dumps({"verdict": "not_visible", "reason": "当前图片未见目标。"})}, "done_reason": "stop"}

    def call(self, frame=None, **kwargs):
        with patch.object(grounding.core, "_create_client", return_value=self.client):
            return grounding.verify_frame(self.frame if frame is None else frame, "戴红头盔的骑车人", model="local:vision", **kwargs)

    def test_valid_single_frame_keeps_true_time_and_omits_unsupported_think(self):
        record = self.call()
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["frame"]["time_seconds"], 3.25)
        self.assertEqual(record["frame"]["sha256"], self.frame["sha256"])
        self.assertEqual(record["digest"], "weights-a")
        self.assertNotIn("think", self.client.chat.call_args.kwargs)
        self.assertEqual(len(self.client.chat.call_args.kwargs["messages"][1]["images"]), 1)
        self.assertNotIn(str(self.path), record["question"])
        self.assertEqual(set(record["structured"]), {"verdict", "reason"})

    def test_missing_or_changed_evidence_never_reaches_model(self):
        for override in ({"sha256": "0" * 64}, {"sha256": None}, {"time_seconds": float("nan")}, {"path": "missing-file.png"}, {"id": None}):
            with self.subTest(override=override):
                self.assertEqual(self.call({**self.frame, **override})["status"], "failed")
        self.client.chat.assert_not_called()
        self.assertEqual(self.call({})["status"], "failed")

    def test_truncated_invalid_extra_time_and_long_reason_keep_raw_but_fail(self):
        for payload, finish in (({"verdict": "visible", "reason": "可以看到目标。"}, "length"),
                                ({"verdict": "visible", "reason": "可以看到目标。", "time_seconds": 99}, "stop"),
                                ({"verdict": "visible", "reason": "长" * 181}, "stop"),
                                ({"verdict": "uncertain", "reason": " "}, "stop")):
            with self.subTest(finish=finish, payload=payload):
                raw = json.dumps(payload)
                self.client.chat.return_value = {"message": {"content": raw}, "done_reason": finish}
                record = self.call()
                self.assertEqual(record["status"], "failed")
                self.assertIsNone(record["structured"])
                self.assertEqual(record["raw_response"], raw)

    def test_qwen_thinking_policy_and_uncertain_are_explicit(self):
        self.client.show.return_value = {"capabilities": ["vision", "thinking"], "details": {"family": "qwen3vl"}}
        self.client.chat.return_value["message"]["content"] = json.dumps({"verdict": "uncertain", "reason": "遮挡使目标无法辨认。"})
        record = self.call()
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["structured"]["verdict"], "uncertain")
        self.assertIs(self.client.chat.call_args.kwargs["think"], False)
        self.assertEqual(self.client.chat.call_args.kwargs["messages"][-1]["content"], "<think>\n\n</think>\n\n")


def sample(stamp, verdict, status="ok"):
    return {"status": status, "frame": {"id": f"F{stamp}", "time_seconds": stamp},
            "target": "truck", "model": "local:vision", "digest": "weights-a",
            "structured": {"verdict": verdict, "reason": "test"} if status == "ok" else None}


class IntervalTests(unittest.TestCase):
    def test_only_adjacent_opposite_samples_produce_brackets(self):
        result = grounding.derive_intervals([sample(3, "visible"), sample(1, "not_visible"), sample(2, "visible"), sample(4, "not_visible")], 1)
        self.assertEqual(result["visible_groups"], [{"evidence_start_seconds": 2, "evidence_end_seconds": 3, "frame_ids": ["F2", "F3"]}])
        self.assertEqual(result["appearance_brackets"][0]["after_seconds"], 1)
        self.assertEqual(result["appearance_brackets"][0]["by_seconds"], 2)
        self.assertEqual(result["disappearance_brackets"][0]["after_seconds"], 3)
        self.assertEqual(result["disappearance_brackets"][0]["by_seconds"], 4)

    def test_uncertain_failed_and_gap_break_continuity(self):
        for middle in (sample(2, "uncertain"), sample(2, None, "failed")):
            result = grounding.derive_intervals([sample(1, "not_visible"), middle, sample(3, "visible")], 5)
            self.assertFalse(result["appearance_brackets"])
            self.assertEqual(len(result["uncertain_or_failed"]), 1)
        result = grounding.derive_intervals([sample(1, "visible"), sample(3, "visible"), sample(8, "not_visible")], 1)
        self.assertEqual(len(result["visible_groups"]), 2)
        self.assertFalse(result["disappearance_brackets"])
        self.assertEqual(len(result["gaps"]), 2)

    def test_missing_evidence_duplicate_time_and_mixed_identity_are_safe(self):
        result = grounding.derive_intervals([sample(1, "not_visible"), {"status": "failed", "frame": {}}, sample(2, "visible")], 5)
        self.assertFalse(result["appearance_brackets"])
        result = grounding.derive_intervals([sample(1, "not_visible"), sample(1, "visible"), sample(2, "visible")], 5)
        self.assertFalse(result["appearance_brackets"])
        with self.assertRaises(ValueError):
            grounding.derive_intervals([sample(1, "visible"), {**sample(2, "visible"), "digest": "other"}], 5)
        self.assertEqual(grounding.derive_intervals([], 1)["visible_groups"], [])
        with self.assertRaises(ValueError):
            grounding.derive_intervals([], float("nan"))


if __name__ == "__main__":
    unittest.main()
