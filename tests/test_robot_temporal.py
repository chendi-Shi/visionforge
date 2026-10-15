import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from robotbench.temporal import event_scores, parse_verdict, select_frames
from robotbench import temporal


class TemporalSamplerTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(600, dtype=float) / 25
        self.e = np.zeros((600, 16))
        self.e[:, 3] = self.e[:, 11] = 1

    def test_equal_budget_static_and_irregular(self):
        for t in (self.t, np.cumsum(np.linspace(0.02, 0.08, len(self.t)))):
            for method in ("uniform", "state_event"):
                selection = select_frames(t, self.e, method)
                self.assertEqual(len(selection["indices"]), 12)
                self.assertEqual(len(set(selection["indices"])), 12)
                self.assertEqual(selection["indices"], sorted(selection["indices"]))
                self.assertEqual(selection["indices"][0], 0)
                self.assertEqual(selection["indices"][-1], 599)
                np.testing.assert_array_equal(selection["timestamps"], t[selection["indices"]])
                self.assertEqual(selection, select_frames(t, self.e, method))

    def test_local_gripper_change_enters_candidate_set(self):
        self.e[150:, 7] = 25
        self.e[390:, 15] = 60
        selection = select_frames(self.t, self.e, "state_event")
        self.assertLessEqual(min(abs(i - 150) for i in selection["event_indices"]), 1)
        self.assertLessEqual(min(abs(i - 390) for i in selection["event_indices"]), 1)
        # Gripper calibration units must not alter selection after scale normalization.
        self.e[:, 7] *= 100
        np.testing.assert_allclose(event_scores(self.t, self.e), selection["scores"], atol=1e-10)

    def test_separate_robot_base_frames_are_translation_invariant(self):
        self.e[:, 0] = np.sin(self.t)
        first = event_scores(self.t, self.e)
        self.e[:, :3] += [100, -50, 5]
        self.e[:, 8:11] += [-10, 60, -4]
        np.testing.assert_allclose(first, event_scores(self.t, self.e), atol=1e-8)

    def test_invalid_time_and_state_rejected(self):
        self.t[5] = self.t[4]
        with self.assertRaises(ValueError):
            select_frames(self.t, self.e, "uniform")
        with self.assertRaises(ValueError):
            select_frames(np.arange(3), np.zeros((3, 16)), "state_event")


class TemporalPredictionTests(unittest.TestCase):
    def test_timestamp_comes_from_selected_frame_not_free_generation(self):
        value = parse_verdict(json.dumps({"decision": "failure_at_frame_01", "reason": "Wrong object grasped."}),
                              [0.0, 1.375, 8.0], "stop")
        self.assertEqual(value["onset_seconds"], 1.375)

    def test_semantic_frame_errors_and_truncation_rejected(self):
        for decision in ("failure", "failure_at_frame_-1", "failure_at_frame_12", "failure_at_frame_1", True, None):
            with self.assertRaises(ValueError):
                parse_verdict(json.dumps({"decision": decision, "reason": "Evidence"}), [0, 1], "stop")
        with self.assertRaises(ValueError):
            parse_verdict('{"decision":"failure_at_frame_01","reason":"x"}', [0, 1], "length")
        for decision in ("success", "abstain"):
            value = parse_verdict(json.dumps({"decision": decision, "reason": "Visible evidence."}), [0, 1], "stop")
            self.assertEqual(value["frame_id"], -1)
            self.assertIsNone(value["onset_seconds"])


class TemporalResumeTests(unittest.TestCase):
    """Exercise production freeze/run with fake local metadata and no inference."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.out = self.root / "data", self.root / "run"
        self.data.mkdir()
        for name, value in (("manifest.json", {"complete": True}), ("plan.json", {"fixed": "selection"})):
            temporal.write_json(self.data / name, value)
        (self.data / "labels.jsonl").write_text('{"label":"failure"}\n', encoding="utf-8")
        self.video = self.data / "head.mp4"
        self.video.write_bytes(b"fixed-video-placeholder")
        self.row = {"sample_id": "opaque-episode", "task_id": "task-a", "robot": "robot",
                    "source_sha256": "a" * 64, "split": "development", "task_description": "Put the cube in the cup",
                    "substages": ["Grasp cube", "Place cube in cup"], "timestamp": (np.arange(120) / 25).tolist(),
                    "state_eef": np.zeros((120, 16)).tolist(), "head_view": "head", "videos": {"head": str(self.video)}}
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.client.chat.return_value = {"message": {"content": '{"decision":"success","reason":"Cube inside cup."}'}, "done_reason": "stop"}
        patches = [patch.object(temporal, "load_episode_inputs", return_value=[self.row]),
                   patch.object(temporal, "server_version", return_value={"version": "fixed-version"}),
                   patch.object(temporal.core, "_create_client", return_value=self.client),
                   patch.object(temporal.core, "_inspect_model", return_value=("local:model", "b" * 64, {"capabilities": ["vision"]})),
                   patch.object(temporal, "make_panel", return_value=b"fixed-panel-bytes")]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def freeze_and_run(self):
        temporal.freeze(self.data, self.out, "local:model", "development")
        temporal.run(self.data, self.out, "development")

    def test_cached_records_require_protocol_fields_and_raw_consistency(self):
        self.freeze_and_run()
        self.assertEqual(self.client.chat.call_count, 2)
        temporal.run(self.data, self.out, "development")
        self.assertEqual(self.client.chat.call_count, 2)
        path = self.out / "predictions" / "opaque-episode--uniform.json"
        baseline = json.loads(path.read_text(encoding="utf-8"))
        corruptions = [("model_digest", "changed"), ("question", "changed"),
                       ("options", {"num_predict": 1}), ("system", "changed"),
                       ("prediction", "failure"), ("status", "unknown")]
        for key, value in corruptions:
            altered = {**deepcopy(baseline), key: value}
            temporal.write_json(path, altered)
            with self.subTest(field=key), self.assertRaises(ValueError):
                temporal.run(self.data, self.out, "development")
            self.assertEqual(self.client.chat.call_count, 2)
        temporal.write_json(path, baseline)

    def test_saved_failed_predictions_cannot_claim_success(self):
        self.freeze_and_run()
        path = self.out / "predictions" / "opaque-episode--uniform.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row.update(status="failed", error="Transport failure", prediction="success")
        temporal.write_json(path, row)
        with self.assertRaisesRegex(ValueError, "failure/abstention"):
            temporal.run(self.data, self.out, "development")
        self.assertEqual(self.client.chat.call_count, 2)

    def test_modified_panel_and_video_fail_before_new_inference(self):
        self.freeze_and_run()
        path = self.out / "predictions" / "opaque-episode--uniform.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        panel = self.out / row["panel_file"]
        panel.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "panel checksum"):
            temporal.run(self.data, self.out, "development")
        panel.write_bytes(b"fixed-panel-bytes")
        self.video.write_bytes(b"different source pixels")
        with self.assertRaisesRegex(ValueError, "Input set changed"):
            temporal.run(self.data, self.out, "development")
        self.assertEqual(self.client.chat.call_count, 2)

    def test_test_freeze_rejects_incomplete_data_and_pins_labels(self):
        self.row["split"] = "test"
        temporal.write_json(self.data / "manifest.json", {"complete": False})
        with self.assertRaisesRegex(ValueError, "entire frozen selection"):
            temporal.freeze(self.data, self.out, "local:model", "test")
        self.client.chat.assert_not_called()
        temporal.write_json(self.data / "manifest.json", {"complete": True})
        temporal.freeze(self.data, self.out, "local:model", "test")
        (self.data / "labels.jsonl").write_text('{"label":"success"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "annotation bytes changed"):
            temporal.run(self.data, self.out, "test")
        self.client.chat.assert_not_called()

    def test_prediction_allowlist_blocks_label_before_client_creation(self):
        with patch.object(temporal.core, "_create_client") as create:
            with self.assertRaisesRegex(ValueError, "Only task"):
                temporal.predict(b"image", {"description": "task", "substages": [], "onset": 1}, [0, 1],
                                 model="local:model", expected_digest="fixed")
            create.assert_not_called()

    def test_cli_uses_shared_cross_process_inference_lock(self):
        args = ["temporal", "run", "--data", str(self.data), "--out", str(self.out), "--split", "development"]
        with patch("sys.argv", args), patch.object(temporal, "run") as run, patch.object(temporal, "FileLock") as lock, patch.object(temporal, "default_data_dir", return_value=self.root / "shared"):
            temporal.main()
        run.assert_called_once()
        self.assertEqual(lock.call_count, 2)
        self.assertTrue(lock.call_args_list[1].args[0].endswith("inference.lock"))


if __name__ == "__main__":
    unittest.main()
