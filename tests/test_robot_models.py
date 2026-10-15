"""Offline action-effect contracts and real train-fold isolation checks."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from robotbench.metrics import evaluate
from robotbench.models import feature_vector, judge_effect, prepare_panel
from robotbench.probe import cross_validate


class RobotPanelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames = []
        for phase, color in (("before", (200, 0, 0)), ("after", (0, 200, 0))):
            for view in ("front", "left", "right"):
                path = self.root / f"private-source-{phase}-{view}.png"
                Image.new("RGB", (40, 30), color).save(path)
                self.frames.append({"frame_id": f"{phase}_{view}", "phase": phase, "view": view,
                                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.model_input = {"task": "Put the apple in the bowl", "subtask": "Pick up the apple", "frames": self.frames}

    def test_panel_preserves_phase_camera_correspondence_and_source_hashes(self):
        before = deepcopy(self.model_input)
        picture, audit = prepare_panel(self.model_input)
        self.assertEqual(self.model_input, before)
        self.assertEqual(audit["size"], [768, 560])
        self.assertEqual(audit["phases"], ["before", "after"])
        self.assertEqual([(f["phase"], f["view"]) for f in audit["frames"]], [(f["phase"], f["view"]) for f in self.frames])
        self.assertEqual([f["sha256"] for f in audit["frames"]], [f["sha256"] for f in self.frames])
        self.assertNotIn(str(self.root), json.dumps(audit))
        with Image.open(io.BytesIO(picture)) as panel:
            for column in range(3):
                self.assertEqual(panel.getpixel((column * 256 + 128, 150)), (200, 0, 0))
                self.assertEqual(panel.getpixel((column * 256 + 128, 430)), (0, 200, 0))
        _, after = prepare_panel({**self.model_input, "frames": self.frames[3:]})
        self.assertEqual(after["phases"], ["after"])
        self.assertEqual(after["size"], [768, 280])

    def test_mismatched_and_duplicate_camera_slots_are_rejected(self):
        bad = deepcopy(self.frames)
        bad[-1] = deepcopy(bad[-2])
        for frames in (self.frames[:-1], bad, [{**self.frames[0], "phase": "ground_truth"}]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                prepare_panel({**self.model_input, "frames": frames})

    def test_labels_and_changed_image_are_rejected_before_client_creation(self):
        for extra in ("label", "failure_category", "execution_reward"):
            with patch("robotbench.models.core._create_client") as create:
                result = judge_effect({**self.model_input, extra: "failure"})
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["prediction"], "abstain")
                create.assert_not_called()
        Image.new("RGB", (40, 30), (0, 0, 255)).save(self.frames[0]["path"])
        with patch("robotbench.models.core._create_client") as create:
            result = judge_effect(self.model_input)
            self.assertIn("checksum", result["error"])
            self.assertEqual(result["prediction"], "abstain")
            create.assert_not_called()

    def test_real_call_contract_contains_only_inference_fields_and_closes_client(self):
        client = Mock()
        client.chat.return_value = {"message": {"content": '{"prediction":"failure","evidence":"The gripper is empty."}'}, "done_reason": "stop"}
        with patch("robotbench.models.core._create_client", return_value=client), patch(
                "robotbench.models.core._inspect_model", return_value=("local:vision", "fixed-digest", {"capabilities": ["vision"]})):
            result = judge_effect(self.model_input, expected_digest="fixed-digest")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prediction"], "failure")
        self.assertEqual(result["digest"], "fixed-digest")
        request = client.chat.call_args.kwargs
        question = json.loads(request["messages"][1]["content"])
        self.assertEqual(set(question), {"overall_task", "subtask_to_evaluate"})
        self.assertEqual(question["subtask_to_evaluate"], self.model_input["subtask"])
        self.assertIsInstance(request["messages"][1]["images"][0], bytes)
        self.assertNotIn(str(self.root), result["question"])
        self.assertEqual(request["options"]["temperature"], 0)
        client.close.assert_called_once()

    def test_truncation_schema_error_and_digest_change_abstain(self):
        raw_cases = [("length", '{"prediction":"success","evidence":"Seen"}'),
                     ("stop", '{"prediction":"success","evidence":"Seen","extra":"label"}')]
        for done, raw in raw_cases:
            client = Mock()
            client.chat.return_value = {"message": {"content": raw}, "done_reason": done}
            with patch("robotbench.models.core._create_client", return_value=client), patch(
                    "robotbench.models.core._inspect_model", return_value=("local:vision", "digest", {})):
                result = judge_effect(self.model_input)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["prediction"], "abstain")
            self.assertEqual(result["raw_response"], raw)
            client.close.assert_called_once()
        client = Mock()
        with patch("robotbench.models.core._create_client", return_value=client), patch(
                "robotbench.models.core._inspect_model", return_value=("local:vision", "changed", {})):
            result = judge_effect(self.model_input, expected_digest="expected")
        self.assertEqual(result["prediction"], "abstain")
        client.chat.assert_not_called()
        client.close.assert_called_once()


class RobotProbeIsolationTests(unittest.TestCase):
    def setUp(self):
        self.samples, rows = [], []
        for group in range(4):
            for index, label in enumerate(("success", "failure")):
                identifier = f"g{group}-{index}"
                # Both counterfactual labels in family 0 share pixels. Another
                # family shares those pixels too, forcing cross-family purging.
                digest = "shared" if group == 0 or (group == 1 and index == 0) else identifier
                self.samples.append({"sample_id": identifier, "label": label, "task_group": f"family-{group}",
                                     "failure_category": "grasp" if label == "failure" else None,
                                     "frames": [{"sha256": hashlib.sha256(digest.encode()).hexdigest()}]})
                rows.append({"sample_id": identifier, "before": [1 + group, 1 + index, 2],
                             "after": [2 + index, 2 + group, 1], "text": [1, group + 1, index + 1]})
        self.features = {"rows": rows}

    def test_each_sample_is_held_out_once_and_shared_images_only_purge_training(self):
        result = cross_validate(self.samples, self.features)
        by_id = {sample["sample_id"]: sample for sample in self.samples}
        for fold in result["folds"]:
            test_groups = {by_id[identifier]["task_group"] for identifier in fold["test_sample_ids"]}
            train_groups = {by_id[identifier]["task_group"] for identifier in fold["train_sample_ids"]}
            self.assertEqual(test_groups, {fold["test_task_group"]})
            self.assertFalse(test_groups & train_groups)
            train_hashes = {frame["sha256"] for identifier in fold["train_sample_ids"] for frame in by_id[identifier]["frames"]}
            test_hashes = {frame["sha256"] for identifier in fold["test_sample_ids"] for frame in by_id[identifier]["frames"]}
            self.assertFalse(train_hashes & test_hashes)
        first = next(fold for fold in result["folds"] if fold["test_task_group"] == "family-0")
        self.assertIn("g1-0", first["purged_shared_image_ids"])
        self.assertEqual(first["test_sample_ids"], ["g0-0", "g0-1"])
        for method, predictions in result["predictions"].items():
            self.assertEqual(len(predictions), len(self.samples), method)
            self.assertEqual({p["sample_id"] for p in predictions}, set(by_id), method)
            self.assertTrue(all(p["held_out_group"] == by_id[p["sample_id"]]["task_group"] for p in predictions))
            measured = evaluate(self.samples, predictions, bootstrap_iterations=0)
            self.assertEqual(measured["n"], 8)
            self.assertEqual(measured["per_class"]["failure"]["support"], 4)

    def test_scaler_fit_receives_only_purged_training_rows_for_every_method(self):
        from sklearn.preprocessing import StandardScaler
        original_fit, observed = StandardScaler.fit, []

        def record_fit(instance, matrix, y=None, **kwargs):
            observed.append(np.asarray(matrix).copy())
            return original_fit(instance, matrix, y, **kwargs)

        with patch.object(StandardScaler, "fit", record_fit):
            result = cross_validate(self.samples, self.features)
        by_id = {row["sample_id"]: row for row in self.features["rows"]}
        expected = []
        for fold in result["folds"]:
            if len(fold["train_class_counts"]) < 2:
                continue
            for method in ("text_only", "after", "transition"):
                expected.append(np.asarray([feature_vector(by_id[identifier]["before"], by_id[identifier]["after"], by_id[identifier]["text"], method)
                                            for identifier in fold["train_sample_ids"]]))
        self.assertEqual(len(observed), len(expected))
        for actual, wanted in zip(observed, expected):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=0)

    def test_missing_or_duplicate_features_cannot_silently_drop_negative_samples(self):
        for rows in (self.features["rows"][:-1], self.features["rows"] + [self.features["rows"][0]]):
            with self.assertRaisesRegex(ValueError, "exactly match"):
                cross_validate(self.samples, {"rows": rows})
        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            cross_validate(self.samples + [self.samples[0]], self.features)


if __name__ == "__main__":
    unittest.main()
