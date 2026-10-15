"""Counterfactual scoring retains unanswered pairs and exposes label conflicts."""
from copy import deepcopy
import hashlib
import unittest

from robotbench.diagnostics import counterfactual_pairs


def sample(identifier, label, subtask, pixels="shared", task="Place objects"):
    frames = []
    for phase in ("before", "after"):
        for view in ("0", "1", "2"):
            frames.append({"phase": phase, "view": view, "sha256": hashlib.sha256(f"{pixels}-{phase}-{view}".encode()).hexdigest()})
    return {"sample_id": identifier, "label": label, "task_group": "family", "task_instruction": task,
            "subtask_instruction": subtask, "frames": frames}


class RobotDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.samples = [sample("a", "success", "Pick up apple"), sample("b", "failure", "Pick up pear"),
                        sample("c", "success", "Close drawer", pixels="other"), sample("d", "failure", "Open drawer", pixels="other")]

    def test_same_pixels_different_instruction_requires_both_labels_correct(self):
        predictions = [{"sample_id": "a", "prediction": "success"}, {"sample_id": "b", "prediction": "failure"},
                       {"sample_id": "c", "prediction": "success"}, {"sample_id": "d", "prediction": "success"}]
        before = deepcopy((self.samples, predictions))
        result = counterfactual_pairs(self.samples, predictions)
        self.assertEqual(result["eligible_pairs"], 2)
        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["both_correct_rate"], .5)
        self.assertEqual(result["pair_coverage"], 1)
        self.assertEqual(result["same_binary_prediction"], 1)
        self.assertEqual(result["opposite_binary_predictions"], 1)
        self.assertEqual((self.samples, predictions), before)

    def test_missing_failed_and_abstaining_outputs_do_not_remove_pairs(self):
        predictions = [{"sample_id": "a", "prediction": "success"},
                       {"sample_id": "c", "prediction": "success", "status": "failed"},
                       {"sample_id": "d", "prediction": "abstain"}]
        result = counterfactual_pairs(self.samples, predictions)
        self.assertEqual(result["eligible_pairs"], 2)
        self.assertEqual(result["both_correct_rate"], 0)
        self.assertEqual(result["pair_coverage"], 0)
        self.assertEqual(result["one_answered"], 1)
        self.assertEqual(result["neither_answered"], 1)
        self.assertEqual(result["same_binary_prediction"], 0)
        self.assertEqual(result["same_prediction_including_abstention"], 1)
        self.assertIsNone(result["selective_both_correct_rate"])

    def test_identical_input_conflicts_and_changed_overall_task_are_explicit(self):
        samples = [sample("a", "success", "Pick up apple"), sample("b", "failure", "Pick up apple"),
                   sample("c", "failure", "Pick up apple", task="A different overall task"),
                   sample("d", "failure", "Pick up pear")]
        result = counterfactual_pairs(samples, [])
        self.assertEqual(result["opposite_label_pairs_total"], 3)
        self.assertEqual(result["eligible_pairs"], 1)
        self.assertEqual(result["identical_input_conflict_pairs"], 1)
        self.assertEqual(result["identical_input_conflict_samples"], 2)
        self.assertEqual(result["identical_input_conflict_groups"], 1)
        self.assertEqual(result["eligible_pairs_involving_conflicted_input"], 1)
        self.assertEqual(result["same_subtask_different_task_pairs"], 1)
        self.assertEqual(len(result["excluded_pairs"]), 2)
        self.assertEqual(result["same_label_pairs_not_scored"], 3)
        self.assertEqual(result["n_samples"], 4)

    def test_no_pairs_returns_null_scores_and_reordered_pixels_do_not_match(self):
        changed = deepcopy(self.samples[:2])
        for view in range(3):
            changed[1]["frames"][view]["sha256"], changed[1]["frames"][view + 3]["sha256"] = (
                changed[1]["frames"][view + 3]["sha256"], changed[1]["frames"][view]["sha256"])
        result = counterfactual_pairs(changed, [])
        self.assertEqual(result["status"], "no_eligible_pairs")
        self.assertEqual(result["eligible_pairs"], 0)
        for metric in ("both_correct_rate", "pair_coverage", "same_binary_prediction_rate"):
            self.assertIsNone(result[metric])

    def test_unknown_or_duplicate_predictions_and_incomplete_observations_reject(self):
        for predictions in ([{"sample_id": "outside", "prediction": "failure"}],
                            [{"sample_id": "a", "prediction": "success"}] * 2):
            with self.assertRaises(ValueError):
                counterfactual_pairs(self.samples, predictions)
        changed = deepcopy(self.samples)
        changed[0]["frames"] = changed[0]["frames"][:3]
        with self.assertRaisesRegex(ValueError, "six ordered"):
            counterfactual_pairs(changed, [])


if __name__ == "__main__":
    unittest.main()
