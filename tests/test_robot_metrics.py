"""Evaluation tests use hand-calculable outcomes, never fabricated model scores."""
from copy import deepcopy
import hashlib
import unittest

from robotbench.metrics import audit_split, compare_paired, evaluate


def sample(identifier, label, group, split):
    return {"sample_id": identifier, "label": label, "task_group": group,
            "episode_group": group + "-episode-" + identifier, "split": split,
            "failure_category": "grasp" if label == "failure" else None,
            "model_input": {"task_instruction": "Move the object", "subtask_instruction": "Place it inside the bowl",
                "frames": [{"frame_id": "after_front", "phase": "after", "view": "front",
                            "path": identifier + ".png", "sha256": hashlib.sha256(identifier.encode()).hexdigest()}]}}


class RobotMetricTests(unittest.TestCase):
    def setUp(self):
        self.samples = [sample("s1", "success", "task-a", "dev"), sample("s2", "success", "task-a", "dev"),
                        sample("f1", "failure", "task-b", "test"), sample("f2", "failure", "task-b", "test")]
        self.predictions = [{"sample_id": "s1", "prediction": "success"}, {"sample_id": "s2", "prediction": "failure"},
                            {"sample_id": "f1", "prediction": "failure"}]

    def test_missing_prediction_stays_in_all_primary_denominators(self):
        before = deepcopy((self.samples, self.predictions))
        result = evaluate(self.samples, self.predictions, bootstrap_iterations=0)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 1 / 2) / 2)
        self.assertEqual(result["coverage"], .75)
        self.assertAlmostEqual(result["selective_accuracy"], 2 / 3)
        self.assertEqual(result["confusion_matrix"]["failure"]["abstain"], 1)
        self.assertEqual(result["abstention_reasons"], {"missing_prediction": 1})
        self.assertEqual(result["per_failure_category"]["grasp"]["failure_recall"], .5)
        self.assertEqual(result["per_failure_category"]["grasp"]["n"], 2)
        self.assertEqual((self.samples, self.predictions), before)

    def test_invalid_failed_and_explicit_abstention_cannot_raise_accuracy(self):
        predictions = [{"sample_id": "s1", "prediction": "success", "status": "failed"},
                       {"sample_id": "s2", "prediction": "abstain"}, {"sample_id": "f1", "prediction": "maybe"}]
        result = evaluate(self.samples, predictions, bootstrap_iterations=0)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["balanced_accuracy"], 0)
        self.assertEqual(result["macro_f1"], 0)
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["selective_accuracy"])
        self.assertEqual(result["abstention_reasons"], {"failed_prediction": 1, "explicit_abstention": 1, "invalid_prediction": 1, "missing_prediction": 1})

    def test_class_imbalance_is_visible_and_single_class_balanced_accuracy_undefined(self):
        all_failure = [{"sample_id": row["sample_id"], "prediction": "failure"} for row in self.samples]
        result = evaluate(self.samples, all_failure, bootstrap_iterations=0)
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["per_class"]["success"]["recall"], 0)
        one_class = evaluate(self.samples[:2], self.predictions[:2], bootstrap_iterations=0)
        self.assertIsNone(one_class["balanced_accuracy"])
        self.assertEqual(one_class["per_class"]["failure"]["support"], 0)

    def test_task_group_bootstrap_is_reproducible_and_reports_undefined_replicates(self):
        result = evaluate(self.samples, self.predictions, bootstrap_iterations=200, seed=17)
        again = evaluate(self.samples, self.predictions, bootstrap_iterations=200, seed=17)
        self.assertEqual(result["bootstrap"], again["bootstrap"])
        bootstrap = result["bootstrap"]
        self.assertEqual(bootstrap["n_groups"], 2)
        self.assertIn("exploratory", bootstrap["warning"])
        balanced = bootstrap["metrics"]["balanced_accuracy"]
        self.assertGreater(balanced["undefined_iterations"], 0)
        self.assertEqual(balanced["valid_iterations"] + balanced["undefined_iterations"], 200)
        single = evaluate(self.samples[:2], self.predictions[:2], bootstrap_iterations=100)
        self.assertEqual(single["bootstrap"]["status"], "insufficient_task_groups")
        self.assertIsNone(single["bootstrap"]["metrics"]["accuracy"]["lower"])

    def test_paired_comparison_matches_all_samples_including_missing_outputs(self):
        perfect = [{"sample_id": row["sample_id"], "prediction": row["label"]} for row in self.samples]
        result = compare_paired(self.samples, self.predictions, perfect, bootstrap_iterations=100, seed=8)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["jointly_answered"], 3)
        self.assertEqual(result["delta"]["accuracy"], .5)
        self.assertEqual(result["delta"]["coverage"], .25)
        self.assertEqual(result["paired_outcomes"], {"both_correct": 2, "a_only_correct": 0, "b_only_correct": 2, "both_wrong_or_abstain": 0})
        identical = compare_paired(self.samples, self.predictions, self.predictions, bootstrap_iterations=100, seed=8)
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "coverage", "task_macro_accuracy"):
            self.assertEqual(identical["delta"][metric], 0)
            self.assertEqual(identical["bootstrap_delta"]["metrics"][metric]["lower"], 0)
            self.assertEqual(identical["bootstrap_delta"]["metrics"][metric]["upper"], 0)

    def test_prediction_ids_and_invalid_manifest_fail_loudly(self):
        for predictions in (self.predictions + [self.predictions[0]], [{"sample_id": "not-selected", "prediction": "failure"}]):
            with self.assertRaises(ValueError):
                evaluate(self.samples, predictions, bootstrap_iterations=0)
        for samples in ([], self.samples + [self.samples[0]], [{**self.samples[0], "label": "unknown"}], [{**self.samples[0], "task_group": ""}]):
            with self.assertRaises(ValueError):
                evaluate(samples, [], bootstrap_iterations=0)
        with self.assertRaises(ValueError):
            evaluate(self.samples, [], bootstrap_iterations=True)

    def test_split_audit_catches_task_episode_hash_and_groundtruth_leakage(self):
        result = audit_split(self.samples)
        self.assertTrue(result["ok"])
        self.assertEqual(result["split_counts"], {"dev": 2, "test": 2})
        changed = deepcopy(self.samples)
        changed[2]["task_group"] = changed[0]["task_group"]
        changed[2]["episode_group"] = changed[0]["episode_group"]
        changed[2]["model_input"]["frames"][0]["sha256"] = changed[0]["model_input"]["frames"][0]["sha256"]
        changed[2]["model_input"]["label"] = "failure"
        bad = audit_split(changed)
        self.assertFalse(bad["ok"])
        overlaps = {issue.get("field") for issue in bad["issues"] if issue["kind"] == "cross_split_overlap"}
        self.assertEqual(overlaps, {"task_group", "episode_group", "image_sha256"})
        self.assertTrue(any(issue["kind"] == "forbidden_inference_field" for issue in bad["issues"]))

    def test_split_audit_requires_images_and_reports_missing_partition(self):
        changed = deepcopy(self.samples[:2])
        changed[0]["model_input"]["frames"] = []
        changed[1].pop("episode_group")
        bad = audit_split(changed)
        kinds = {issue["kind"] for issue in bad["issues"]}
        self.assertIn("missing_required_split", kinds)
        self.assertIn("missing_inference_frames", kinds)
        self.assertIn("missing_episode_group", kinds)


if __name__ == "__main__":
    unittest.main()
