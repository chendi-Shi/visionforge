"""Synthetic protocol/invariance tests; no real benchmark fitting or encoder."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from robotbench import relational


def fixture():
    samples, rows = [], []
    for group in range(4):
        for visual in (-1, 1):
            for goal in (-1, 1):
                identifier = f"g{group}-v{visual}-t{goal}"
                frames = [{"phase": phase, "view": str(view),
                           "sha256": hashlib.sha256(f"{group}/{visual}/{phase}/{view}".encode()).hexdigest()}
                          for phase in ("before", "after") for view in range(3)]
                samples.append({"sample_id": identifier, "task_group": f"g{group}", "episode_group": f"episode-{group}-{visual}",
                                "subtask_instruction": f"target {goal}", "task_instruction": "synthetic task", "frames": frames,
                                "label": "success" if visual == goal else "failure"})
                rows.append({"sample_id": identifier, "before": [0, 0, 1], "after": [visual, 0, 0], "text": [0, goal, 0]})
    return samples, {"identity": {"dimension": 3, "model": "synthetic"}, "input_sha256": relational._input_digest(samples), "rows": rows}


class RelationalKernelTests(unittest.TestCase):
    def test_product_represents_instruction_visual_xor_with_identical_pixels(self):
        samples, features = fixture()
        space = relational.FeatureSpace([s["sample_id"] for s in samples], features)
        fit, query, _ = space.kernels(list(range(4)), list(range(4, 8)))
        labels = [s["label"] for s in samples[:4]]
        expected = [s["label"] for s in samples[4:8]]
        predicted = {}
        for method in relational.METHODS:
            train = relational.combine_kernel(fit, method)
            test = relational.combine_kernel(query, method)
            predicted[method], _, status = relational._fit_predict(labels, train, test, 10)
            self.assertEqual(status["status"], "ok")
            np.testing.assert_allclose(np.diag(train), 1)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(train).min()), -1e-10)
        self.assertEqual(predicted["kernel_relational"], expected)
        self.assertLess(sum(a == b for a, b in zip(predicted["kernel_additive"], expected)), 4)
        self.assertEqual(predicted["kernel_vision"][0], predicted["kernel_vision"][1])
        self.assertEqual(predicted["kernel_vision"][2], predicted["kernel_vision"][3])

    def test_zero_change_is_preserved_and_a_constant_delta_has_valid_kernel(self):
        samples, features = fixture()
        for row in features["rows"]:
            row["before"] = row["after"][:]
        space = relational.FeatureSpace([s["sample_id"] for s in samples], features)
        fit, query, bandwidth = space.kernels([0, 1, 2, 3], [4, 5])
        self.assertTrue(bandwidth["delta"]["constant_block"])
        np.testing.assert_array_equal(space.blocks["delta"], 0)
        np.testing.assert_array_equal(fit["delta"], np.ones((4, 4)))
        np.testing.assert_array_equal(query["delta"], np.ones((2, 4)))

    def test_feature_allowlist_rejects_oracle_fields_and_wrong_input_identity(self):
        samples, features = fixture()
        changed = deepcopy(features)
        changed["rows"][0]["failure_category"] = "oracle"
        with self.assertRaisesRegex(ValueError, "Only before/after/text"):
            relational.FeatureSpace([s["sample_id"] for s in samples], changed)
        changed = deepcopy(samples)
        changed[0]["subtask_instruction"] += " altered"
        with self.assertRaisesRegex(ValueError, "does not match"):
            relational.nested_cross_validate(changed, features)


class NestedSelectionTests(unittest.TestCase):
    def test_outer_labels_and_oracle_metadata_cannot_change_own_fold_model(self):
        samples, features = fixture()
        first = relational.nested_cross_validate(samples, features, outer_groups=["g0"])
        changed = deepcopy(samples)
        for sample in changed:
            sample.update(failure_category="do not read", execution_reward=999)
            if sample["task_group"] == "g0":
                sample["label"] = "failure" if sample["label"] == "success" else "success"
        second = relational.nested_cross_validate(changed, features, outer_groups=["g0"])
        self.assertEqual(first["folds"], second["folds"])
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertFalse(first["complete_outer_coverage"])
        self.assertEqual(first["evaluated_samples"], 4)

    def test_held_out_feature_distribution_cannot_change_training_bandwidth_or_C(self):
        samples, features = fixture()
        first = relational.nested_cross_validate(samples, features, methods=("kernel_relational",), outer_groups=["g0"])
        changed = deepcopy(features)
        for row in changed["rows"][:4]:
            row["text"], row["after"] = [1, 1, 1], [1, -1, 1]
        second = relational.nested_cross_validate(samples, changed, methods=("kernel_relational",), outer_groups=["g0"])
        self.assertEqual(first["folds"][0]["bandwidths"], second["folds"][0]["bandwidths"])
        self.assertEqual(first["folds"][0]["inner_folds"], second["folds"][0]["inner_folds"])
        self.assertEqual(first["folds"][0]["methods"], second["folds"][0]["methods"])

    def test_episode_and_image_overlap_are_purged_in_outer_and_inner_folds(self):
        samples, features = fixture()
        # Independent sources of overlap: one shared image and one shared episode.
        samples[4]["frames"][0]["sha256"] = samples[0]["frames"][0]["sha256"]
        samples[8]["episode_group"] = samples[1]["episode_group"]
        samples[12]["frames"][0]["sha256"] = samples[5]["frames"][0]["sha256"]
        features["input_sha256"] = relational._input_digest(samples)
        result = relational.nested_cross_validate(samples, features, methods=("kernel_relational",), c_values=(1,), outer_groups=["g0"])
        outer = result["folds"][0]
        reasons = {p["sample_id"]: p for p in outer["purged"]}
        self.assertTrue(reasons[samples[4]["sample_id"]]["shared_image"])
        self.assertTrue(reasons[samples[8]["sample_id"]]["shared_episode"])
        by_id = {s["sample_id"]: s for s in samples}
        self.assertTrue(any(fold["purged"] for fold in outer["inner_folds"]))
        for fold in outer["inner_folds"]:
            train = [by_id[i] for i in fold["train_sample_ids"]]
            validation = [by_id[i] for i in fold["validation_sample_ids"]]
            self.assertFalse({s["episode_group"] for s in train} & {s["episode_group"] for s in validation})
            self.assertFalse({f["sha256"] for s in train for f in s["frames"]} & {f["sha256"] for s in validation for f in s["frames"]})
        self.assertEqual(len(result["predictions"]["kernel_relational"]), 4)

    def test_all_candidates_score_same_rows_and_C_ties_choose_smaller_value(self):
        samples, features = fixture()
        result = relational.nested_cross_validate(samples, features, methods=("kernel_vision",), outer_groups=["g0"])
        fold = result["folds"][0]
        candidate = fold["methods"]["kernel_vision"]
        for row in candidate["candidates"]:
            self.assertEqual(set(row["validation_sample_ids"]), set(fold["train_sample_ids"]))
            self.assertEqual(len(row["predictions"]), len(fold["train_sample_ids"]))
        self.assertEqual(len({c["class_balanced_group_recall"] for c in candidate["candidates"]}), 1)
        self.assertEqual(candidate["selected_C"], 0.1)

    def test_single_class_validation_group_has_defined_prespecified_scoring(self):
        samples = [{"label": "failure", "task_group": "small"},
                   {"label": "failure", "task_group": "large"},
                   {"label": "failure", "task_group": "large"},
                   {"label": "success", "task_group": "large"}]
        # Failure: mean(group recalls 1, .5)=.75; success: 0; mean=.375.
        self.assertEqual(relational.class_balanced_group_recall(samples, [0, 1, 2, 3], ["failure", "failure", "abstain", "failure"]), 0.375)

    def test_single_class_training_keeps_denominator_and_fixed_C_without_test_labels(self):
        samples, features = fixture()
        for sample in samples:
            if sample["task_group"] != "g0":
                sample["label"] = "success"
        result = relational.nested_cross_validate(samples, features, methods=("kernel_relational",), outer_groups=["g0"])
        method = result["folds"][0]["methods"]["kernel_relational"]
        self.assertEqual(method["selected_C"], 1)
        self.assertEqual(method["selection_status"], "undefined_inner_target_fixed_nearest_C_1")
        self.assertEqual(method["outer_fit"]["status"], "single_class")
        self.assertEqual(len(result["predictions"]["kernel_relational"]), 4)
        self.assertTrue(all(r["prediction"] == "success" for r in result["predictions"]["kernel_relational"]))

    def test_solver_errors_remain_abstentions_in_every_candidate_and_test_denominator(self):
        samples, features = fixture()
        with patch("sklearn.svm.SVC.fit", side_effect=ValueError("injected solver failure")):
            result = relational.nested_cross_validate(samples, features, methods=("kernel_relational",), outer_groups=["g0"])
        method = result["folds"][0]["methods"]["kernel_relational"]
        self.assertEqual(method["outer_fit"]["status"], "fit_error")
        self.assertTrue(all(c["class_balanced_group_recall"] == 0 for c in method["candidates"]))
        self.assertEqual(len(result["predictions"]["kernel_relational"]), 4)
        self.assertTrue(all(r["prediction"] == "abstain" for r in result["predictions"]["kernel_relational"]))


if __name__ == "__main__":
    unittest.main()
