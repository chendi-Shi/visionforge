from copy import deepcopy
import json
import subprocess
import sys
import unittest

from robotbench.temporal_evaluation import controls, evaluate, compare_paired, evaluate_all


def fixture():
    episodes, labels = [], []
    for identifier, task, label, onset in (("f1", "a", "failure", 2.0), ("s1", "a", "success", None),
                                            ("f2", "b", "failure", 7.0), ("s2", "b", "success", None)):
        episodes.append({"sample_id": identifier, "task_id": task, "split": "test", "timestamp": [0., 1.2, 2., 3.8, 7., 10.], "duration_seconds": 10.1})
        labels.append({"sample_id": identifier, "task_id": task, "split": "test", "label": label, "onset_seconds": onset})
    return labels, episodes


def prediction(identifier, verdict="failure", frame=1, method="uniform"):
    times = [0., 3.8, 10.]
    return {"sample_id": identifier, "method": method, "status": "ok", "prediction": verdict,
            "frame_id": frame if verdict == "failure" else -1, "onset_seconds": times[frame] if verdict == "failure" else None,
            "selection": {"indices": [0, 3, 5], "timestamps": times, "budget": 3, "scores": [0, 1, 2, 3, 1, 0]}}


class TemporalEvaluationTests(unittest.TestCase):
    def test_full_denominators_and_conditional_mae_are_separate(self):
        labels, episodes = fixture()
        records = [prediction("f1"), prediction("s1", frame=0), prediction("s2", verdict="abstain")]
        result = evaluate(labels, records, episodes, bootstrap_iterations=0)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accuracy"], .25)
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["failure_recall"], .5)
        self.assertEqual(result["success_false_alarm_rate"], .5)
        self.assertEqual(result["valid_onset_coverage"], .5)
        self.assertAlmostEqual(result["conditional_onset_mae_seconds"], 1.8)
        self.assertEqual(result["hit_at_1s"], 0)
        self.assertEqual(result["hit_at_2s"], .5)
        self.assertEqual(result["hit_at_5s"], .5)
        self.assertEqual(result["sampler_oracle_at_2s"], .5)
        self.assertEqual(result["abstention_reasons"], {"explicit_abstention": 1, "missing_prediction": 1})

    def test_invalid_onset_retains_detection_but_cannot_improve_localization(self):
        labels, episodes = fixture()
        for invalid in (None, float("nan"), float("inf"), True, -1, 1000, 2.0):
            with self.subTest(onset=invalid):
                row = prediction("f1")
                row["onset_seconds"] = invalid
                result = evaluate(labels, [row], episodes, bootstrap_iterations=0)
                self.assertEqual(result["failure_recall"], .5)
                self.assertEqual(result["hit_at_5s"], 0)
                self.assertEqual(result["valid_onset_coverage"], 0)
                self.assertIsNone(result["conditional_onset_mae_seconds"])

    def test_failed_response_abstains_but_sampler_oracle_is_independent(self):
        labels, episodes = fixture()
        row = prediction("f1")
        row["status"] = "failed"
        result = evaluate(labels, [row], episodes, bootstrap_iterations=0)
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["failure_recall"], 0)
        self.assertEqual(result["sampler_oracle_at_2s"], .5)
        self.assertEqual(result["sampler_valid_coverage"], .5)

    def test_selection_must_match_measured_timestamp_and_valid_index(self):
        labels, episodes = fixture()
        row = prediction("f1")
        row["selection"]["timestamps"][1] = 2.0  # fabricated perfect onset
        row["onset_seconds"] = 2.0
        result = evaluate(labels, [row], episodes, bootstrap_iterations=0)
        self.assertEqual(result["hit_at_1s"], 0)
        self.assertEqual(result["sampler_oracle_at_1s"], 0)
        self.assertEqual(result["sampler_valid_coverage"], 0)

    def test_labels_and_prediction_ids_fail_closed(self):
        labels, episodes = fixture()
        with self.assertRaisesRegex(ValueError, "Unknown or duplicate"):
            evaluate(labels, [prediction("f1"), prediction("f1")], episodes)
        with self.assertRaisesRegex(ValueError, "Unknown or duplicate"):
            evaluate(labels, [prediction("unknown")], episodes)
        with self.assertRaisesRegex(ValueError, "Missing labels"):
            evaluate(labels[:-1], [], episodes)
        altered = deepcopy(labels)
        altered[0]["onset_seconds"] = None
        with self.assertRaisesRegex(ValueError, "author onset"):
            evaluate(altered, [], episodes)

    def test_no_failure_cohort_returns_undefined_rates_instead_of_perfect_scores(self):
        labels, episodes = fixture()
        labels, episodes = [x for x in labels if x["label"] == "success"], [x for x in episodes if x["sample_id"].startswith("s")]
        result = evaluate(labels, [], episodes, bootstrap_iterations=20)
        self.assertIsNone(result["failure_recall"])
        self.assertIsNone(result["hit_at_2s"])
        self.assertIsNone(result["conditional_onset_mae_seconds"])
        self.assertIsNone(result["bootstrap"]["metrics"]["hit_at_2s"]["lower"])
        self.assertEqual(result["bootstrap"]["metrics"]["hit_at_2s"]["undefined_iterations"], 20)

    def test_paired_bootstrap_matches_full_IDs_and_is_seed_reproducible(self):
        labels, episodes = fixture()
        a = [prediction("f1")]
        b = [prediction("f1"), prediction("f2", frame=2)]
        first = compare_paired(labels, a, b, episodes, bootstrap_iterations=80)
        second = compare_paired(labels, a, b, episodes, bootstrap_iterations=80)
        self.assertEqual(first, second)
        self.assertEqual(first["n"], 4)
        self.assertEqual(first["bootstrap_delta"]["n_groups"], 2)
        self.assertEqual(first["bootstrap_delta"]["seed"], 1337)
        self.assertEqual(first["delta"]["hit_at_5s"], .5)
        self.assertEqual(first["joint_valid_localized_failure_n"], 1)
        self.assertEqual(first["joint_conditional_mae_delta_seconds"], 0)
        self.assertGreater(first["delta"]["conditional_onset_mae_seconds"], 0)

    def test_controls_ignore_annotations_and_use_measured_times(self):
        _, episodes = fixture()
        episodes = [{**episodes[0], "timestamp": [0, .05, 10.], "duration_seconds": 10.1}]
        records = [{"sample_id": "f1", "selection": {"scores": [0, 3, 0]}, "prediction": "success", "onset_seconds": None}]
        first = controls(episodes, records)
        episodes[0].update(label="failure", onset_seconds=9.5, failure_types=["oracle"])
        records[0].update(prediction="failure", onset_seconds=9.5)
        second = controls(episodes, records)
        self.assertEqual(first, second)
        self.assertEqual(first["clock_midpoint"][0]["onset_seconds"], 5)
        self.assertEqual(first["state_peak"][0]["onset_seconds"], .05)
        self.assertEqual(first["state_peak"][0]["prediction"], "failure")

    def test_state_peak_missing_scores_abstains_and_equal_scores_choose_earliest(self):
        _, episodes = fixture()
        result = controls(episodes, [{"sample_id": "f1", "selection": {"scores": [0] * 6}}])
        self.assertEqual(result["state_peak"][0]["onset_seconds"], 0)
        self.assertTrue(result["state_peak"][0]["all_scores_zero"])
        self.assertEqual(result["state_peak"][1]["prediction"], "abstain")
        self.assertEqual(len(result["state_peak"]), 4)
        with self.assertRaisesRegex(ValueError, "disagree"):
            controls(episodes, [{"sample_id": "f1", "selection": {"scores": [0] * 6}},
                                {"sample_id": "f1", "selection": {"scores": [1] * 6}}])

    def test_missing_method_keeps_every_episode_in_combined_report(self):
        labels, episodes = fixture()
        result = evaluate_all(labels, [prediction("f1")], episodes, bootstrap_iterations=0)
        self.assertEqual(result["methods"]["state_event"]["n"], 4)
        self.assertEqual(result["methods"]["state_event"]["abstained"], 4)
        self.assertEqual(result["paired"]["uniform_to_state_event"]["n"], 4)
        self.assertEqual(result["methods"]["clock_midpoint"]["success_false_alarm_rate"], 1)

    def test_module_import_requires_only_standard_library(self):
        program = "import sys,json; import robotbench.temporal_evaluation; print(json.dumps(sorted(set(sys.modules)&{'numpy','h5py','ollama','cv2','PIL','scipy'})))"
        result = subprocess.run([sys.executable, "-B", "-c", program], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
