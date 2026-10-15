"""Review regressions for a fixed, label-blind post-hoc protocol; no fitting."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robotbench.data import sha256_file
from robotbench.probe import _hash
from robotbench.sensitivity import assign_groups, make_plan, execute_plan, GROUPS, FIXED_CONFIG, METHODS


class SensitivityProtocolReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.samples = [{"sample_id": f"sample-{index}", "task_group": group,
                         "label": "success" if index % 2 else "failure",
                         "subtask_instruction": "fixed task instruction", "frames": []}
                        for index, group in enumerate(member for members in GROUPS.values() for member in members)]
        self.enterContext(patch("robotbench.sensitivity.load_samples", side_effect=lambda _: deepcopy(self.samples)))
        self.runtime = self.root / "protocol.py"
        self.runtime.write_text("# Fixed source fixture; never executed.\n", encoding="utf-8")
        self.enterContext(patch("robotbench.sensitivity._runtime_sources", return_value={"protocol.py": self.runtime}))
        for name in ("inputs.jsonl", "labels.jsonl"):
            (self.root / name).write_text("source byte identity fixture\n", encoding="utf-8")
        self.features = self.root / "features.json"
        self.features.write_text(json.dumps({"input_sha256": _hash([{key: s[key] for key in ("sample_id", "frames", "subtask_instruction")} for s in self.samples]),
                                             "identity": {"encoder": "fixture"}, "rows": []}), encoding="utf-8")
        self.reference = self.root / "primary.json"
        repository = Path(__file__).resolve().parents[1]
        reference = {"inputs_sha256": sha256_file(self.root / "inputs.jsonl"),
                     "labels_sha256": sha256_file(self.root / "labels.jsonl"), "features_sha256": sha256_file(self.features),
                     "config": FIXED_CONFIG, "predictions": {method: [] for method in METHODS},
                     "code_sha256": sha256_file(repository / "robotbench/probe.py"),
                     "models_code_sha256": sha256_file(repository / "robotbench/models.py")}
        self.reference.write_text(json.dumps(reference), encoding="utf-8")

    def plan(self):
        return make_plan(self.root, self.features, self.reference, self.output)

    def test_mapping_is_exhaustive_immutable_and_label_independent(self):
        original = deepcopy(self.samples)
        assigned = assign_groups(self.samples)
        self.assertEqual(self.samples, original)
        self.assertEqual(len(assigned), 16)
        self.assertEqual(len({row["task_group"] for row in assigned}), 4)
        for sample in self.samples:
            sample["label"] = "failure" if sample["label"] == "success" else "success"
            sample["model_score"] = 0.999
        changed = assign_groups(self.samples)
        self.assertEqual([(r["sample_id"], r["original_task_group"], r["task_group"]) for r in assigned],
                         [(r["sample_id"], r["original_task_group"], r["task_group"]) for r in changed])
        with self.assertRaisesRegex(ValueError, "cover every"):
            assign_groups(self.samples[:-1])
        with self.assertRaisesRegex(ValueError, "cover every"):
            assign_groups(self.samples + [{**self.samples[0], "sample_id": "extra", "task_group": "unknown_family"}])

    def test_existing_plan_and_runtime_are_not_rewritten_when_source_drifts(self):
        first = self.plan()
        self.assertEqual(self.plan(), first)
        plan_path = self.output / "plan.json"
        old_plan = plan_path.read_bytes()
        snapshot = self.output / "runtime/protocol.py"
        old_snapshot = snapshot.read_bytes()
        self.runtime.write_text("# Changed computation after freezing.\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen sensitivity plan changed"):
            self.plan()
        self.assertEqual(plan_path.read_bytes(), old_plan)
        self.assertEqual(snapshot.read_bytes(), old_snapshot)

    def test_separate_plan_is_required_and_feature_drift_cannot_be_refit(self):
        with patch("robotbench.sensitivity.cross_validate") as fit:
            with self.assertRaisesRegex(ValueError, "separate plan command"):
                execute_plan(self.root, self.features, self.reference, self.output)
            fit.assert_not_called()
            self.plan()
            contents = json.loads(self.features.read_text(encoding="utf-8"))
            contents["rows"] = [{"sample_id": "sample-0", "text": [0, 1]}]
            self.features.write_text(json.dumps(contents), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact data and features"):
                execute_plan(self.root, self.features, self.reference, self.output)
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
