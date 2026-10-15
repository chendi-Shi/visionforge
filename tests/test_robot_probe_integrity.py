"""Independent integrity regressions; no downloaded data or model inference."""
from contextlib import redirect_stdout
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robotbench.metrics import evaluate
from robotbench import probe
from robotbench import sensitivity


class RecordingEncoder:
    def __init__(self, name="fixed"):
        self.name = name
        self.requests = []

    def identity(self):
        return {"encoder": self.name, "dimension": 3, "threads": 2}

    def embed_images(self, paths):
        self.requests.extend(("image", path) for path in paths)
        return [[1, 2, 3] for _ in paths]

    def embed_text(self, text):
        self.requests.append(("text", text))
        return [3, 2, 1]


class EmbeddingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.asset = self.root / "opaque-pixels.bin"
        self.asset.write_bytes(b"fixed test pixels; never sent to a real encoder")
        digest = hashlib.sha256(self.asset.read_bytes()).hexdigest()
        self.sample = {"sample_id": "fixed-id", "subtask_instruction": "grasp target",
            "frames": [{"path": self.asset.name, "sha256": digest, "phase": phase, "view": str(view)}
                       for phase in ("before", "after") for view in range(3)],
            "label": "failure", "failure_category": "wrong_object", "task_group": "family",
            "episode_group": "episode", "oracle_explanation": "private answer"}
        self.output = self.root / "output"
        self.output.mkdir()

    def encode(self, encoder, sample=None, output=None):
        with redirect_stdout(io.StringIO()):
            return probe.encode_samples([sample or self.sample], self.root, output or self.output, embedder=encoder)

    def test_oracle_metadata_cannot_change_features_or_encoder_requests(self):
        first = RecordingEncoder()
        original = self.encode(first)
        changed = deepcopy(self.sample)
        changed.update(label="success", failure_category="success", task_group="different",
                       episode_group="different", oracle_explanation="opposite answer")
        other_output = self.root / "independent"
        other_output.mkdir()
        second = RecordingEncoder()
        altered = self.encode(second, changed, other_output)
        self.assertEqual(first.requests, second.requests)
        self.assertEqual(first.requests, [("image", str(self.asset.resolve())), ("text", "grasp target")])
        self.assertEqual(original["rows"], altered["rows"])
        self.assertEqual(original["input_sha256"], altered["input_sha256"])
        self.assertNotIn("private answer", json.dumps(original))

    def test_short_cache_names_keep_full_keys_and_reject_wrong_dimension(self):
        encoder = RecordingEncoder()
        self.encode(encoder)
        requests = list(encoder.requests)
        self.encode(encoder)
        self.assertEqual(encoder.requests, requests)
        namespace = self.output / "embeddings" / probe._hash(encoder.identity())[:12]
        vector_files = [p for p in namespace.glob("*.json") if p.name != "identity.json"]
        self.assertEqual(len(vector_files), 2)
        self.assertTrue(all(len(path.stem) == 24 for path in vector_files))
        image = next(path for path in vector_files if json.loads(path.read_text())["key"].startswith("image-"))
        saved = json.loads(image.read_text())
        self.assertEqual(saved["identity"], encoder.identity())
        self.assertEqual(saved["key"], "image-" + self.sample["frames"][0]["sha256"])
        saved["vector"] = [1, 2]
        saved["sha256"] = probe._hash(saved["vector"])
        image.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.encode(encoder)

    def test_actual_short_key_and_namespace_collisions_fail_closed(self):
        original_hash = probe._hash

        def collide_keys(value):
            return "f" * 64 if isinstance(value, str) else original_hash(value)

        with patch.object(probe, "_hash", side_effect=collide_keys):
            with self.assertRaisesRegex(ValueError, "key collision"):
                self.encode(RecordingEncoder())
        other_output = self.root / "namespace-collision"
        other_output.mkdir()

        def collide_namespaces(value):
            if isinstance(value, dict) and "encoder" in value:
                return "0" * 12 + ("a" if value["encoder"] == "first" else "b") * 52
            return original_hash(value)

        with patch.object(probe, "_hash", side_effect=collide_namespaces):
            self.encode(RecordingEncoder("first"), output=other_output)
            with self.assertRaisesRegex(ValueError, "namespace collision"):
                self.encode(RecordingEncoder("second"), output=other_output)

    def test_changed_pixels_are_rejected_even_when_valid_embedding_is_cached(self):
        encoder = RecordingEncoder()
        self.encode(encoder)
        count = len(encoder.requests)
        self.asset.write_bytes(b"changed pixels")
        with self.assertRaisesRegex(ValueError, "changed input image"):
            self.encode(encoder)
        self.assertEqual(len(encoder.requests), count)


class PredictionIntegrityTests(unittest.TestCase):
    def test_held_out_labels_do_not_affect_their_own_predictions(self):
        samples, features = [], {"rows": []}
        for group in range(4):
            for index, label in enumerate(("success", "failure")):
                identifier = f"{group}-{index}"
                samples.append({"sample_id": identifier, "task_group": f"family-{group}", "label": label,
                                "frames": [{"sha256": hashlib.sha256(identifier.encode()).hexdigest()}]})
                features["rows"].append({"sample_id": identifier, "before": [group + 1, index + 1, 2],
                                         "after": [index + 2, group + 2, 1], "text": [1, group + 1, index + 1]})
        first = probe.cross_validate(samples, features)
        changed = deepcopy(samples)
        for sample in changed:
            if sample["task_group"] == "family-0":
                sample["label"] = "failure" if sample["label"] == "success" else "success"
        # Row order in the feature file must not accidentally align labels.
        second = probe.cross_validate(changed, {"rows": list(reversed(features["rows"]))})
        for method in first["predictions"]:
            a = [p for p in first["predictions"][method] if p["held_out_group"] == "family-0"]
            b = [p for p in second["predictions"][method] if p["held_out_group"] == "family-0"]
            self.assertEqual([(p["sample_id"], p["prediction"]) for p in a],
                             [(p["sample_id"], p["prediction"]) for p in b])
            np.testing.assert_array_equal([p["score_success"] for p in a], [p["score_success"] for p in b])

    def test_abstention_metrics_match_independent_sklearn_definitions(self):
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        truth = ["success", "success", "failure", "failure", "failure", "success"]
        samples = [{"sample_id": str(i), "task_group": f"family-{i // 2}", "label": label}
                   for i, label in enumerate(truth)]
        predictions = [{"sample_id": "0", "prediction": "success", "status": "ok"},
                       {"sample_id": "1", "prediction": "failure", "status": "ok"},
                       {"sample_id": "2", "prediction": "failure", "status": "ok"},
                       {"sample_id": "3", "prediction": "failure", "status": "failed"},
                       {"sample_id": "4", "prediction": "abstain", "status": "ok"}]
        predicted = ["success", "failure", "failure", "abstain", "abstain", "abstain"]
        result = evaluate(samples, predictions, bootstrap_iterations=0)
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["coverage"], .5)
        self.assertAlmostEqual(result["accuracy"], accuracy_score(truth, predicted))
        self.assertAlmostEqual(result["macro_f1"], f1_score(truth, predicted, labels=["success", "failure"], average="macro", zero_division=0))
        self.assertAlmostEqual(result["balanced_accuracy"], recall_score(truth, predicted, labels=["success", "failure"], average="macro", zero_division=0))


class SensitivityPlanTests(unittest.TestCase):
    def test_fit_requires_a_separately_saved_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(sensitivity, "cross_validate") as fit:
                with self.assertRaisesRegex(ValueError, "separate plan command"):
                    sensitivity.execute_plan(output, output / "features", output / "cv", output)
                fit.assert_not_called()

    def test_plan_is_repeatable_and_rejects_plan_or_source_snapshot_changes(self):
        samples = [{"sample_id": member, "task_group": member}
                   for members in sensitivity.GROUPS.values() for member in members]
        self.assertEqual(len({s["task_group"] for s in sensitivity.assign_groups(samples)}), 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "fixture.py"
            source.write_text("# frozen source\n")
            hashes = {"inputs_sha256": "a", "labels_sha256": "b", "features_sha256": "c", "reference_cv_sha256": "d"}
            with patch.object(sensitivity, "_inputs", return_value=(samples, {"identity": {"dimension": 3}}, {}, hashes)), patch.object(
                    sensitivity, "_runtime_sources", return_value={"fixture.py": source}), patch.object(sensitivity, "cross_validate") as fit:
                first = sensitivity.make_plan(output, source, source, output)
                again = sensitivity.make_plan(output, source, source, output)
                self.assertEqual(first, again)
                self.assertTrue(first["post_hoc"])
                self.assertEqual(first["methods"], list(sensitivity.METHODS))
                self.assertEqual(first["hyperparameters"], sensitivity.FIXED_CONFIG)
                self.assertEqual(first["plan_sha256"], sensitivity.digest({k: v for k, v in first.items() if k != "plan_sha256"}))
                changed = deepcopy(first)
                changed["post_hoc"] = False
                (output / "plan.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "plan changed"):
                    sensitivity.make_plan(output, source, source, output)
                (output / "plan.json").write_text(json.dumps(first))
                (output / "runtime" / "fixture.py").write_text("# corrupted snapshot\n")
                with self.assertRaisesRegex(ValueError, "Runtime snapshot differs"):
                    sensitivity.make_plan(output, source, source, output)
                fit.assert_not_called()


class SensitivityBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main, self.output, self.bundle = (self.root / name for name in ("main", "run", "portable"))
        self.main.mkdir()
        self.output.mkdir()
        self.samples = []
        for index, name in enumerate(member for members in sensitivity.GROUPS.values() for member in members):
            self.samples.append({"sample_id": str(index), "task_group": name, "episode_group": f"episode-{index}",
                "label": "success" if index % 2 else "failure", "failure_category": "fixture",
                "task_instruction": "fixture task", "subtask_instruction": f"subtask {index}",
                "frames": [{"phase": phase, "view": str(view), "sha256": hashlib.sha256(f"{index}-{phase}-{view}".encode()).hexdigest()}
                           for phase in ("before", "after") for view in range(3)]})
        inputs = [{k: s[k] for k in ("sample_id", "task_instruction", "subtask_instruction", "frames")} for s in self.samples]
        labels = [{k: s[k] for k in ("sample_id", "task_group", "episode_group", "label", "failure_category")} for s in self.samples]
        for name, rows in (("source-inputs.jsonl", inputs), ("source-labels.jsonl", labels)):
            (self.main / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        features = {"identity": {"dimension": 3}, "input_sha256": sensitivity.digest(
            [{k: s[k] for k in ("sample_id", "frames", "subtask_instruction")} for s in self.samples]),
            "rows": [{"sample_id": s["sample_id"], "before": [1, 2, 3], "after": [2, 1, 3], "text": [3, 2, 1]} for s in self.samples]}
        sensitivity._atomic_json(self.main / "features.json", features)
        module_root = Path(sensitivity.__file__).parent
        runtime = {}
        for name in ("probe.py", "models.py", "metrics.py"):
            target = self.output / "runtime" / "robotbench" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((module_root / name).read_bytes())
            runtime["robotbench/" + name] = sensitivity._sha256(target)
        primary = self.make_cv(self.samples)
        primary.update(inputs_sha256=sensitivity._sha256(self.main / "source-inputs.jsonl"),
                       labels_sha256=sensitivity._sha256(self.main / "source-labels.jsonl"),
                       features_sha256=sensitivity._sha256(self.main / "features.json"))
        sensitivity._atomic_json(self.main / "cv.json", primary)
        sources = {key: sensitivity._sha256(self.main / name) for name, key in sensitivity.MAIN_FILES.items()}
        coarse = sensitivity.assign_groups(self.samples)
        plan = {"post_hoc": True, "created_at": "2026-01-01T00:00:00+00:00", "groups": sensitivity.GROUPS,
                "group_counts": dict(Counter(s["task_group"] for s in coarse)), "samples": len(self.samples),
                "sample_assignments": [{"sample_id": s["sample_id"], "original_task_group": s["original_task_group"], "coarse_group": s["task_group"]} for s in coarse],
                "sources": sources, "runtime_source_sha256": runtime, "feature_identity": features["identity"],
                "hyperparameters": sensitivity.FIXED_CONFIG, "methods": list(sensitivity.METHODS),
                "scoring": {"bootstrap_iterations": 20, "bootstrap_seed": 0, "confidence": .95}, "limits": ["Synthetic offline fixture"]}
        plan["plan_sha256"] = sensitivity.digest(plan)
        sensitivity._atomic_json(self.output / "plan.json", plan)
        cv = self.make_cv(coarse)
        cv.update(plan_sha256=plan["plan_sha256"], sources=sources)
        sensitivity._atomic_json(self.output / "cv.json", cv)
        evaluation = sensitivity._evaluation(plan, self.samples, coarse, primary, cv, sensitivity._sha256(self.output / "cv.json"))
        sensitivity._atomic_json(self.output / "evaluation.json", evaluation)
        sensitivity.export_bundle(self.output, self.bundle)

    def make_cv(self, samples):
        folds, predictions = [], {m: [] for m in sensitivity.METHODS}
        for group in sorted({s["task_group"] for s in samples}):
            train = [s for s in samples if s["task_group"] != group]
            test = [s for s in samples if s["task_group"] == group]
            folds.append({"test_task_group": group, "train_sample_ids": [s["sample_id"] for s in train],
                          "test_sample_ids": [s["sample_id"] for s in test], "purged_shared_image_ids": [],
                          "train_class_counts": dict(Counter(s["label"] for s in train)), "warnings": {}})
            for method in predictions:
                predictions[method].extend({"sample_id": s["sample_id"], "prediction": s["label"], "status": "ok",
                    "method": method, "held_out_group": group, "fold_train_n": len(train)} for s in test)
        return {"folds": folds, "predictions": predictions, "config": sensitivity.FIXED_CONFIG,
                "code_sha256": sensitivity._sha256(Path(probe.__file__)),
                "models_code_sha256": sensitivity._sha256(Path(probe.__file__).with_name("models.py"))}

    def rehash_envelope(self, *, plan_sha256=None):
        path = self.bundle / sensitivity.MANIFEST
        manifest = json.loads(path.read_text())
        for relative in manifest["files"]:
            manifest["files"][relative] = sensitivity._sha256(self.bundle / relative)
        if plan_sha256:
            manifest["plan_sha256"] = plan_sha256
        sensitivity._atomic_json(path, manifest)
        (self.bundle / sensitivity.MANIFEST_CHECKSUM).write_text(sensitivity._sha256(path) + "\n")

    def test_recompute_is_portable_has_no_fit_and_requires_exact_source_bytes(self):
        with patch.object(sensitivity, "cross_validate", side_effect=AssertionError("fit forbidden")):
            result = sensitivity.verify_bundle(self.bundle, self.main)
        self.assertTrue(result["verified"])
        self.assertEqual(result["samples"], 16)
        self.assertEqual(result["fit_calls"], 0)
        with (self.main / "source-labels.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "source checksum"):
            sensitivity.verify_bundle(self.bundle, self.main)

    def test_rehashed_missing_prediction_and_changed_fold_still_fail(self):
        path = self.bundle / "cv.json"
        original = path.read_bytes()
        result = json.loads(original)
        result["predictions"]["after"].pop()
        sensitivity._atomic_json(path, result)
        self.rehash_envelope()
        with self.assertRaisesRegex(ValueError, "every sample ID exactly once"):
            sensitivity.verify_bundle(self.bundle, self.main)
        path.write_bytes(original)
        result = json.loads(original)
        result["folds"][0]["train_sample_ids"].pop()
        sensitivity._atomic_json(path, result)
        self.rehash_envelope()
        with self.assertRaisesRegex(ValueError, "source-derived split"):
            sensitivity.verify_bundle(self.bundle, self.main)

    def test_rehashed_wrong_metric_is_recomputed_and_rejected(self):
        path = self.bundle / "evaluation.json"
        value = json.loads(path.read_text())
        value["metrics"]["after"]["accuracy"] = .12345
        sensitivity._atomic_json(path, value)
        self.rehash_envelope()
        with self.assertRaisesRegex(ValueError, "Recomputed metrics"):
            sensitivity.verify_bundle(self.bundle, self.main)

    def test_self_consistent_untrusted_runtime_code_is_never_executed(self):
        metric_path = self.bundle / "runtime" / "robotbench" / "metrics.py"
        sentinel = self.root / "UNTRUSTED_CODE_EXECUTED"
        with metric_path.open("a") as handle:
            handle.write(f"\nfrom pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n")
        plan_path, cv_path, evaluation_path = (self.bundle / name for name in ("plan.json", "cv.json", "evaluation.json"))
        plan = json.loads(plan_path.read_text())
        plan["runtime_source_sha256"]["robotbench/metrics.py"] = sensitivity._sha256(metric_path)
        plan["plan_sha256"] = sensitivity.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        sensitivity._atomic_json(plan_path, plan)
        cv = json.loads(cv_path.read_text())
        cv["plan_sha256"] = plan["plan_sha256"]
        sensitivity._atomic_json(cv_path, cv)
        evaluation = json.loads(evaluation_path.read_text())
        evaluation.update(plan_sha256=plan["plan_sha256"], cv_sha256=sensitivity._sha256(cv_path))
        sensitivity._atomic_json(evaluation_path, evaluation)
        self.rehash_envelope(plan_sha256=plan["plan_sha256"])
        with self.assertRaisesRegex(ValueError, "Trusted evaluator version"):
            sensitivity.verify_bundle(self.bundle, self.main)
        self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
