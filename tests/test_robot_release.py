"""Transactional publication and independent raw/source consistency checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import zipfile

import tests.test_robot_report as fixtures
from robotbench.data import DATASET_ID, model_input, sha256_file
from robotbench.metrics import evaluate, compare_paired
from robotbench.models import SYSTEM
from robotbench.run import request_identity
from tools.export_robot_run import export
from tools.verify_robot_release import verify


class RobotReleaseTests(unittest.TestCase):
    def setUp(self):
        fixtures.RobotReportTests.setUp(self)
        self.repository = Path(__file__).resolve().parents[1]
        self.probe = Path(self.temp.name) / "probe"
        self.probe.mkdir()
        self.destination = Path(self.temp.name) / "published"
        # CI resampling has separate tests. These tests exercise complete export
        # and recomputation while keeping integrity failure checks inexpensive.
        for namespace in ("robotbench.run", "tools.verify_robot_release"):
            self.enterContext(patch(namespace + ".evaluate", side_effect=lambda s, p: evaluate(s, p, bootstrap_iterations=0)))
            self.enterContext(patch(namespace + ".compare_paired", side_effect=lambda s, a, b: compare_paired(s, a, b, bootstrap_iterations=0)))
        inputs = [{key: sample[key] for key in ("sample_id", "task_instruction", "subtask_instruction", "frames")} for sample in self.samples]
        labels = [{key: sample[key] for key in ("sample_id", "task_group", "episode_group", "label", "failure_category")} for sample in self.samples]
        for filename, rows in (("inputs.jsonl", inputs), ("labels.jsonl", labels)):
            (self.root / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(dataset=DATASET_ID, files={name: sha256_file(self.root / name) for name in ("inputs.jsonl", "labels.jsonl")})
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.plan.update(inputs_sha256=sha256_file(self.root / "inputs.jsonl"),
                         system_prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(), code_sha256={})
        (self.output / "protocol_sources").mkdir()
        for relative in ("robotbench/run.py", "robotbench/models.py", "robotbench/data.py", "core.py"):
            source = self.repository / relative
            shutil.copyfile(source, self.output / "protocol_sources" / source.name)
            self.plan["code_sha256"][source.name] = sha256_file(source)
        (self.output / "vlm-plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        features = {"input_sha256": digest([{key: sample[key] for key in ("sample_id", "frames", "subtask_instruction")} for sample in self.samples]),
                    "rows": [{"sample_id": sample["sample_id"], "before": [1, 0], "after": [0, 1], "text": [1, 0]} for sample in self.samples]}
        (self.probe / "features.json").write_text(json.dumps(features), encoding="utf-8")
        self.cv = {"inputs_sha256": sha256_file(self.root / "inputs.jsonl"), "labels_sha256": sha256_file(self.root / "labels.jsonl"),
                   "features_sha256": sha256_file(self.probe / "features.json"), "code_sha256": sha256_file(self.repository / "robotbench/probe.py"),
                   "models_code_sha256": sha256_file(self.repository / "robotbench/models.py"),
                   "predictions": {method: [{"sample_id": s["sample_id"], "prediction": s["label"], "status": "ok"} for s in self.samples]
                                   for method in ("train_majority", "text_only", "after", "transition")}}
        (self.probe / "cv.json").write_text(json.dumps(self.cv), encoding="utf-8")
        for sample in self.samples:
            for condition in self.plan["conditions"]:
                inputs = model_input(sample, self.root, condition=condition, view="all")
                answer = {"prediction": sample["label"], "evidence": "Synthetic fixture only; no model inference."}
                row = {"sample_id": sample["sample_id"], "condition": condition, **answer, "status": "ok",
                       "model": self.plan["model"], "digest": self.plan["digest"], "system_prompt": SYSTEM,
                       "options": {"num_ctx": 4096, "num_predict": 160, "num_thread": 4, "temperature": 0, "seed": 0},
                       "raw_response": json.dumps(answer), "elapsed_seconds": 0, "done_reason": "stop",
                       "question": json.dumps({"overall_task": inputs["task"], "subtask_to_evaluate": inputs["subtask"]}),
                       "input": {"frames": [{key: frame[key] for key in ("phase", "view", "sha256")} for frame in inputs["frames"]]},
                       "request_sha256": request_identity(self.plan, inputs, condition)}
                (self.output / "predictions" / f"{sample['sample_id']}.{condition}.json").write_text(json.dumps(row), encoding="utf-8")

    def publish(self):
        return export(self.root, self.output, self.probe, self.destination)

    def resign(self):
        """Update outer hashes to test semantic checks beyond checksum matching."""
        path = self.destination / "release-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"] = [{"path": p.relative_to(self.destination).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                             for p in sorted(self.destination.rglob("*")) if p.is_file() and p.name not in {"release-manifest.json", "robot-benchmark.zip"}]
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_export_verifies_offline_and_preserves_source_run(self):
        before = {p.relative_to(self.output).as_posix(): sha256_file(p) for p in self.output.rglob("*") if p.is_file()}
        result = self.publish()
        self.assertEqual(result["verification"]["model_calls"], 0)
        self.assertEqual(verify(self.destination)["cv_samples"], 2)
        self.assertEqual(before, {p.relative_to(self.output).as_posix(): sha256_file(p) for p in self.output.rglob("*") if p.is_file()})
        self.assertEqual(sha256_file(self.destination / "source-labels.jsonl"), sha256_file(self.root / "labels.jsonl"))
        self.assertTrue((self.destination / "LICENSE-DATA.txt").is_file())
        with zipfile.ZipFile(self.destination / "robot-benchmark.zip") as bundle:
            self.assertIn("release-manifest.json", bundle.namelist())
            self.assertNotIn("robot-benchmark.zip", bundle.namelist())

    def test_failed_copy_leaves_no_destination_and_same_path_retry_succeeds(self):
        original = shutil.copyfile
        def fail_prediction(source, destination, *args, **kwargs):
            if Path(source).parent.name == "predictions":
                raise OSError("simulated interrupted copy")
            return original(source, destination, *args, **kwargs)
        with patch("tools.export_robot_run.shutil.copyfile", side_effect=fail_prediction):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.publish()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(Path(self.temp.name).glob(".rb-*")), [])
        self.publish()
        self.assertTrue(self.destination.is_dir())

    def test_missing_protocol_source_never_silently_passes(self):
        self.publish()
        (self.destination / "protocol_sources" / "models.py").unlink()
        self.resign()
        with self.assertRaisesRegex(ValueError, "source missing"):
            verify(self.destination)

    def test_unlisted_and_duplicate_manifest_paths_are_rejected(self):
        self.publish()
        extra = self.destination / "unlisted.txt"
        extra.write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(self.destination)
        extra.unlink()
        path = self.destination / "release-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"].append(deepcopy(manifest["files"][0]))
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify(self.destination)

    def test_source_groundtruth_and_raw_prediction_must_match_derived_values(self):
        self.publish()
        path = self.destination / "evaluation-samples.json"
        samples = json.loads(path.read_text(encoding="utf-8"))
        samples[0]["label"] = "failure"
        path.write_text(json.dumps(samples), encoding="utf-8")
        self.resign()
        with self.assertRaisesRegex(ValueError, "ground truth"):
            verify(self.destination)
        samples[0]["label"] = "success"
        path.write_text(json.dumps(samples), encoding="utf-8")
        prediction = self.destination / "predictions" / "sample-0.ordered.json"
        row = json.loads(prediction.read_text(encoding="utf-8"))
        row["raw_response"] = json.dumps({"prediction": "failure", "evidence": row["evidence"]})
        prediction.write_text(json.dumps(row), encoding="utf-8")
        self.resign()
        with self.assertRaisesRegex(ValueError, "raw response"):
            verify(self.destination)

    def test_private_path_refuses_export_and_incomplete_run_is_not_published(self):
        path = self.output / "predictions" / "sample-0.ordered.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["error"] = "Local path C:\\Users\\private\\secret.txt"
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Local filesystem path"):
            self.publish()
        self.assertFalse(self.destination.exists())
        path.unlink()
        with self.assertRaisesRegex(ValueError, "complete prespecified"):
            self.publish()
        self.assertFalse(self.destination.exists())

    def test_failed_and_explicit_abstention_keep_denominators_and_three_state_flips(self):
        for condition, status in (("after_only", "failed"), ("ordered", "ok")):
            path = self.output / "predictions" / f"sample-0.{condition}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            row.update(prediction="abstain", status=status)
            row["raw_response"] = "truncated invalid reply" if status == "failed" else json.dumps({"prediction": "abstain", "evidence": row["evidence"]})
            path.write_text(json.dumps(row), encoding="utf-8")
        self.publish()
        report = json.loads((self.destination / "metrics.json").read_text(encoding="utf-8"))
        for condition in ("after_only", "ordered"):
            self.assertEqual(report["vlm"]["metrics"][condition]["n"], 2)
            self.assertEqual(report["vlm"]["metrics"][condition]["coverage"], .5)
        self.assertEqual(report["vlm"]["order_sensitivity"], {"paired_valid_outputs": 2, "prediction_flips": 1, "flip_rate": .5})
        self.assertEqual(report["vlm"]["successful_calls"], 5)
        self.assertEqual(verify(self.destination)["vlm_samples"], 2)

    def test_clone_without_zip_is_verifiable_but_existing_or_required_zip_is_checked(self):
        self.publish()
        archive = self.destination / "robot-benchmark.zip"
        with zipfile.ZipFile(archive, "a") as bundle:
            bundle.writestr("unlisted.txt", "not in manifest")
        with self.assertRaisesRegex(ValueError, "Archive inventory"):
            verify(self.destination)
        archive.unlink()
        self.assertEqual(verify(self.destination)["model_calls"], 0)
        with self.assertRaisesRegex(ValueError, "archive is missing"):
            verify(self.destination, require_archive=True)

    def test_derived_csv_duration_and_flip_counts_are_recomputed(self):
        self.publish()
        path = self.destination / "metrics.json"
        original = path.read_bytes()
        manifest_path = self.destination / "release-manifest.json"
        original_manifest = manifest_path.read_bytes()
        for key, value, message in (("inference_seconds", 999, "Inference duration"),
                                    ("order_sensitivity", {"paired_valid_outputs": 0, "prediction_flips": 0, "flip_rate": None}, "Order-sensitivity")):
            with self.subTest(key=key):
                report = json.loads(original)
                report["vlm"][key] = value
                path.write_text(json.dumps(report), encoding="utf-8")
                self.resign()
                with self.assertRaisesRegex(ValueError, message):
                    verify(self.destination)
                path.write_bytes(original)
                manifest_path.write_bytes(original_manifest)
        import csv
        path = self.destination / "metrics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["failure_recall"] = "0"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.resign()
        with self.assertRaisesRegex(ValueError, "CSV values"):
            verify(self.destination)


if __name__ == "__main__":
    unittest.main()
