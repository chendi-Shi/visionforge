from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from robotbench import temporal_evaluation as evaluation
from robotbench import temporal_report as report


def make_fixture(root):
    def put(name, value=None, content=None):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if content is not None else report.json_bytes(value))
        return path
    labels, selection, inputs = [], [], []
    for i in range(19):
        identifier, task, split = f"ft-{i:020x}", f"task{i//2}", "test" if i < 16 else "development"
        label = "failure" if i % 2 == 0 else "success"
        onset = 5.5 if label == "failure" else None
        annotation = {"verdict": "fail" if label == "failure" else "success", "type": ["grasp"] if label == "failure" else []}
        if onset is not None:
            annotation["onset"] = onset
        selection.append({"sample_id": identifier, "task_id": task, "split": split, "annotation": annotation})
        labels.append({"sample_id": identifier, "task_id": task, "split": split, "label": label,
                       "onset_seconds": onset, "failure_types": annotation["type"]})
        if i >= 16:
            continue
        inputs.append({"sample_id": identifier, "task_id": task, "robot": "synthetic", "split": split,
                       "source_sha256": hashlib.sha256(identifier.encode()).hexdigest(), "task_description": "Move object <script>bad()</script>",
                       "substages": ["grasp", "place"], "timestamp": [float(i) for i in range(12)], "state_eef": [[0.] * 16 for _ in range(12)],
                       "head_view": "head", "head_video_sha256": hashlib.sha256((identifier + 'source').encode()).hexdigest(), "fps": 1., "duration_seconds": 12.})
    put("dataset_snapshot/plan.json", {"samples": selection})
    put("dataset_snapshot/manifest.json", {"complete": True, "samples_prepared": 19, "samples_planned": 19})
    put("dataset_snapshot/labels.jsonl", content=b"".join((json.dumps(row) + "\n").encode() for row in labels))
    protocol = {"version": 2, "methods": list(report.METHODS), "frame_budget": 12, "model_options": {"temperature": 0},
                "system": "frozen system", "schema": {"type": "object"}}
    # Deliberately executable-looking snapshot content is never imported.
    for name, key in report.PREDICTION_SOURCES.items():
        path = put("prediction_sources/" + name, content=b"raise RuntimeError('must never execute frozen inference code')\n")
        protocol[key] = report.sha(path)
    identities = []
    for row in inputs:
        identity = {k: row[k] for k in ("sample_id", "task_id", "robot", "source_sha256", "split")}
        identity["prediction_input_sha256"] = report.digest({k: row[k] for k in ("task_description", "substages", "timestamp", "state_eef", "head_view", "head_video_sha256")})
        identities.append(identity)
    plan = {"split": "test", "model": "fixture:4b", "model_digest": "a" * 64, "protocol": protocol, "inputs": identities}
    for name, key in report.SNAPSHOTS.items():
        plan[key] = report.sha(root / "dataset_snapshot" / name)
    plan["plan_sha256"] = report.digest(plan)
    put("plan.json", plan)
    records = []
    for episode in inputs:
        for method in report.METHODS:
            identifier = episode["sample_id"]
            stem = identifier + "--" + method
            panel = put("panels/" + stem + ".jpg", content=b"\xff\xd8\xfffixture-original-panel" + stem.encode())
            raw = {"decision": "failure_at_frame_06", "reason": "Visible object dropped"}
            record = {k: episode[k] for k in ("sample_id", "task_id", "robot", "split", "source_sha256")}
            record.update(method=method, plan_sha256=plan["plan_sha256"], model=plan["model"], model_digest=plan["model_digest"],
                          options=protocol["model_options"], system=protocol["system"], schema=protocol["schema"],
                          selection={"method": method, "indices": list(range(12)), "timestamps": episode["timestamp"], "scores": [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0], "budget": 12, "event_indices": []},
                          duration_seconds=11., panel_file="panels/" + stem + ".jpg", panel_sha256=report.sha(panel),
                          question=json.dumps({"task": episode["task_description"], "ordered_steps": episode["substages"], "frame_times_seconds": episode["timestamp"]}),
                          status="ok", decision=raw["decision"], prediction="failure", frame_id=6, onset_seconds=6., reason=raw["reason"], raw_response=json.dumps(raw), done_reason="stop")
            put("predictions/" + stem + ".json", record)
            records.append(record)
    media = {}
    for episode in inputs:
        identifier = episode["sample_id"]
        video = put("videos/" + identifier + ".mp4", content=b"fixture-review-video" + identifier.encode())
        receipt = {"source_sha256": episode["head_video_sha256"], "review_sha256": report.sha(video), "review_bytes": video.stat().st_size}
        put("videos/" + identifier + ".receipt.json", receipt)
        media[identifier] = receipt
    selected_labels = [r for r in labels if r["split"] == "test"]
    result = evaluation.evaluate_all(selected_labels, records, inputs)
    samples = report.sample_rows(inputs, selected_labels, {r["sample_id"]: "videos/" + r["sample_id"] + ".mp4" for r in inputs})
    put("input-evidence.json", inputs)
    put("media-manifest.json", media)
    put("samples.json", samples)
    put("evaluation.json", result)
    put("report.html", content=report.render_html(plan, samples, records, result).encode())
    manifest = {"plan_sha256": plan["plan_sha256"], "files": {}, "verifier_source_sha256": {
        "temporal_report.py": report.sha(Path(report.__file__)), "temporal_evaluation.py": report.sha(Path(evaluation.__file__))}}
    put("bundle-manifest.json", manifest)
    rehash(root)
    return plan, inputs, records


def rehash(root):
    manifest = report.read(root / "bundle-manifest.json")
    manifest["files"] = {p.relative_to(root).as_posix(): report.sha(p) for p in root.rglob("*") if p.is_file() and p.name not in {"bundle-manifest.json", "bundle-manifest.sha256"}}
    (root / "bundle-manifest.json").write_bytes(report.json_bytes(manifest))
    (root / "bundle-manifest.sha256").write_text(report.sha(root / "bundle-manifest.json") + "\n")


class TemporalReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.plan, cls.inputs, cls.records = make_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        import shutil
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temporary.name) / "bundle"
        shutil.copytree(self.root, self.bundle)
        self.addCleanup(self.temporary.cleanup)

    def test_complete_bundle_recomputes_without_importing_prediction_sources(self):
        checked = report.verify_bundle(self.bundle)
        self.assertTrue(checked["verified"])
        self.assertEqual(checked["samples"], 16)
        self.assertEqual(checked["prediction_records"], 32)
        self.assertEqual(checked["inference_calls"], 0)

    def test_removed_prediction_rejected_even_after_manifest_rehash(self):
        next((self.bundle / "predictions").glob("*.json")).unlink()
        rehash(self.bundle)
        with self.assertRaisesRegex(ValueError, "every episode"):
            report.verify_bundle(self.bundle)

    def test_raw_response_cannot_disagree_with_stored_verdict_after_rehash(self):
        path = next((self.bundle / "predictions").glob("*.json"))
        record = report.read(path)
        record["raw_response"] = json.dumps({"decision": "success", "reason": "Changed raw response"})
        path.write_bytes(report.json_bytes(record))
        rehash(self.bundle)
        with self.assertRaisesRegex(ValueError, "raw model response"):
            report.verify_bundle(self.bundle)

    def test_wrong_report_metric_rejected_after_manifest_rehash(self):
        path = self.bundle / "evaluation.json"
        result = report.read(path)
        result["methods"]["uniform"]["hit_at_1s"] = .123
        path.write_bytes(report.json_bytes(result))
        rehash(self.bundle)
        with self.assertRaisesRegex(ValueError, "Recomputed metrics"):
            report.verify_bundle(self.bundle)

    def test_wrong_public_ground_truth_rejected_after_manifest_rehash(self):
        path = self.bundle / "samples.json"
        samples = report.read(path)
        samples[0]["label"] = "success"
        path.write_bytes(report.json_bytes(samples))
        rehash(self.bundle)
        with self.assertRaisesRegex(ValueError, "sample labels"):
            report.verify_bundle(self.bundle)

    def test_annotation_snapshot_must_agree_with_source_annotation(self):
        path = self.bundle / "dataset_snapshot" / "labels.jsonl"
        labels = report.rows(path)
        labels[0]["onset_seconds"] = 1.
        path.write_text("".join(json.dumps(r) + "\n" for r in labels))
        plan = report.read(self.bundle / "plan.json")
        plan["labels_sha256"] = report.sha(path)
        plan["plan_sha256"] = report.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        (self.bundle / "plan.json").write_bytes(report.json_bytes(plan))
        with self.assertRaisesRegex(ValueError, "source author"):
            report.load_frozen_labels(self.bundle)

    def test_numeric_input_tamper_and_annotation_fields_rejected(self):
        inputs = deepcopy(self.inputs)
        inputs[0]["timestamp"][2] += .1
        with self.assertRaisesRegex(ValueError, "frozen identity"):
            report.validate_inputs(self.plan, inputs)
        inputs = deepcopy(self.inputs)
        inputs[0]["label"] = "failure"
        with self.assertRaisesRegex(ValueError, "annotation-bearing"):
            report.validate_inputs(self.plan, inputs)

    def test_version_one_and_two_are_checked_without_execution(self):
        plan = deepcopy(self.plan)
        plan["protocol"]["version"] = 1
        record = deepcopy(self.records[0])
        record["raw_response"] = json.dumps({"prediction": "failure", "frame_id": 6, "reason": record["reason"]})
        report.validate_record(record, self.inputs[0], plan, self.bundle)
        record["raw_response"] = json.dumps({"prediction": "success", "frame_id": 6, "reason": "wrong schema consistency"})
        with self.assertRaisesRegex(ValueError, "Invalid raw verdict"):
            report.validate_record(record, self.inputs[0], plan, self.bundle)

    def test_html_escapes_task_text_and_keeps_video_navigation_and_raw_links(self):
        document = (self.bundle / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>bad()</script>", document)
        self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", document)
        self.assertIn('data-video="v-ft-00000000000000000000"', document)
        self.assertIn('data-seek="6.000000000"', document)
        self.assertIn('predictions/ft-00000000000000000000--uniform.json', document)

    def test_absolute_paths_and_escaped_artifact_paths_are_rejected(self):
        for value in (r"C:\Users\name\private.mp4", "file:///secret", "error: C:/Users/name/secret"):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                report.reject_local_paths({"nested": [value]})
        report.reject_local_paths("https://huggingface.co/datasets/ChangUoA/FailTime")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            report.safe(self.bundle, "../outside.json")

    def test_verifier_import_is_standard_library_only(self):
        code = "import sys,json; import robotbench.temporal_report; print(json.dumps(sorted(set(sys.modules)&{'numpy','h5py','ollama','cv2','PIL','scipy'})))"
        result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])

    def test_atomic_export_and_standalone_verifier_work_outside_repository(self):
        original = Path(self.temporary.name) / "original-videos"
        original.mkdir()
        actual = []
        for row in self.inputs:
            path = original / (row["sample_id"] + ".mp4")
            path.write_bytes((row["sample_id"] + "source").encode())
            actual.append({**{k: v for k, v in row.items() if k != "head_video_sha256"},
                           "videos": {"head": str(path)}, "video_sha256": {"head": report.sha(path)}})
        def transcode(source, destination, expected):
            self.assertEqual(report.sha(source), expected)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"test-preview" + source.read_bytes())
            receipt = {"source_sha256": expected, "review_sha256": report.sha(destination), "review_bytes": destination.stat().st_size}
            destination.with_suffix(".receipt.json").write_bytes(report.json_bytes(receipt))
            return receipt
        output = Path(self.temporary.name) / "portable"
        with patch("robotbench.temporal_data.load_inputs", return_value=actual), patch.object(report, "_transcode", side_effect=transcode):
            result = report.export_bundle(Path("unused-test-data"), self.bundle, output)
        self.assertTrue(result["verified"])
        self.assertFalse(output.with_name(".portable.staging").exists())
        completed = subprocess.run([sys.executable, "-B", str(output / "verify.py")], cwd=self.temporary.name,
                                   text=True, capture_output=True, check=True)
        self.assertTrue(json.loads(completed.stdout)["verified"])
        for record in self.records:
            name = record["panel_file"]
            self.assertEqual((self.bundle / name).read_bytes(), (output / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
