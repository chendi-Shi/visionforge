"""Study identity checks use synthetic bytes; no model or numeric runtime."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from robotbench import temporal_comparison as comparison
from robotbench import temporal_evaluation as evaluation
from robotbench import temporal_report as report
from tests.test_temporal_report import make_fixture, rehash


def write(path, value):
    path.write_bytes(report.json_bytes(value))


def study_fixture(root):
    mini, qwen = root / "mini", root / "qwen"
    make_fixture(mini)
    shutil.copytree(mini, qwen)
    selection = report.read(mini / "dataset_snapshot" / "plan.json")
    selection["plan_sha256"] = report.digest(selection)
    official = {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": "sha256:" + "c" * 64, "size": 177},
                "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "d" * 64, "size": 1234}]}
    # Ollama may add a transport-only `from` attribute. Ordered layer identity
    # and exact local full-manifest byte digest must both still be checked.
    local = deepcopy(official)
    local["layers"][0]["from"] = "qwen3.5:4b"
    write(qwen / "model-manifest.json", local)
    models = ((comparison.MINI, mini, "a" * 64), (comparison.QWEN, qwen, report.sha(qwen / "model-manifest.json")))
    for model, bundle, digest in models:
        write(bundle / "dataset_snapshot" / "plan.json", selection)
        plan = report.read(bundle / "plan.json")
        plan.update(model=model, model_digest=digest, created_at="2026-10-15T06:10:00+00:00", server={"version": "fixture"},
                    dataset_selection_sha256=report.sha(bundle / "dataset_snapshot" / "plan.json"))
        plan["plan_sha256"] = report.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        write(bundle / "plan.json", plan)
        records = []
        for path in sorted((bundle / "predictions").glob("*.json")):
            record = report.read(path)
            record.update(model=model, model_digest=digest, plan_sha256=plan["plan_sha256"])
            write(path, record)
            records.append(record)
        # Predictions are unchanged, so the original full bootstrap evaluation
        # is retained rather than recomputing it during fixture construction.
        result = report.read(bundle / "evaluation.json")
        samples = report.read(bundle / "samples.json")
        (bundle / "report.html").write_text(report.render_html(plan, samples, records, result), encoding="utf-8")
        manifest = report.read(bundle / "bundle-manifest.json")
        manifest["plan_sha256"] = plan["plan_sha256"]
        write(bundle / "bundle-manifest.json", manifest)
        rehash(bundle)
    study = {"created_at": "2026-10-15T06:00:00+00:00", "expected_test_calls": 64,
             "test_sample_ids": [r["sample_id"] for r in selection["samples"] if r["split"] == "test"],
             "development_sample_ids": [r["sample_id"] for r in selection["samples"] if r["split"] == "development"],
             "selection_plan_sha256": selection["plan_sha256"], "protocol": plan["protocol"],
             "evaluation_source_sha256": report.sha(Path(evaluation.__file__)),
             "uncertainty": {"seed": 1337, "iterations": 2000}, "sampling_limits": ["Synthetic test cohort."],
             "models": [{"name": comparison.MINI, "digest": "a" * 64},
                        {"name": comparison.QWEN, "registry_manifest": official, "manifest_canonical_sha256": report.digest(official)}]}
    study["study_plan_sha256"] = report.digest(study)
    write(root / "study.json", study)
    return study


class TemporalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.study = study_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        shutil.copytree(self.root / "mini", self.work / "mini")
        shutil.copytree(self.root / "qwen", self.work / "qwen")
        shutil.copyfile(self.root / "study.json", self.work / "study.json")
        self.bundles = {comparison.MINI: self.work / "mini", comparison.QWEN: self.work / "qwen"}

    def test_identical_conditions_preserve_all_four_cells_and_zero_paired_differences(self):
        result = comparison.comparison(self.work / "study.json", *self.bundles.values())
        self.assertEqual(result["prediction_calls"], 64)
        self.assertEqual(len(result["four_model_sampler_conditions"]), 4)
        for row in result["four_model_sampler_conditions"]:
            self.assertEqual(row["metrics"]["n"], 16)
            self.assertEqual(row["metrics"]["failure_n"], 8)
        for pair in result["paired_minicpm_to_qwen"].values():
            self.assertEqual(pair["n"], 16)
            self.assertEqual(pair["delta"]["hit_at_2s"], 0.)
            self.assertEqual(pair["bootstrap_delta"]["iterations"], 2000)
            self.assertEqual(pair["bootstrap_delta"]["seed"], 1337)
        document = comparison.render_html(result)
        self.assertIn('../mini/report.html', document)
        self.assertIn('../qwen/report.html', document)
        self.assertNotIn(str(self.work), document)

    def test_changed_protocol_or_server_cannot_be_compared(self):
        path = self.work / "qwen" / "plan.json"
        original = report.read(path)
        for field, value, message in (("protocol", {**original["protocol"], "frame_budget": 11}, "protocol"),
                                      ("server", {"version": "different"}, "server")):
            plan = {**original, field: value}
            plan["plan_sha256"] = report.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            write(path, plan)
            for record_path in (path.parent / "predictions").glob("*.json"):
                record = report.read(record_path)
                record["plan_sha256"] = plan["plan_sha256"]
                write(record_path, record)
            with self.assertRaisesRegex(ValueError, message):
                comparison.validate_study(self.study, self.bundles)

    def test_qwen_raw_manifest_and_predeclared_ordered_layers_are_both_bound(self):
        path = self.work / "qwen" / "model-manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "full manifest"):
            comparison.validate_study(self.study, self.bundles)
        shutil.copyfile(self.root / "qwen" / "model-manifest.json", path)
        study = deepcopy(self.study)
        study["models"][1]["registry_manifest"]["layers"][0]["digest"] = "sha256:" + "e" * 64
        study["models"][1]["manifest_canonical_sha256"] = report.digest(study["models"][1]["registry_manifest"])
        study["study_plan_sha256"] = report.digest({k: v for k, v in study.items() if k != "study_plan_sha256"})
        with self.assertRaisesRegex(ValueError, "full manifest"):
            comparison.validate_study(study, self.bundles)

    def test_panel_bytes_must_be_identical_even_with_individually_valid_records(self):
        bundle = self.work / "qwen"
        record_path = next((bundle / "predictions").glob("*.json"))
        record = report.read(record_path)
        panel = bundle / record["panel_file"]
        panel.write_bytes(panel.read_bytes() + b"different input bytes")
        record["panel_sha256"] = report.sha(panel)
        write(record_path, record)
        with self.assertRaisesRegex(ValueError, "panel bytes"):
            comparison.validate_study(self.study, self.bundles)

    def test_frozen_study_membership_date_and_evaluator_are_enforced(self):
        cases = [("test_sample_ids", self.study["test_sample_ids"][:-1], "membership"),
                 ("created_at", "2026-10-16T00:00:00+00:00", "before"),
                 ("evaluation_source_sha256", "f" * 64, "evaluator")]
        for field, value, message in cases:
            study = {**self.study, field: value}
            study["study_plan_sha256"] = report.digest({k: v for k, v in study.items() if k != "study_plan_sha256"})
            with self.assertRaisesRegex(ValueError, message):
                comparison.validate_study(study, self.bundles)

    def test_standard_library_import_does_not_load_numeric_or_model_packages(self):
        code = "import sys,json; import robotbench.temporal_comparison; print(json.dumps(sorted(set(sys.modules)&{'numpy','h5py','ollama','cv2','PIL','scipy'})))"
        result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])

    def test_export_verifies_before_publish_and_standalone_detects_rehashed_metric_change(self):
        destination = self.work / "study"
        result = comparison.export_comparison(self.work / "study.json", *self.bundles.values(), destination)
        self.assertTrue(result["verified"])
        self.assertFalse((self.work / ".study.staging").exists())
        completed = subprocess.run([sys.executable, "-B", str(destination / "verify.py")], cwd=self.work,
                                   text=True, capture_output=True, check=True)
        self.assertTrue(json.loads(completed.stdout)["verified"])
        altered = report.read(destination / "comparison.json")
        altered["paired_minicpm_to_qwen"]["uniform"]["delta"]["hit_at_2s"] = 1.
        write(destination / "comparison.json", altered)
        rehash(destination)
        with self.assertRaisesRegex(ValueError, "Recomputed fixed-study"):
            comparison.verify_comparison(destination, *self.bundles.values())


if __name__ == "__main__":
    unittest.main()
