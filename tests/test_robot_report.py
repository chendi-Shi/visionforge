"""Portable reporting and cache identities, without contacting live Ollama."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from robotbench.data import model_input, REVISION, sha256_file
from robotbench.report import write_report
from robotbench.run import build_report, freeze_plan, predict_vlm, request_identity, select_pilot


class RobotReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "data"
        self.output = Path(self.temp.name) / "output"
        (self.root / "images").mkdir(parents=True)
        (self.output / "predictions").mkdir(parents=True)
        self.samples = []
        for index, label in enumerate(("success", "failure")):
            frames = []
            for phase_index, phase in enumerate(("before", "after")):
                for view in range(3):
                    buffer = io.BytesIO()
                    Image.new("RGB", (20, 20), (index * 90, phase_index * 100, view * 80)).save(buffer, format="PNG")
                    data = buffer.getvalue()
                    digest = hashlib.sha256(data).hexdigest()
                    relative = f"images/{digest}.png"
                    (self.root / relative).write_bytes(data)
                    frames.append({"frame_id": f"{phase}_{view}", "phase": phase, "view": str(view), "path": relative, "sha256": digest})
            self.samples.append({"sample_id": f"sample-{index}", "task_group": f"family-{index}", "episode_group": f"episode-{index}",
                "label": label, "failure_category": "grasp", "task_instruction": '<script>alert("task")</script>',
                "subtask_instruction": 'Pick up object <img src=x onerror=alert(1)>', "frames": frames})
        (self.root / "inputs.jsonl").write_text("fixed inference input manifest\n", encoding="utf-8")
        (self.root / "labels.jsonl").write_text("fixed evaluation labels\n", encoding="utf-8")
        manifest = {"samples": 2, "task_groups": 2, "episode_groups": 2, "success": 1, "failure": 1,
                    "source": "https://huggingface.co/datasets/paulpacaud/Guardian-FailCoT-OOD-datasets", "revision": REVISION}
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.plan = {"dataset_revision": REVISION, "sample_ids": [s["sample_id"] for s in self.samples],
                     "conditions": ["after_only", "ordered", "reversed"], "model": "local:vision", "digest": "weights-v1",
                     "view": "all", "max_tokens": 160, "inputs_sha256": sha256_file(self.root / "inputs.jsonl")}
        (self.output / "vlm-plan.json").write_text(json.dumps(self.plan), encoding="utf-8")

    def save_prediction(self, sample, condition_name, prediction, **extra):
        inputs = model_input(sample, self.root, condition=condition_name, view="all")
        row = {"sample_id": sample["sample_id"], "condition": condition_name, "prediction": prediction, "status": "ok",
               "model": self.plan["model"], "digest": self.plan["digest"], "elapsed_seconds": 1,
               "evidence": '<script>alert("answer")</script>', "raw_response": '<script>alert("raw")</script>',
               "request_sha256": request_identity(self.plan, inputs, condition_name)} | extra
        path = self.output / "predictions" / f"{sample['sample_id']}.{condition_name}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        return path

    def test_partial_report_keeps_all_samples_and_escapes_model_and_dataset_text(self):
        self.save_prediction(self.samples[0], "ordered", "success")
        self.save_prediction(self.samples[0], "after_only", "failure")
        report = build_report(self.samples, self.root, self.output)
        self.assertEqual(report["vlm"]["completed_calls"], 2)
        self.assertEqual(report["vlm"]["expected_calls"], 6)
        ordered = report["vlm"]["metrics"]["ordered"]
        self.assertEqual(ordered["n"], 2)
        self.assertEqual(ordered["accuracy"], .5)
        self.assertEqual(ordered["coverage"], .5)
        self.assertEqual(report["vlm"]["metrics"]["reversed"]["n"], 2)
        self.assertEqual(report["vlm"]["metrics"]["reversed"]["coverage"], 0)
        page = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('<script>alert(', page)
        self.assertNotIn('<img src=x onerror=', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertNotIn(str(self.root), page)
        self.assertIn("2 / 6", page)
        self.assertIn("pending", page)
        for sample in self.samples:
            for frame in sample["frames"]:
                exported = self.output / "assets" / (frame["sha256"] + ".png")
                self.assertEqual(sha256_file(exported), frame["sha256"])

    def test_mismatched_request_model_sample_or_condition_refuses_report(self):
        for changes in ({"request_sha256": "wrong"}, {"digest": "other-model"},
                        {"sample_id": "other-sample"}, {"condition": "reversed"}):
            with self.subTest(changes=changes):
                self.save_prediction(self.samples[0], "ordered", "success", **changes)
                with self.assertRaises(ValueError):
                    build_report(self.samples, self.root, self.output)
        self.save_prediction(self.samples[0], "ordered", "success")
        (self.root / "inputs.jsonl").write_text("changed inference inputs", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            build_report(self.samples, self.root, self.output)

    def test_report_refuses_tampered_exported_image(self):
        self.save_prediction(self.samples[0], "ordered", "success")
        report = build_report(self.samples, self.root, self.output)
        frame = self.samples[0]["frames"][0]
        (self.output / "assets" / (frame["sha256"] + ".png")).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            write_report(report, self.samples, self.root, self.output)

    def test_request_identity_uses_pixels_phase_goal_and_settings_but_not_local_paths(self):
        inputs = model_input(self.samples[0], self.root, condition="ordered", view="all")
        original = request_identity(self.plan, inputs, "ordered")
        moved = deepcopy(inputs)
        for frame in moved["frames"]:
            frame["path"] = "/portable/" + frame["sha256"] + ".png"
        self.assertEqual(request_identity(self.plan, moved, "ordered"), original)
        for changes in ("pixels", "phase", "goal", "settings"):
            altered, plan = deepcopy(inputs), deepcopy(self.plan)
            if changes == "pixels":
                altered["frames"][0]["sha256"] = "0" * 64
            elif changes == "phase":
                altered["frames"] = list(reversed(altered["frames"]))
            elif changes == "goal":
                altered["subtask"] = "Open gripper"
            else:
                plan["max_tokens"] = 128
            self.assertNotEqual(request_identity(plan, altered, "ordered"), original)

    def test_selection_ignores_labels_and_scores_and_freeze_rejects_changed_model(self):
        changed = deepcopy(self.samples)
        for sample in changed:
            sample["label"] = "failure" if sample["label"] == "success" else "success"
            sample["score"] = 0.999
        self.assertEqual([s["sample_id"] for s in select_pilot(self.samples)], [s["sample_id"] for s in select_pilot(changed)])
        fresh = self.output / "fresh-plan"
        fresh.mkdir()
        client = Mock()
        with patch("core._create_client", return_value=client), patch("core._inspect_model", return_value=("local:vision", "weights-v1", {})):
            plan, selected = freeze_plan(self.samples, self.root, fresh, "local:vision", "all", 160)
            again, _ = freeze_plan(self.samples, self.root, fresh, "local:vision", "all", 160)
        self.assertEqual(plan, again)
        self.assertEqual(len(selected), 2)
        with patch("core._create_client", return_value=client), patch("core._inspect_model", return_value=("local:vision", "weights-v2", {})):
            with self.assertRaisesRegex(ValueError, "Run identity changed"):
                freeze_plan(self.samples, self.root, fresh, "local:vision", "all", 160)

    def test_restart_reuses_saved_calls_and_request_mismatch_stops_before_inference(self):
        args = SimpleNamespace(model="local:vision", view="all", max_tokens=160)
        result = lambda *a, **k: {"status": "ok", "prediction": "success", "model": "local:vision", "digest": "weights-v1", "elapsed_seconds": 0}
        with patch("robotbench.run.freeze_plan", return_value=(self.plan, self.samples)), patch(
                "video_jobs.default_data_dir", return_value=Path(self.temp.name) / "resource-lock"), patch(
                "robotbench.models.judge_effect", side_effect=result) as judge, redirect_stdout(io.StringIO()):
            first = predict_vlm(self.samples, self.root, self.output, args)
            self.assertEqual(first["completed"], 6)
            self.assertEqual(judge.call_count, 6)
            judge.reset_mock()
            resumed = predict_vlm(self.samples, self.root, self.output, args)
            self.assertEqual(resumed["completed"], 6)
            judge.assert_not_called()
            path = self.output / "predictions" / "sample-0.after_only.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["request_sha256"] = "changed"
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed inputs"):
                predict_vlm(self.samples, self.root, self.output, args)
            judge.assert_not_called()

    def test_plan_binds_evaluation_labels_before_inference(self):
        fresh = self.output / "label-plan"
        fresh.mkdir()
        with patch("core._create_client", return_value=Mock()), patch("core._inspect_model", return_value=("local:vision", "weights-v1", {})):
            plan, _ = freeze_plan(self.samples, self.root, fresh, "local:vision", "all", 160)
            self.assertEqual(plan["labels_sha256"], sha256_file(self.root / "labels.jsonl"))
            (self.root / "labels.jsonl").write_text("different evaluation labels", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Run identity changed"):
                freeze_plan(self.samples, self.root, fresh, "local:vision", "all", 160)

    def test_resume_rejects_wrong_cached_metadata_before_calling_model(self):
        args = SimpleNamespace(model="local:vision", view="all", max_tokens=160)
        for changes in ({"sample_id": "other"}, {"condition": "ordered"}, {"digest": "changed"},
                        {"status": "unknown"}, {"status": "failed"}):
            with self.subTest(changes=changes):
                self.save_prediction(self.samples[0], "after_only", "success", **changes)
                with patch("robotbench.run.freeze_plan", return_value=(self.plan, self.samples)), patch(
                        "video_jobs.default_data_dir", return_value=Path(self.temp.name) / "resource-lock"), patch(
                        "robotbench.models.judge_effect") as judge, redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        predict_vlm(self.samples, self.root, self.output, args)
                    judge.assert_not_called()

    def test_report_renders_validated_snapshot_when_files_change_after_scoring(self):
        path = self.save_prediction(self.samples[0], "ordered", "success", evidence="validated evidence")
        def mutate_after_scoring(report, samples, root, output, **kwargs):
            row = json.loads(path.read_text(encoding="utf-8"))
            row.update(prediction="failure", evidence="unvalidated replacement")
            path.write_text(json.dumps(row), encoding="utf-8")
            write_report(report, samples, root, output, **kwargs)
        with patch("robotbench.report.write_report", side_effect=mutate_after_scoring):
            report = build_report(self.samples, self.root, self.output)
        page = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertEqual(report["vlm"]["metrics"]["ordered"]["correct"], 1)
        self.assertIn("validated evidence", page)
        self.assertNotIn("unvalidated replacement", page)


if __name__ == "__main__":
    unittest.main()
