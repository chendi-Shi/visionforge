"""No inference or network: fixed-cell protocol and resume integrity tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from robotbench import vlm_diagnostic as d
from robotbench.models import prepare_panel
from video_media import _atomic_json, _sha256


class VLMDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "diagnostic"
        self.out.mkdir()
        image = self.root / "frame.png"
        Image.new("RGB", (32, 32), "red").save(image)
        self.inputs = {"task": "Move the fruit", "subtask": "Grasp the apple",
            "frames": [{"frame_id": f"{phase}_0", "phase": phase, "view": "0", "path": str(image),
                        "sha256": _sha256(image)} for phase in ("before", "after")]}
        _, panel = prepare_panel(self.inputs)
        self.entries = [{"sample_id": f"sample-{index}", "label": "success" if index < 2 else "failure",
                         "input": d._clean(self.inputs), "panel": panel} for index in range(4)]
        self.model = {"model": "local:qwen", "digest": "a" * 64, "family": "qwen3vl", "capabilities": ["vision"],
                      "thinking": None, "configuration_sha256": "b" * 64}
        self.plan = d._seal({"version": 1, "server_version": "test-version", "conditions": d.protocols(),
            "samples": self.entries, "models": {"qwen": self.model, "minicpm": {**self.model, "model": "local:mini"}},
            "execution_order": [[c, s["sample_id"]] for c in d.CONDITIONS for s in self.entries]}, "plan_sha256")
        _atomic_json(self.out / "plan.json", self.plan)

    def descriptor(self, condition="baseline_short", index=0):
        return d.request_descriptor(self.plan, self.entries[index], condition)

    def record(self, condition="baseline_short", index=0, content='{"prediction":"success"}'):
        descriptor = self.descriptor(condition, index)
        response = {"message": {"content": content}, "done_reason": "stop"}
        status, prediction, error = d.parse_response(response)
        return d._seal({"request": descriptor, "request_sha256": d.digest(descriptor), "response": response,
                       "status": status, "prediction": prediction, "error_kind": error}, "record_sha256")

    def test_conditions_isolate_enum_and_prefill_without_silent_prompt_changes(self):
        specs = d.protocols()
        baseline = specs["baseline_short"]
        reverse = deepcopy(baseline)
        reverse["format"]["properties"]["prediction"]["enum"].reverse()
        self.assertEqual(reverse, specs["enum_reversed"])
        self.assertEqual({**baseline, "assistant_prefill": None}, specs["no_prefill"])
        self.assertEqual({**baseline, "model_key": "minicpm", "assistant_prefill": None}, specs["minicpm"])
        self.assertEqual(baseline["options"]["num_predict"], 48)
        self.assertNotIn("evidence", baseline["format"]["properties"])
        self.assertNotIn("Give a short concrete visual evidence sentence", baseline["system_prompt"])

    def test_selection_uses_author_labels_and_stable_hash_not_predictions(self):
        samples = [{"sample_id": str(i), "label": "success" if i < 5 else "failure", "prediction": "success"} for i in range(10)]
        pilot = {"sample_ids": [s["sample_id"] for s in samples]}
        first = d.select_samples(samples, pilot)
        for sample in samples:
            sample["prediction"] = "failure"
        second = d.select_samples(list(reversed(samples)), {"sample_ids": list(reversed(pilot["sample_ids"]))})
        self.assertEqual([s["sample_id"] for s in first], [s["sample_id"] for s in second])
        self.assertEqual([s["label"] for s in second], ["success", "success", "failure", "failure"])

    def test_thinking_only_and_truncated_json_are_not_promoted_to_valid_final(self):
        value = '{"prediction":"failure"}'
        cases = [({"message": {"content": "", "thinking": value}, "done_reason": "stop"}, "thinking_without_final"),
                 ({"message": {"content": value}, "done_reason": "length"}, "truncated"),
                 ({"message": {"content": '{"prediction":"success","evidence":"extra"}'}}, "invalid_schema"),
                 ({"message": {"content": '{"prediction":[]}'}}, "invalid_schema")]
        for response, reason in cases:
            self.assertEqual(d.parse_response(response), ("failed", "abstain", reason))
        self.assertEqual(d.parse_response({"message": {"content": value, "thinking": "retained"}}), ("ok", "failure", None))

    def test_image_label_and_runtime_drift_block_before_model_call(self):
        for inputs in ({**self.inputs, "label": "failure"}, {**self.inputs, "task": "A changed task"}):
            with patch.object(d.core, "_create_client") as create, self.assertRaises(ValueError):
                d.invoke(inputs, self.descriptor())
            create.assert_not_called()
        with patch.object(d, "server_version", return_value="changed"), patch.object(d.core, "_create_client") as create:
            with self.assertRaisesRegex(ValueError, "version changed"):
                d.invoke(self.inputs, self.descriptor())
            create.assert_not_called()
        client = Mock()
        with patch.object(d, "server_version", return_value="test-version"), patch.object(d.core, "_create_client", return_value=client), patch.object(d, "model_snapshot", return_value={**self.model, "digest": "changed"}):
            with self.assertRaisesRegex(ValueError, "digest or template"):
                d.invoke(self.inputs, self.descriptor())
        client.chat.assert_not_called()
        client.close.assert_called_once()

    def test_wire_contract_preserves_thinking_and_no_gt_enters_request(self):
        response = {"message": {"content": '{"prediction":"failure"}', "thinking": "raw diagnostic trace"}, "done_reason": "stop"}
        for condition in d.CONDITIONS:
            client = Mock()
            client.chat.return_value = deepcopy(response)
            descriptor = self.descriptor(condition)
            with patch.object(d, "server_version", return_value="test-version"), patch.object(d.core, "_create_client", return_value=client), patch.object(d, "model_snapshot", return_value=descriptor["model_identity"]):
                row = d.invoke(self.inputs, descriptor)
            self.assertEqual(row["response"], response)
            self.assertEqual(row["prediction"], "failure")
            d.validate_record(row, descriptor)
            kwargs = client.chat.call_args.kwargs
            self.assertFalse(kwargs["think"])
            self.assertEqual(len(kwargs["messages"]), 3 if condition in d.CONDITIONS[:2] else 2)
            self.assertEqual(set(json.loads(kwargs["messages"][1]["content"])), {"overall_task", "subtask_to_evaluate"})
            self.assertIsInstance(kwargs["messages"][1]["images"][0], bytes)
            self.assertNotIn("label", descriptor["input"])

    def test_cache_validation_recomputes_result_from_raw_and_request(self):
        row = self.record()
        d.validate_record(row, self.descriptor())
        row["prediction"] = "failure"
        row = d._seal({k: v for k, v in row.items() if k != "record_sha256"}, "record_sha256")
        with self.assertRaisesRegex(ValueError, "raw response"):
            d.validate_record(row, self.descriptor())
        with self.assertRaisesRegex(ValueError, "request identity"):
            d.validate_record(self.record(), self.descriptor("no_prefill"))

    def test_missing_and_failed_cells_keep_fixed_denominators(self):
        records = self.out / "records"
        records.mkdir()
        _atomic_json(records / "sample-0.baseline_short.json", self.record())
        _atomic_json(records / "sample-2.baseline_short.json", self.record(index=2, content="invalid"))
        result = d.report(self.out)
        self.assertEqual(len(result["rows"]), 16)
        self.assertEqual(result["recorded_calls"], 2)
        baseline = result["conditions"]["baseline_short"]
        self.assertEqual(baseline["expected"], 4)
        self.assertEqual(baseline["correct_of_fixed_four"], 1)
        self.assertEqual(baseline["answered"], 1)
        self.assertEqual(baseline["error_counts"], {"missing": 2, "invalid_schema": 1})
        self.assertEqual(result["versus_baseline_short"]["no_prefill"]["unpaired_or_invalid"], 4)
        _atomic_json(records / "unexpected.json", {})
        with self.assertRaisesRegex(ValueError, "Unexpected prediction"):
            d.report(self.out)

    def test_freeze_is_metadata_only_and_never_overwrites_a_changed_plan(self):
        (self.root / "inputs.jsonl").write_text("input", encoding="utf-8")
        (self.root / "labels.jsonl").write_text("label", encoding="utf-8")
        pilot = {"sample_ids": [s["sample_id"] for s in self.entries], "dataset_revision": d.REVISION,
                 "inputs_sha256": _sha256(self.root / "inputs.jsonl")}
        path = self.root / "pilot.json"
        _atomic_json(path, pilot)
        target = self.root / "frozen"
        client = Mock()
        with patch.object(d, "load_samples", return_value=self.entries), patch.object(d, "model_input", return_value=self.inputs), patch.object(d.core, "_create_client", return_value=client), patch.object(d, "model_snapshot", return_value=self.model), patch.object(d, "server_version", return_value="test-version"):
            first = d.freeze(self.root, target, path)
            self.assertEqual(d.freeze(self.root, target, path), first)
            original = (target / "plan.json").read_bytes()
            (self.root / "labels.jsonl").write_text("different labels", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                d.freeze(self.root, target, path)
            self.assertEqual((target / "plan.json").read_bytes(), original)
        client.chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
