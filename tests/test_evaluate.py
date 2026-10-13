"""Offline checks for scoring and resume integrity; never invoke Ollama."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import is_reusable_record, normalize_answer, run_evaluation, score_answer, seal_record


class EvaluationTests(unittest.TestCase):
    def test_exact_match_is_conservative(self):
        self.assertEqual(normalize_answer("  ＭＡＰＬＥ\n Cafe  "), "maple cafe")
        self.assertTrue(score_answer(" $３.５０ ", ["$3.50"]))
        self.assertFalse(score_answer("The price is $3.50", ["$3.50"]))
        self.assertFalse(score_answer("3.50", ["$3.50"]))
        self.assertFalse(score_answer("4.", ["4"]))
        self.assertIsNone(score_answer("anything", []))

    def test_resume_only_reuses_complete_matching_untampered_results(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "image.png").write_bytes(b"first-image")
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps([{"id": "one", "image": "image.png", "question": "Count?", "accepted_answers": ["4"]}]), encoding="utf-8")
            calls = []

            def fake_analyzer(*args, **kwargs):
                calls.append((args, kwargs))
                return {"status": "ok", "answer": "4", "elapsed_seconds": 0.1, "digest": options["model_digest"]}

            options = dict(model="test-model", model_digest="weights-a", analyzer=fake_analyzer, implementation_sha256="core-a")
            first = run_evaluation(manifest, base / "result.json", **options)
            record = first["results"][0]
            self.assertTrue(is_reusable_record(record, record["request"]))
            second = run_evaluation(manifest, base / "result.json", **options)
            self.assertEqual(len(calls), 1)
            self.assertEqual(second["summary"]["cached_results"], 1)
            for field, value in (("model", "different"), ("model_digest", "weights-b"), ("max_edge", 512), ("max_tokens", 64), ("implementation_sha256", "core-b")):
                request = deepcopy(record["request"])
                request["config"][field] = value
                self.assertFalse(is_reusable_record(record, request))
            for field, value in (("question", "Changed?"), ("image_sha256", "changed"), ("accepted_answers", ["5"])):
                request = deepcopy(record["request"])
                request[field] = value
                self.assertFalse(is_reusable_record(record, request))
            for mutation in ("answer", "missing", "complete"):
                altered = deepcopy(record)
                if mutation == "answer":
                    altered["result"]["answer"] = "5"
                elif mutation == "missing":
                    del altered["wall_seconds"]
                else:
                    altered["complete"] = False
                self.assertFalse(is_reusable_record(altered, record["request"]))
            # Even an intact checkpoint must agree with the actual model used.
            altered = deepcopy(record)
            altered["result"]["digest"] = "weights-b"
            self.assertFalse(is_reusable_record(seal_record(altered), record["request"]))
            options["model_digest"] = "weights-b"
            run_evaluation(manifest, base / "result.json", **options)
            self.assertEqual(len(calls), 2)
            (base / "image.png").write_bytes(b"changed-image")
            run_evaluation(manifest, base / "result.json", **options)
            self.assertEqual(len(calls), 3)
            run_evaluation(manifest, base / "result.json", **options, force=True)
            self.assertEqual(len(calls), 4)

    def test_errors_and_truncation_do_not_inflate_accuracy_or_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "image.png").write_bytes(b"synthetic")
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps([
                {"id": name, "image": "image.png", "question": name, "accepted_answers": ["4"]}
                for name in ("correct", "truncated", "failure")
            ]), encoding="utf-8")
            calls = []

            def fake_analyzer(image_path, question, **kwargs):
                calls.append(question)
                if question == "failure":
                    raise ConnectionError("test service unavailable")
                return {"status": "truncated" if question == "truncated" else "ok", "answer": "4", "digest": "weights-a"}

            options = dict(model="test-model", model_digest="weights-a", analyzer=fake_analyzer, implementation_sha256="core-a")
            document = run_evaluation(manifest, base / "result.json", **options)
            summary = document["summary"]
            self.assertEqual(summary["call_successes"], 1)
            self.assertEqual(summary["complete_answers"], 1)
            self.assertEqual(summary["exact_matches"], 1)
            self.assertAlmostEqual(summary["exact_match_accuracy_all_labelled"], 1 / 3)
            self.assertEqual(summary["exact_match_accuracy_successful_calls"], 1.0)
            self.assertEqual(document["results"][2]["result"]["error"]["type"], "ConnectionError")
            run_evaluation(manifest, base / "result.json", **options)
            self.assertEqual(calls.count("correct"), 1)
            self.assertEqual(calls.count("truncated"), 2)
            self.assertEqual(calls.count("failure"), 2)

    def test_changed_or_missing_actual_digest_is_failed_and_never_cached(self):
        for actual_digest in ("weights-b", None):
            with self.subTest(actual_digest=actual_digest), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                (base / "image.png").write_bytes(b"synthetic")
                manifest = base / "manifest.json"
                manifest.write_text(json.dumps([
                    {"id": "one", "image": "image.png", "question": "Count?", "accepted_answers": ["4"]}
                ]), encoding="utf-8")
                calls = []

                def fake_analyzer(*args, **kwargs):
                    calls.append(True)
                    result = {"status": "ok", "answer": "4"}
                    if actual_digest is not None:
                        result["digest"] = actual_digest
                    return result

                options = dict(model="test-model", model_digest="weights-a", analyzer=fake_analyzer, implementation_sha256="core-a")
                document = run_evaluation(manifest, base / "result.json", **options)
                record = document["results"][0]
                self.assertEqual(record["result"]["status"], "error")
                self.assertIn("Model digest mismatch", record["result"]["error"]["message"])
                self.assertIn("weights-a", record["result"]["error"]["message"])
                self.assertFalse(record["call_succeeded"])
                self.assertFalse(record["complete"])
                self.assertFalse(record["exact_match"])
                self.assertEqual(document["summary"]["exact_matches"], 0)
                self.assertEqual(document["summary"]["exact_match_accuracy_all_labelled"], 0)
                self.assertEqual(document["cache"], {})
                run_evaluation(manifest, base / "result.json", **options)
                self.assertEqual(len(calls), 2)

    def test_interrupt_preserves_completed_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "image.png").write_bytes(b"synthetic")
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps([
                {"id": str(i), "image": "image.png", "question": str(i), "accepted_answers": ["4"]}
                for i in range(2)
            ]), encoding="utf-8")

            def interrupted(image_path, question, **kwargs):
                if question == "1":
                    raise KeyboardInterrupt()
                return {"status": "ok", "answer": "4", "digest": "weights-a"}

            with self.assertRaises(KeyboardInterrupt):
                run_evaluation(manifest, base / "result.json", model="test-model", model_digest="weights-a", analyzer=interrupted, implementation_sha256="core-a")
            checkpoint = json.loads((base / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(checkpoint["interrupted"])
            self.assertEqual(checkpoint["summary"]["pending_examples"], 1)
            self.assertEqual(len(checkpoint["cache"]), 1)


if __name__ == "__main__":
    unittest.main()
