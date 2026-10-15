"""Recompute robot scores and cross-check raw predictions/source labels offline.

Hashes establish internal integrity, not an author's digital signature.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotbench.data import sha256_file
from robotbench.diagnostics import counterfactual_pairs
from robotbench.metrics import evaluate, compare_paired

CONDITIONS = ("after_only", "ordered", "reversed")
METHODS = {"train_majority", "text_only", "after", "transition"}
REQUIRED = {"metrics.json", "metrics.csv", "index.html", "cv.json", "features.json", "vlm-plan.json",
            "diagnostics.json", "samples.json", "evaluation-samples.json", "source-inputs.jsonl",
            "source-labels.jsonl", "source-manifest.json", "ATTRIBUTION.txt", "LICENSE-DATA.txt"}
LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\[A-Za-z0-9_.-]+[\\/]|/(?:Users|home|mnt|private|tmp)/)")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe release path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name in {".", ".."}:
        raise ValueError("Unsafe release path")
    return path


def _indexed(rows, name):
    result = {}
    for row in rows:
        identifier = row.get("sample_id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in result:
            raise ValueError(f"Invalid or duplicate sample ID in {name}")
        result[identifier] = row
    return result


def check_public_metadata(root):
    """Reject machine paths; do not silently rewrite recorded model outputs."""
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".html", ".csv", ".txt"}:
            if LOCAL_PATH.search(path.read_text(encoding="utf-8-sig")):
                raise ValueError(f"Local filesystem path in public metadata: {path.relative_to(root)}")


def _inference_input(sample, condition, view):
    view = {"front": "0", "left": "1", "right": "2"}.get(str(view), str(view))
    if condition not in CONDITIONS or view not in {"all", "0", "1", "2"}:
        raise ValueError("Unknown released input condition or view")
    frames = [{key: frame[key] for key in ("frame_id", "phase", "view", "sha256")}
              for frame in sample["frames"] if view == "all" or frame["view"] == view]
    before, after = [f for f in frames if f["phase"] == "before"], [f for f in frames if f["phase"] == "after"]
    expected = 3 if view == "all" else 1
    if len(before) != expected or len(after) != expected:
        raise ValueError("Released inference views are incomplete")
    selected = after if condition == "after_only" else before + after if condition == "ordered" else after + before
    if condition == "reversed":
        for index, frame in enumerate(selected):
            frame["phase"] = "before" if index < expected else "after"
            frame["frame_id"] = f"{frame['phase']}_{frame['view']}"
    return {"task": sample["task_instruction"], "subtask": sample["subtask_instruction"], "frames": selected}


def _check_prediction(row, sample, plan, condition):
    if row.get("sample_id") != sample["sample_id"] or row.get("condition") != condition:
        raise ValueError("Raw prediction sample or condition mismatch")
    inputs = _inference_input(sample, condition, plan["view"])
    if row.get("request_sha256") != _digest({"plan": plan, "input": inputs, "condition": condition}):
        raise ValueError("Raw prediction request identity mismatch")
    if hashlib.sha256(row.get("system_prompt", "").encode()).hexdigest() != plan["system_prompt_sha256"]:
        raise ValueError("Raw prediction system prompt mismatch")
    if row.get("options") != {"num_ctx": 4096, "num_predict": plan["max_tokens"], "num_thread": 4, "temperature": 0, "seed": 0}:
        raise ValueError("Raw prediction inference options mismatch")
    if row.get("status") == "ok":
        if row.get("digest") != plan["digest"] or row.get("model") != plan["model"]:
            raise ValueError("Raw prediction model identity mismatch")
        if row.get("done_reason") in {"length", "max_tokens", "max_length"}:
            raise ValueError("Truncated prediction claimed successful")
        answer = json.loads(row.get("raw_response", ""))
        if set(answer) != {"prediction", "evidence"} or answer["prediction"] not in {"success", "failure", "abstain"}:
            raise ValueError("Raw prediction schema mismatch")
        if not isinstance(answer["evidence"], str) or not 1 <= len(answer["evidence"]) <= 280 or any(row.get(key) != answer[key] for key in answer):
            raise ValueError("Parsed prediction differs from valid raw response")
        if json.loads(row.get("question", "")) != {"overall_task": inputs["task"], "subtask_to_evaluate": inputs["subtask"]}:
            raise ValueError("Raw prediction question differs from source inputs")
        observed = [{key: frame[key] for key in ("phase", "view", "sha256")} for frame in row.get("input", {}).get("frames", [])]
        if observed != [{key: frame[key] for key in ("phase", "view", "sha256")} for frame in inputs["frames"]]:
            raise ValueError("Raw prediction image audit differs from source inputs")
    elif row.get("status") != "failed" or row.get("prediction") != "abstain":
        raise ValueError("Failed prediction must remain an abstention")


def verify(root, *, require_archive=False):
    root = Path(root).resolve()
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    manifest, listed = read("release-manifest.json"), {}
    if manifest.get("format") != 2:
        raise ValueError("Unsupported robot release manifest format")
    for entry in manifest["files"]:
        name = _relative(entry["path"]).as_posix()
        if name in listed or name in {"release-manifest.json", "robot-benchmark.zip"}:
            raise ValueError("Duplicate or reserved manifest path")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Release asset changed or missing: {name}")
        listed[name] = entry
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"release-manifest.json", "robot-benchmark.zip"}
    if actual != set(listed) or not REQUIRED <= set(listed):
        raise ValueError("Release file inventory is incomplete or contains unlisted files")
    check_public_metadata(root)
    source = read("source-manifest.json")
    inputs_hash, labels_hash = sha256_file(root / "source-inputs.jsonl"), sha256_file(root / "source-labels.jsonl")
    if source["files"]["inputs.jsonl"] != inputs_hash or source["files"]["labels.jsonl"] != labels_hash:
        raise ValueError("Source input or label hash differs from source manifest")
    inputs = _indexed([json.loads(line) for line in (root / "source-inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()], "source inputs")
    labels = _indexed([json.loads(line) for line in (root / "source-labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()], "source labels")
    samples, report, cv = read("evaluation-samples.json"), read("metrics.json"), read("cv.json")
    by_id = _indexed(samples, "evaluation samples")
    if set(inputs) != set(labels) or set(inputs) != set(by_id) or len(samples) != source["samples"]:
        raise ValueError("Source input, label and evaluation sample IDs differ")
    for identifier, sample in by_id.items():
        expected = inputs[identifier] | labels[identifier]
        for key in ("sample_id", "task_group", "episode_group", "label", "failure_category", "task_instruction", "subtask_instruction"):
            if sample.get(key) != expected.get(key):
                raise ValueError(f"Evaluation sample differs from source ground truth or input: {identifier} / {key}")
        if len(sample.get("frames", [])) != len(expected["frames"]):
            raise ValueError("Evaluation frame count differs from source")
        for frame, original in zip(sample["frames"], expected["frames"]):
            if any(frame.get(key) != original.get(key) for key in ("frame_id", "phase", "view", "sha256")):
                raise ValueError("Evaluation frame identity differs from source")
            name = "assets/" + original["sha256"] + ".png"
            if frame.get("path") != (name if name in listed else None) or frame.get("asset_included") != (name in listed):
                raise ValueError("Evaluation asset availability is inconsistent")
            if name in listed and listed[name]["sha256"] != original["sha256"]:
                raise ValueError("Released image differs from source pixels")
    if report["dataset"] != source or manifest["dataset_revision"] != source["revision"]:
        raise ValueError("Report or release dataset identity differs from source manifest")
    if cv["inputs_sha256"] != inputs_hash or cv["labels_sha256"] != labels_hash or cv["features_sha256"] != sha256_file(root / "features.json"):
        raise ValueError("CV input, label or feature artifact identity mismatch")
    features = read("features.json")
    original_order = [inputs[identifier] for identifier in inputs]
    if features["input_sha256"] != _digest([{key: sample[key] for key in ("sample_id", "frames", "subtask_instruction")} for sample in original_order]):
        raise ValueError("Features were generated from different input metadata")
    if set(_indexed(features["rows"], "features")) != set(by_id):
        raise ValueError("Feature sample set differs from source")
    if {k: v for k, v in cv.items() if k != "predictions"} != {k: v for k, v in report["cv"].items() if k not in {"metrics", "paired_after_to_transition", "counterfactual_pairs"}}:
        raise ValueError("CV report provenance differs from raw CV output")
    diagnostics = read("diagnostics.json")
    if "counterfactual_pairs" in report["cv"] and report["cv"]["counterfactual_pairs"] != diagnostics:
        raise ValueError("Report counterfactual diagnostic differs from released diagnostics")
    if set(cv["predictions"]) != METHODS or set(report["cv"]["metrics"]) != METHODS or set(diagnostics) != METHODS:
        raise ValueError("Incomplete prespecified CV methods")
    for method, rows in cv["predictions"].items():
        if set(_indexed(rows, method)) != set(by_id):
            raise ValueError("CV prediction denominator differs from full source set")
        if evaluate(samples, rows) != report["cv"]["metrics"][method]:
            raise ValueError(f"CV score cannot be reproduced: {method}")
        if counterfactual_pairs(samples, rows) != diagnostics[method]:
            raise ValueError(f"Counterfactual diagnostic changed: {method}")
    if compare_paired(samples, cv["predictions"]["after"], cv["predictions"]["transition"]) != report["cv"]["paired_after_to_transition"]:
        raise ValueError("Paired CV difference cannot be reproduced")
    plan = read("vlm-plan.json")
    if plan != report["vlm"]["plan"] or plan["inputs_sha256"] != inputs_hash or plan["dataset_revision"] != source["revision"] or tuple(plan["conditions"]) != CONDITIONS:
        raise ValueError("VLM plan or source identity mismatch")
    if plan.get("labels_sha256") and plan["labels_sha256"] != labels_hash:
        raise ValueError("VLM planned evaluation labels differ from source")
    if set(plan.get("code_sha256", {})) != {"run.py", "models.py", "data.py", "core.py"}:
        raise ValueError("Frozen inference source inventory is incomplete")
    if len(plan["sample_ids"]) != len(set(plan["sample_ids"])) or not set(plan["sample_ids"]) <= set(by_id):
        raise ValueError("Invalid VLM selected sample IDs")
    chosen = [by_id[identifier] for identifier in plan["sample_ids"]]
    if read("samples.json") != chosen:
        raise ValueError("Pilot metadata differs from source evaluation samples")
    for sample in chosen:
        if any("assets/" + frame["sha256"] + ".png" not in listed for frame in _inference_input(sample, "ordered", plan["view"])["frames"]):
            raise ValueError("Pilot image is missing from release")
    records, expected_files = {condition: [] for condition in CONDITIONS}, set()
    for sample in chosen:
        for condition in CONDITIONS:
            name = f"predictions/{sample['sample_id']}.{condition}.json"
            expected_files.add(name)
            row = read(name)
            _check_prediction(row, sample, plan, condition)
            records[condition].append(row)
    if {name for name in listed if name.startswith("predictions/")} != expected_files:
        raise ValueError("Prediction inventory differs from frozen plan")
    vlm = report["vlm"]
    if vlm["expected_calls"] != len(chosen) * len(CONDITIONS) or vlm["completed_calls"] != len(expected_files):
        raise ValueError("Release completion counts differ from raw prediction files")
    if vlm["class_counts"] != dict(Counter(sample["label"] for sample in chosen)):
        raise ValueError("Pilot class counts differ from source labels")
    if vlm["successful_calls"] != sum(row["status"] == "ok" for rows in records.values() for row in rows):
        raise ValueError("Successful-call count differs from raw records")
    durations = [row.get("elapsed_seconds", 0) for rows in records.values() for row in rows]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in durations):
        raise ValueError("Invalid raw inference duration")
    if vlm["inference_seconds"] != sum(durations):
        raise ValueError("Inference duration differs from raw records")
    # The frozen run counts all valid schema outputs, including explicit
    # abstentions. Preserve that three-state definition instead of silently
    # presenting a binary-answer-only sensitivity rate.
    ordered = {row["sample_id"]: row["prediction"] for row in records["ordered"] if row["status"] == "ok"}
    reversed_rows = {row["sample_id"]: row["prediction"] for row in records["reversed"] if row["status"] == "ok"}
    matched = set(ordered) & set(reversed_rows)
    flips = sum(ordered[identifier] != reversed_rows[identifier] for identifier in matched)
    expected_sensitivity = {"paired_valid_outputs": len(matched), "prediction_flips": flips,
                            "flip_rate": flips / len(matched) if matched else None}
    if vlm["order_sensitivity"] != expected_sensitivity:
        raise ValueError("Order-sensitivity counts differ from raw records")
    if "order_sensitivity_changed_inputs" in vlm:
        identical = []
        for sample in chosen:
            frames = _inference_input(sample, "ordered", plan["view"])["frames"]
            if [f["sha256"] for f in frames if f["phase"] == "before"] == [f["sha256"] for f in frames if f["phase"] == "after"]:
                identical.append(sample["sample_id"])
        distinct = matched - set(identical)
        changes = sum(ordered[identifier] != reversed_rows[identifier] for identifier in distinct)
        expected_changed = {"identical_before_after_ids": identical, "selected_samples": len(chosen) - len(identical),
                            "paired_valid_outputs": len(distinct), "prediction_flips": changes,
                            "flip_rate": changes / len(distinct) if distinct else None,
                            "definition": "Exclude identical ordered before/after source image hashes; valid schema outputs include explicit abstentions."}
        if vlm["order_sensitivity_changed_inputs"] != expected_changed:
            raise ValueError("Changed-input order sensitivity differs from source observations and raw records")
    for condition, rows in records.items():
        if evaluate(chosen, rows) != vlm["metrics"][condition]:
            raise ValueError(f"VLM score cannot be reproduced: {condition}")
    for name, left, right in (("paired_after_to_ordered", "after_only", "ordered"), ("paired_ordered_to_reversed", "ordered", "reversed")):
        if compare_paired(chosen, records[left], records[right]) != vlm[name]:
            raise ValueError(f"Paired VLM difference cannot be reproduced: {name}")
    for method, rows in cv["predictions"].items():
        if evaluate(chosen, [row for row in rows if row["sample_id"] in plan["sample_ids"]]) != vlm["cv_on_same_pilot"][method]:
            raise ValueError("Matched-pilot CV score cannot be reproduced")
    source_hashes = [("protocol_sources/" + name, expected) for name, expected in plan["code_sha256"].items()]
    source_hashes += [("protocol_sources/probe.py", cv["code_sha256"]), ("protocol_sources/models.py", cv["models_code_sha256"])]
    source_hashes += list(manifest.get("evaluation_sources", {}).items())
    if not {"evaluation_sources/metrics.py", "evaluation_sources/diagnostics.py"} <= set(manifest.get("evaluation_sources", {})):
        raise ValueError("Evaluation code snapshot is missing")
    for name, expected in source_hashes:
        _relative(name)
        if name not in listed or not (root / name).is_file() or sha256_file(root / name) != expected:
            raise ValueError(f"Protocol or evaluation source missing or changed: {name}")
    for name in ("metrics.py", "diagnostics.py"):
        current = Path(__file__).resolve().parents[1] / "robotbench" / name
        if sha256_file(current) != manifest["evaluation_sources"]["evaluation_sources/" + name]:
            raise ValueError("Evaluator source version differs from recorded scoring code")
    with (root / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_csv = {(experiment, method): metrics for experiment, methods in
                    (("task_family_cv", report["cv"]["metrics"]), ("local_vlm_pilot", vlm["metrics"])) for method, metrics in methods.items()}
    if len(csv_rows) != len(expected_csv) or {(row["experiment"], row["method"]) for row in csv_rows} != set(expected_csv):
        raise ValueError("CSV method inventory differs from recomputed scores")
    for row in csv_rows:
        metric = expected_csv[(row["experiment"], row["method"])]
        for key in ("n", "correct", "accuracy", "balanced_accuracy", "macro_f1", "failure_recall", "coverage"):
            expected = metric["per_class"]["failure"]["recall"] if key == "failure_recall" else metric[key]
            actual = None if row[key] == "" else float(row[key])
            if actual != expected:
                raise ValueError("CSV values differ from recomputed scores")
    archive = root / "robot-benchmark.zip"
    if require_archive or archive.exists():
        if not archive.is_file():
            raise ValueError("Release archive is missing")
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or set(names) != set(listed) | {"release-manifest.json"}:
                raise ValueError("Archive inventory differs from release")
            for name in names:
                _relative(name)
                if hashlib.sha256(bundle.read(name)).hexdigest() != sha256_file(root / name):
                    raise ValueError("Archive content differs from release")
    return {"verified_files": len(listed), "cv_samples": len(samples), "vlm_samples": len(chosen),
            "headline_scores": "exactly recomputed", "source_labels": "matched to packaged source bytes and recorded hashes",
            "model_calls": 0, "authenticity_limit": "Hash consistency is not a digital signature; compare published source hashes independently."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("artifacts/robot-benchmark"))
    parser.add_argument("--require-archive", action="store_true", help="Require the locally generated ZIP in addition to the Git-tracked payload")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, require_archive=args.require_archive), ensure_ascii=False))
