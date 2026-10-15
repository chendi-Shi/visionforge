"""Reproducible CPU robot action-effect experiments.

Example: python -m robotbench.run --data data/robotbench --output data/robot-run features
Then: cv, vlm, report. Downloads are a separate explicit data-preparation command.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from filelock import FileLock

from robotbench.data import load_samples, model_input, REVISION
from robotbench.metrics import evaluate, compare_paired
from video_media import _atomic_json, _sha256


CONDITIONS = ("after_only", "ordered", "reversed")
SELECTION_SEED = "visionforge-ur5-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select_pilot(samples):
    """One sample per base task family, chosen BEFORE inference without labels."""
    groups = sorted({s["task_group"] for s in samples})
    return [min((s for s in samples if s["task_group"] == group),
                key=lambda s: digest([SELECTION_SEED, s["sample_id"]])) for group in groups]


def request_identity(plan, inputs, condition):
    clean = {"task": inputs["task"], "subtask": inputs["subtask"],
             "frames": [{k: v for k, v in f.items() if k != "path"} for f in inputs["frames"]]}
    return digest({"plan": plan, "input": clean, "condition": condition})


def freeze_plan(samples, root, output, model, view, max_tokens):
    import core
    from robotbench.models import SYSTEM
    client = core._create_client()
    try:
        name, model_digest, _ = core._inspect_model(client, model)
    finally:
        client.close()
    if not model_digest:
        raise ValueError("A pinned local model digest is required")
    selected = select_pilot(samples)
    plan = {"dataset_revision": REVISION, "selection": "one sample per task family, minimum SHA256(seed,sample_id), labels unused",
            "seed": SELECTION_SEED, "sample_ids": [s["sample_id"] for s in selected], "conditions": list(CONDITIONS),
            "model": name, "digest": model_digest, "view": view, "max_tokens": max_tokens,
            "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "inputs_sha256": _sha256(root / "inputs.jsonl"),
            "labels_sha256": _sha256(root / "labels.jsonl"),
            "code_sha256": {p.name: _sha256(p) for p in (Path(__file__), Path(__file__).with_name("models.py"),
                                                       Path(__file__).with_name("data.py"), Path(core.__file__))}}
    path = output / "vlm-plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Run identity changed; choose a new output directory instead of mixing experiments")
    else:
        _atomic_json(path, plan)
    return plan, selected


def validate_saved_prediction(row, sample, condition, plan, request_id):
    if row.get("request_sha256") != request_id:
        raise ValueError("Existing prediction belongs to changed inputs or settings")
    if row.get("sample_id") != sample["sample_id"] or row.get("condition") != condition:
        raise ValueError("Mismatched prediction file")
    if row.get("status") not in {"ok", "failed"} or row.get("prediction") not in {"success", "failure", "abstain"}:
        raise ValueError("Invalid saved prediction status or label")
    if row["status"] == "failed" and row["prediction"] != "abstain":
        raise ValueError("Failed prediction must remain an abstention")
    if row["status"] == "ok" and (row.get("digest") != plan["digest"] or row.get("model") != plan["model"]):
        raise ValueError("Successful prediction model identity mismatch")


def predict_vlm(samples, root, output, args):
    from robotbench.models import judge_effect
    from video_jobs import default_data_dir
    plan, selected = freeze_plan(samples, root, output, args.model, args.view, args.max_tokens)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    # Same resource lock as the existing video/search workers. A separate process
    # must not defeat their single-CPU inference policy.
    lock_root = default_data_dir()
    lock_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_root / "inference.lock"), timeout=0):
        completed = 0
        for sample in selected:
            for condition in CONDITIONS:
                path = predictions_dir / f"{sample['sample_id']}.{condition}.json"
                inputs = model_input(sample, root, condition=condition, view=plan["view"])
                request_id = request_identity(plan, inputs, condition)
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    validate_saved_prediction(saved, sample, condition, plan, request_id)
                else:
                    print(json.dumps({"stage": "vlm", "sample_id": sample["sample_id"], "condition": condition,
                                      "completed": completed, "total": len(selected) * len(CONDITIONS)}), flush=True)
                    saved = judge_effect(inputs, model=plan["model"], max_tokens=plan["max_tokens"], expected_digest=plan["digest"])
                    saved.update(sample_id=sample["sample_id"], condition=condition, request_sha256=request_id,
                                 created_at=datetime.now(timezone.utc).isoformat())
                    _atomic_json(path, saved)
                completed += 1
                print(json.dumps({"stage": "result", "completed": completed, "total": len(selected) * len(CONDITIONS),
                                  "condition": condition, "status": saved["status"], "prediction": saved["prediction"],
                                  "seconds": saved.get("elapsed_seconds")}), flush=True)
    return {"completed": completed, "plan": plan}


def build_report(samples, root, output):
    from robotbench.report import write_report
    from robotbench.diagnostics import counterfactual_pairs
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = None
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": manifest,
              "hardware": {"platform": platform.system(), "machine": platform.machine(), "execution": "CPU; local Ollama only"},
              "cv": None, "vlm": None,
              "limits": ["This measures two-state subtask outcome verification, not continuous motion or failure-onset localization.",
                         "UR5 includes counterfactual instructions and altered state pairs; 140 rows are not 140 independent robot rollouts.",
                         "Task-family cross-validation probes are exploratory, not an official zero-shot OOD leaderboard score.",
                         "Base taskvar groups can contain similar or identical task meanings across groups; exact-image separation is not semantic OOD separation.",
                         "Reversed inputs are a corruption diagnostic scored against original labels, not labeled reverse-action trajectories.",
                         "No robot control, policy training, real-time deployment, or causal failure diagnosis is demonstrated."]}
    cv_path = output / "cv.json"
    if cv_path.exists():
        cv = json.loads(cv_path.read_text(encoding="utf-8"))
        if cv.get("inputs_sha256") != _sha256(root / "inputs.jsonl") or cv.get("labels_sha256") != _sha256(root / "labels.jsonl"):
            raise ValueError("CV input or label identity changed")
        report["cv"] = {k: v for k, v in cv.items() if k != "predictions"}
        report["cv"]["metrics"] = {method: evaluate(samples, rows) for method, rows in cv["predictions"].items()}
        report["cv"]["paired_after_to_transition"] = compare_paired(samples, cv["predictions"]["after"], cv["predictions"]["transition"])
        report["cv"]["counterfactual_pairs"] = {method: counterfactual_pairs(samples, rows) for method, rows in cv["predictions"].items()}
    plan_path = output / "vlm-plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["inputs_sha256"] != _sha256(root / "inputs.jsonl") or plan["dataset_revision"] != REVISION:
            raise ValueError("VLM dataset identity changed")
        if plan.get("labels_sha256") and plan["labels_sha256"] != _sha256(root / "labels.jsonl"):
            raise ValueError("VLM evaluation label identity changed")
        by_id = {s["sample_id"]: s for s in samples}
        selected = [by_id[i] for i in plan["sample_ids"]]
        records = {condition: [] for condition in CONDITIONS}
        for sample in selected:
            for condition in CONDITIONS:
                path = output / "predictions" / f"{sample['sample_id']}.{condition}.json"
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    inputs = model_input(sample, root, condition=condition, view=plan["view"])
                    validate_saved_prediction(row, sample, condition, plan, request_identity(plan, inputs, condition))
                    records[condition].append(row)
        report["vlm"] = {"plan": plan, "class_counts": dict(Counter(s["label"] for s in selected)),
                         "expected_calls": len(selected) * len(CONDITIONS), "completed_calls": sum(map(len, records.values())),
                         "successful_calls": sum(r["status"] == "ok" for rows in records.values() for r in rows),
                         "inference_seconds": sum(r.get("elapsed_seconds", 0) for rows in records.values() for r in rows),
                         "metrics": {c: evaluate(selected, rows) for c, rows in records.items()},
                         "paired_after_to_ordered": compare_paired(selected, records["after_only"], records["ordered"]),
                         "paired_ordered_to_reversed": compare_paired(selected, records["ordered"], records["reversed"])}
        a = {r["sample_id"]: r["prediction"] for r in records["ordered"] if r["status"] == "ok"}
        b = {r["sample_id"]: r["prediction"] for r in records["reversed"] if r["status"] == "ok"}
        matched = set(a) & set(b)
        report["vlm"]["order_sensitivity"] = {"paired_valid_outputs": len(matched), "prediction_flips": sum(a[i] != b[i] for i in matched),
                                                "flip_rate": sum(a[i] != b[i] for i in matched) / len(matched) if matched else None}
        identical_ids = []
        for sample in selected:
            frames = model_input(sample, root, view=plan["view"])["frames"]
            if [f["sha256"] for f in frames if f["phase"] == "before"] == [f["sha256"] for f in frames if f["phase"] == "after"]:
                identical_ids.append(sample["sample_id"])
        changed = matched - set(identical_ids)
        flips = sum(a[i] != b[i] for i in changed)
        report["vlm"]["order_sensitivity_changed_inputs"] = {
            "identical_before_after_ids": identical_ids, "selected_samples": len(selected) - len(identical_ids),
            "paired_valid_outputs": len(changed), "prediction_flips": flips,
            "flip_rate": flips / len(changed) if changed else None,
            "definition": "Exclude identical ordered before/after source image hashes; valid schema outputs include explicit abstentions."}
        if cv_path.exists():
            ids = set(plan["sample_ids"])
            report["vlm"]["cv_on_same_pilot"] = {m: evaluate(selected, [r for r in rows if r["sample_id"] in ids]) for m, rows in cv["predictions"].items()}
    _atomic_json(output / "metrics.json", report)
    write_report(report, samples, root, output, prediction_records=records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/robotbench"))
    parser.add_argument("--output", type=Path, default=Path("data/robot-run"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("features")
    commands.add_parser("cv")
    vlm = commands.add_parser("vlm")
    vlm.add_argument("--model", default="qwen3-vl:2b")
    vlm.add_argument("--view", choices=("all", "0", "1", "2"), default="all")
    vlm.add_argument("--max-tokens", type=int, default=160)
    commands.add_parser("report")
    args = parser.parse_args()
    root, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(root)
    with FileLock(str(output / ".run.lock"), timeout=0):
        if args.command == "features":
            from robotbench.probe import encode_samples
            encode_samples(samples, root, output)
        elif args.command == "cv":
            from robotbench.probe import cross_validate, _hash
            features = json.loads((output / "features.json").read_text(encoding="utf-8"))
            expected = _hash([{k: s[k] for k in ("sample_id", "frames", "subtask_instruction")} for s in samples])
            if features["input_sha256"] != expected:
                raise ValueError("Features belong to changed inputs")
            result = cross_validate(samples, features)
            result.update(inputs_sha256=_sha256(root / "inputs.jsonl"), labels_sha256=_sha256(root / "labels.jsonl"),
                          features_sha256=_sha256(output / "features.json"))
            _atomic_json(output / "cv.json", result)
        elif args.command == "vlm":
            predict_vlm(samples, root, output, args)
        else:
            report = build_report(samples, root, output)
            print(json.dumps({"report": str(output / "index.html"), "cv_ready": report["cv"] is not None,
                              "vlm_completed": (report["vlm"] or {}).get("completed_calls")}), flush=True)


if __name__ == "__main__":
    main()
