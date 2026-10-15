"""Post hoc, four-sample Ollama protocol diagnostic; never a benchmark rerun.

Freeze performs metadata reads only. Run is the only command that invokes models.
The previous experiment and its raw predictions are never edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from filelock import FileLock
import httpx

import core
from robotbench.data import load_samples, model_input, REVISION
from robotbench.models import prepare_panel, SYSTEM
from video_media import _atomic_json, _sha256

SEED = "visionforge-vlm-protocol-diagnostic-v1"
PREFILL = "<think>\n\n</think>\n\n"
CONDITIONS = ("baseline_short", "enum_reversed", "no_prefill", "minicpm")
SHORT_SYSTEM = SYSTEM.replace("Give a short concrete visual evidence sentence. ", "Return only the prediction, without an explanation. ")
OPTIONS = {"num_ctx": 4096, "num_predict": 48, "num_thread": 4, "temperature": 0, "seed": 0}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seal(value, key):
    return {**value, key: digest(value)}


def _verify(value, key):
    if value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"Invalid {key}")


def server_version():
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
        response = client.get(core.HOST + "/api/version")
        response.raise_for_status()
        return response.json()["version"]


def model_snapshot(client, name):
    actual, model_digest, info = core._inspect_model(client, name)
    if not model_digest:
        raise ValueError("A pinned local model digest is required")
    return {"model": actual, "digest": model_digest,
            "family": (info.get("details") or {}).get("family"),
            "capabilities": sorted(info.get("capabilities") or []),
            "thinking": info.get("thinking"),
            # Only hashes of templates/Modelfiles: local absolute weight paths
            # in the latter must not leak into a portable diagnostic record.
            "configuration_sha256": digest({k: info.get(k) for k in
                ("template", "parameters", "modelfile", "model_info", "projector_info")})}


def protocols():
    schema = {"type": "object", "properties": {"prediction": {"type": "string",
              "enum": ["success", "failure", "abstain"]}},
              "required": ["prediction"], "additionalProperties": False}
    result = {}
    for condition in CONDITIONS:
        variant = deepcopy(schema)
        if condition == "enum_reversed":
            variant["properties"]["prediction"]["enum"].reverse()
        result[condition] = {"model_key": "minicpm" if condition == "minicpm" else "qwen",
                             "format": variant, "system_prompt": SHORT_SYSTEM,
                             "options": dict(OPTIONS), "think": False,
                             "assistant_prefill": PREFILL if condition in CONDITIONS[:2] else None,
                             "keep_alive": "5m"}
    return result


def source_hashes():
    return {p.name: _sha256(p) for p in (Path(__file__), Path(__file__).with_name("models.py"),
                                        Path(__file__).with_name("data.py"), Path(core.__file__))}


def select_samples(samples, pilot):
    by_id = {s["sample_id"]: s for s in samples}
    ids = pilot["sample_ids"]
    if len(ids) != len(set(ids)) or any(i not in by_id for i in ids):
        raise ValueError("Invalid original pilot sample IDs")
    selected = []
    for label in ("success", "failure"):
        candidates = sorted((by_id[i] for i in ids if by_id[i]["label"] == label),
                            key=lambda sample: digest([SEED, sample["sample_id"]]))
        if len(candidates) < 2:
            raise ValueError("The fixed pilot must contain two examples of each label")
        selected.extend(candidates[:2])
    return selected


def _clean(inputs):
    return {"task": inputs["task"], "subtask": inputs["subtask"],
            "frames": [{k: v for k, v in f.items() if k != "path"} for f in inputs["frames"]]}


def freeze(root, output, pilot_path, *, qwen="qwen3-vl:2b", minicpm="minicpm-v4.6:1b"):
    root, output, pilot_path = Path(root), Path(output), Path(pilot_path)
    samples, pilot = load_samples(root), _read(pilot_path)
    if pilot.get("inputs_sha256") != _sha256(root / "inputs.jsonl") or pilot.get("dataset_revision") != REVISION:
        raise ValueError("Original pilot belongs to different data")
    if pilot.get("labels_sha256") and pilot["labels_sha256"] != _sha256(root / "labels.jsonl"):
        raise ValueError("Original pilot label identity changed")
    selected = select_samples(samples, pilot)
    entries = []
    for sample in selected:
        inputs = model_input(sample, root, condition="ordered", view="all")
        _, panel = prepare_panel(inputs)
        entries.append({"sample_id": sample["sample_id"], "label": sample["label"],
                        "input": _clean(inputs), "panel": panel})
    client = core._create_client()
    try:
        models = {"qwen": model_snapshot(client, qwen), "minicpm": model_snapshot(client, minicpm)}
    finally:
        client.close()
    if models["qwen"]["family"] != "qwen3vl":
        raise ValueError("Qwen prefill diagnostic requires the qwen3vl family")
    plan = {"version": 1, "purpose": "Post hoc protocol diagnostic, not a reproduction of the original 48 calls or a held-out accuracy estimate",
            "selection": "Original pilot only; stable SHA256(seed,sample_id) lowest two per author label; predictions unused",
            "seed": SEED, "dataset_revision": REVISION,
            "inputs_sha256": _sha256(root / "inputs.jsonl"), "labels_sha256": _sha256(root / "labels.jsonl"),
            "pilot_plan_sha256": _sha256(pilot_path), "server_version": server_version(),
            "models": models, "conditions": protocols(), "samples": entries,
            "execution_order": [[condition, s["sample_id"]] for condition in CONDITIONS for s in selected],
            "code_sha256": source_hashes(),
            "differences_from_original": ["All conditions use prediction-only JSON without evidence",
                "Evidence-sentence prompt replaced by prediction-only instruction", "num_predict reduced from 160 to 48",
                "Only four label-balanced post hoc development samples, ordered three-camera panels"]}
    plan = _seal(plan, "plan_sha256")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists() and _read(path) != plan:
        raise ValueError("Frozen diagnostic identity changed; use another output directory")
    if not path.exists():
        _atomic_json(path, plan)
    return plan


def validate_plan(plan):
    _verify(plan, "plan_sha256")
    if plan.get("version") != 1 or plan.get("conditions") != protocols():
        raise ValueError("Unsupported diagnostic protocol")
    ids = [s["sample_id"] for s in plan["samples"]]
    if len(ids) != 4 or len(set(ids)) != 4 or Counter(s["label"] for s in plan["samples"]) != {"success": 2, "failure": 2}:
        raise ValueError("Diagnostic must preserve all four fixed samples")
    if plan["execution_order"] != [[c, i] for c in CONDITIONS for i in ids]:
        raise ValueError("Diagnostic execution matrix changed")


def request_descriptor(plan, entry, condition):
    spec = plan["conditions"][condition]
    return {"plan_sha256": plan["plan_sha256"], "sample_id": entry["sample_id"], "condition": condition,
            "input": entry["input"], "panel": entry["panel"], "protocol": spec,
            "model_identity": plan["models"][spec["model_key"]], "server_version": plan["server_version"]}


def parse_response(response):
    message = response.get("message") or {}
    content, thinking = str(message.get("content") or ""), str(message.get("thinking") or "")
    if response.get("done_reason") in {"length", "max_tokens", "max_length"}:
        return "failed", "abstain", "truncated"
    if not content.strip():
        return "failed", "abstain", "thinking_without_final" if thinking.strip() else "empty_final"
    try:
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {"prediction"} or value["prediction"] not in {"success", "failure", "abstain"}:
            raise ValueError("Unexpected prediction schema")
    except (ValueError, TypeError):
        return "failed", "abstain", "invalid_schema"
    return "ok", value["prediction"], None


def validate_record(row, descriptor):
    _verify(row, "record_sha256")
    if row.get("request_sha256") != digest(descriptor) or row.get("request") != descriptor:
        raise ValueError("Saved prediction request identity changed")
    expected = parse_response(row["response"]) if row.get("response") is not None else ("failed", "abstain", "transport_error")
    if (row.get("status"), row.get("prediction"), row.get("error_kind")) != expected:
        raise ValueError("Saved prediction disagrees with raw response")


def invoke(inputs, descriptor):
    """One call; verify bytes, runtime, and model before sending an image."""
    picture, panel = prepare_panel(inputs)
    if _clean(inputs) != descriptor["input"] or panel != descriptor["panel"]:
        raise ValueError("Diagnostic input or composed panel changed")
    if not all(isinstance(inputs[k], str) and 0 < len(inputs[k]) <= 4000 for k in ("task", "subtask")):
        raise ValueError("Expected bounded task/subtask text")
    if server_version() != descriptor["server_version"]:
        raise ValueError("Ollama server version changed")
    spec, identity = descriptor["protocol"], descriptor["model_identity"]
    started = time.perf_counter()
    response, error = None, None
    with core._INFERENCE_LOCK:
        client = core._create_client()
        try:
            if model_snapshot(client, identity["model"]) != identity:
                raise ValueError("Model digest or template identity changed")
            messages = [{"role": "system", "content": spec["system_prompt"]},
                        {"role": "user", "content": json.dumps({"overall_task": inputs["task"],
                           "subtask_to_evaluate": inputs["subtask"]}, ensure_ascii=False), "images": [picture]}]
            if spec["assistant_prefill"] is not None:
                messages.append({"role": "assistant", "content": spec["assistant_prefill"]})
            try:
                response = core._as_dict(client.chat(model=identity["model"], messages=messages,
                    format=spec["format"], options=spec["options"], think=spec["think"],
                    stream=False, keep_alive=spec["keep_alive"]))
            except Exception as exc:
                # Failed calls remain present in the fixed 16-call denominator.
                error = f"{type(exc).__name__}: {exc}"
        finally:
            client.close()
    status, prediction, error_kind = parse_response(response) if response is not None else ("failed", "abstain", "transport_error")
    return _seal({"request": descriptor, "request_sha256": digest(descriptor),
        "status": status, "prediction": prediction, "error_kind": error_kind, "error": error,
        "response": response, "elapsed_seconds": round(time.perf_counter() - started, 6),
        "created_at": datetime.now(timezone.utc).isoformat()}, "record_sha256")


def run(root, output):
    from video_jobs import default_data_dir
    root, output = Path(root), Path(output)
    plan = _read(output / "plan.json")
    validate_plan(plan)
    if plan["code_sha256"] != source_hashes():
        raise ValueError("Diagnostic implementation changed after freeze")
    if plan["dataset_revision"] != REVISION or any(plan[k + "_sha256"] != _sha256(root / (k + ".jsonl")) for k in ("inputs", "labels")):
        raise ValueError("Diagnostic dataset identity changed")
    samples = {s["sample_id"]: s for s in load_samples(root)}
    entries = {s["sample_id"]: s for s in plan["samples"]}
    # Validate every input before the first call, including cached entries.
    prepared = {}
    for identifier, entry in entries.items():
        inputs = model_input(samples[identifier], root, condition="ordered", view="all")
        _, panel = prepare_panel(inputs)
        if _clean(inputs) != entry["input"] or panel != entry["panel"] or samples[identifier]["label"] != entry["label"]:
            raise ValueError("Diagnostic sample identity changed")
        prepared[identifier] = inputs
    records = output / "records"
    records.mkdir(exist_ok=True)
    lock_root = default_data_dir()
    lock_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_root / "inference.lock"), timeout=0):
        for index, (condition, identifier) in enumerate(plan["execution_order"]):
            path = records / f"{identifier}.{condition}.json"
            descriptor = request_descriptor(plan, entries[identifier], condition)
            if path.exists():
                row = _read(path)
                validate_record(row, descriptor)
            else:
                print(json.dumps({"stage": "diagnostic", "completed": index, "total": 16,
                                  "condition": condition, "sample_id": identifier}), flush=True)
                row = invoke(prepared[identifier], descriptor)
                _atomic_json(path, row)
            print(json.dumps({"stage": "result", "completed": index + 1, "total": 16,
                              "condition": condition, "prediction": row["prediction"], "status": row["status"],
                              "error_kind": row["error_kind"], "seconds": row["elapsed_seconds"]}), flush=True)
    return report(output)


def report(output):
    """Offline summary; missing, failed, and explicit abstentions stay visible."""
    output = Path(output)
    plan = _read(output / "plan.json")
    validate_plan(plan)
    expected_paths = {f"{s['sample_id']}.{c}.json" for s in plan["samples"] for c in CONDITIONS}
    actual_paths = {p.name for p in (output / "records").glob("*.json")}
    if actual_paths - expected_paths:
        raise ValueError("Unexpected prediction files in diagnostic")
    rows, conditions = [], {}
    for condition in CONDITIONS:
        selected = []
        for entry in plan["samples"]:
            path = output / "records" / f"{entry['sample_id']}.{condition}.json"
            row = {"sample_id": entry["sample_id"], "label": entry["label"], "condition": condition,
                   "status": "missing", "prediction": "abstain", "error_kind": "missing",
                   "has_thinking": False, "done_reason": None}
            if path.exists():
                saved = _read(path)
                validate_record(saved, request_descriptor(plan, entry, condition))
                response = saved.get("response") or {}
                row.update({k: saved[k] for k in ("status", "prediction", "error_kind")})
                row.update(has_thinking=bool((response.get("message") or {}).get("thinking")),
                           done_reason=response.get("done_reason"))
            rows.append(row)
            selected.append(row)
        conditions[condition] = {"expected": 4, "recorded": sum(r["status"] != "missing" for r in selected),
            "valid_final": sum(r["status"] == "ok" for r in selected),
            "answered": sum(r["status"] == "ok" and r["prediction"] != "abstain" for r in selected),
            "correct_of_fixed_four": sum(r["status"] == "ok" and r["prediction"] == r["label"] for r in selected),
            "verdict_counts": dict(Counter(r["prediction"] for r in selected)),
            "error_counts": dict(Counter(r["error_kind"] for r in selected if r["error_kind"]))}
    by_cell = {(r["condition"], r["sample_id"]): r for r in rows}
    comparisons = {}
    for condition in CONDITIONS[1:]:
        pairs = [(by_cell[CONDITIONS[0], s["sample_id"]], by_cell[condition, s["sample_id"]]) for s in plan["samples"]]
        valid = [(a, b) for a, b in pairs if a["status"] == b["status"] == "ok"]
        comparisons[condition] = {"expected_pairs": 4, "paired_valid_final": len(valid),
            "three_state_flips": sum(a["prediction"] != b["prediction"] for a, b in valid),
            "unpaired_or_invalid": 4 - len(valid)}
    result = {"plan_sha256": plan["plan_sha256"], "expected_calls": 16,
              "recorded_calls": sum(c["recorded"] for c in conditions.values()),
              "conditions": conditions, "versus_baseline_short": comparisons, "rows": rows,
              "limitations": ["Post hoc, label-balanced development diagnostic; four samples cannot estimate generalization or statistical significance.",
                  "The short baseline differs from the original evidence-producing 48-call protocol.",
                  "A different verdict or fluent final answer does not prove visual grounding.",
                  "Missing/invalid outputs remain in all fixed denominators; thinking-only JSON is not a valid final answer.",
                  "Checksums detect accidental mixing or corruption, not a maliciously rewritten and re-signed experiment."]}
    _atomic_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/robotbench"))
    parser.add_argument("--output", type=Path, default=Path("data/vlm-diagnostic"))
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_cmd = commands.add_parser("freeze", help="Metadata only; no model inference")
    freeze_cmd.add_argument("--pilot-plan", type=Path, default=Path("data/robot-run/vlm-plan.json"))
    freeze_cmd.add_argument("--qwen", default="qwen3-vl:2b")
    freeze_cmd.add_argument("--minicpm", default="minicpm-v4.6:1b")
    commands.add_parser("run")
    commands.add_parser("report")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".diagnostic.lock"), timeout=0):
        if args.command == "freeze":
            result = freeze(args.data, args.output, args.pilot_plan, qwen=args.qwen, minicpm=args.minicpm)
            print(json.dumps({"plan_sha256": result["plan_sha256"], "samples": [s["sample_id"] for s in result["samples"]], "calls": 16}), flush=True)
        else:
            result = run(args.data, args.output) if args.command == "run" else report(args.output)
            print(json.dumps({k: result[k] for k in ("plan_sha256", "expected_calls", "recorded_calls", "conditions")}), flush=True)


if __name__ == "__main__":
    main()
