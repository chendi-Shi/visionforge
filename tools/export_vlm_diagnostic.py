"""Export and independently verify the fixed VLM diagnostic, without inference.

The verify command uses only Python's standard library and never imports the
preserved runtime. Inspect this verifier before executing a downloaded copy.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

CONDITIONS = ("baseline_short", "enum_reversed", "no_prefill", "minicpm")
SEED = "visionforge-vlm-protocol-diagnostic-v1"
LIMITATIONS = ["Post hoc, label-balanced development diagnostic; four samples cannot estimate generalization or statistical significance.",
    "The short baseline differs from the original evidence-producing 48-call protocol.",
    "A different verdict or fluent final answer does not prove visual grounding.",
    "Missing/invalid outputs remain in all fixed denominators; thinking-only JSON is not a valid final answer.",
    "Checksums detect accidental mixing or corruption, not a maliciously rewritten and re-signed experiment."]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_seal(value, key):
    if value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError("Invalid " + key)


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source row")
    return {row["sample_id"]: row for row in rows}


def safe_file(root, name):
    pure = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or pure.is_absolute() or ".." in pure.parts:
        raise ValueError("Unsafe bundle path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError("Missing or unsafe bundle file: " + name)
    return path


def parse_response(response):
    if response is None:
        return "failed", "abstain", "transport_error"
    message = response.get("message") or {}
    content, thinking = str(message.get("content") or ""), str(message.get("thinking") or "")
    if response.get("done_reason") in {"length", "max_tokens", "max_length"}:
        return "failed", "abstain", "truncated"
    if not content.strip():
        return "failed", "abstain", "thinking_without_final" if thinking.strip() else "empty_final"
    try:
        value = json.loads(content)
        if not isinstance(value, dict) or set(value) != {"prediction"} or value["prediction"] not in {"success", "failure", "abstain"}:
            raise ValueError("Bad schema")
    except (ValueError, TypeError):
        return "failed", "abstain", "invalid_schema"
    return "ok", value["prediction"], None


def verify(bundle):
    root = Path(bundle).resolve()
    manifest = read(root / "manifest.json")
    if (root / "manifest.sha256").read_text().strip() != sha(root / "manifest.json"):
        raise ValueError("Manifest envelope checksum mismatch")
    if manifest.get("excluded_envelope_files") != ["manifest.json", "manifest.sha256"]:
        raise ValueError("Unexpected manifest exclusions")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json", "manifest.sha256"}:
        raise ValueError("Bundle file inventory mismatch")
    for name, checksum in manifest["files"].items():
        path = safe_file(root, name)
        if sha(path) != checksum:
            raise ValueError("Payload checksum mismatch: " + name)
        if path.suffix in {".json", ".jsonl", ".py", ".txt", ".md"}:
            content = path.read_text(encoding="utf-8")
            if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", content) or re.search(r"X-Amz-Signature=[0-9a-fA-F]{24,}", content):
                raise ValueError("Private absolute path or signed URL in public payload: " + name)
    plan, saved_report = read(root / "plan.json"), read(root / "report.json")
    check_seal(plan, "plan_sha256")
    if manifest["plan_sha256"] != plan["plan_sha256"] or plan["version"] != 1:
        raise ValueError("Plan identity mismatch")
    required = {"plan.json", "report.json", "pilot-plan.json", "source-inputs.jsonl", "source-labels.jsonl",
                "verify.py", "README.txt", "LICENSE-DATA.txt", "ATTRIBUTION.txt"}
    required |= {"runtime/" + name for name in plan["code_sha256"]}
    for name, checksum in plan["code_sha256"].items():
        if sha(safe_file(root, "runtime/" + name)) != checksum:
            raise ValueError("Frozen runtime source changed")
    for file, key in (("source-inputs.jsonl", "inputs_sha256"), ("source-labels.jsonl", "labels_sha256"), ("pilot-plan.json", "pilot_plan_sha256")):
        if sha(safe_file(root, file)) != plan[key]:
            raise ValueError("Source identity mismatch")
    inputs, labels = read_rows(root / "source-inputs.jsonl"), read_rows(root / "source-labels.jsonl")
    if set(inputs) != set(labels):
        raise ValueError("Source labels and inputs have different IDs")
    pilot = read(root / "pilot-plan.json")
    if pilot["inputs_sha256"] != plan["inputs_sha256"] or pilot["dataset_revision"] != plan["dataset_revision"]:
        raise ValueError("Original pilot identity mismatch")
    ids = pilot["sample_ids"]
    if len(ids) != len(set(ids)) or any(i not in labels for i in ids):
        raise ValueError("Invalid pilot membership")
    selected = []
    for label in ("success", "failure"):
        candidates = sorted((i for i in ids if labels[i]["label"] == label), key=lambda i: digest([SEED, i]))
        if len(candidates) < 2:
            raise ValueError("Missing label stratum")
        selected.extend(candidates[:2])
    if plan["seed"] != SEED or [s["sample_id"] for s in plan["samples"]] != selected:
        raise ValueError("Diagnostic selection no longer matches the fixed hash rule")
    if plan["execution_order"] != [[c, i] for c in CONDITIONS for i in selected] or set(plan["conditions"]) != set(CONDITIONS):
        raise ValueError("Diagnostic execution matrix differs")
    rows, conditions = [], {}
    for condition in CONDITIONS:
        cells = []
        spec = plan["conditions"][condition]
        for sample in plan["samples"]:
            identifier = sample["sample_id"]
            source = inputs[identifier]
            clean = {"task": source["task_instruction"], "subtask": source["subtask_instruction"],
                     "frames": [{k: v for k, v in f.items() if k != "path"} for f in source["frames"]]}
            if sample["input"] != clean or sample["label"] != labels[identifier]["label"]:
                raise ValueError("Frozen sample differs from source input or author label")
            panel = f"panels/{identifier}.png"
            required.add(panel)
            if sha(safe_file(root, panel)) != sample["panel"]["panel_sha256"]:
                raise ValueError("Frozen panel checksum mismatch")
            for frame in source["frames"]:
                image = "images/" + frame["sha256"] + ".png"
                required.add(image)
                if sha(safe_file(root, image)) != frame["sha256"]:
                    raise ValueError("Source image checksum mismatch")
            filename = f"records/{identifier}.{condition}.json"
            required.add(filename)
            raw = read(safe_file(root, filename))
            check_seal(raw, "record_sha256")
            expected = {"plan_sha256": plan["plan_sha256"], "sample_id": identifier, "condition": condition,
                "input": sample["input"], "panel": sample["panel"], "protocol": spec,
                "model_identity": plan["models"][spec["model_key"]], "server_version": plan["server_version"]}
            if raw["request"] != expected or raw["request_sha256"] != digest(expected):
                raise ValueError("Raw prediction belongs to a different request")
            status, prediction, error = parse_response(raw["response"])
            if (raw["status"], raw["prediction"], raw["error_kind"]) != (status, prediction, error):
                raise ValueError("Prediction fields disagree with raw model response")
            response = raw["response"] or {}
            cell = {"sample_id": identifier, "label": labels[identifier]["label"], "condition": condition,
                    "status": status, "prediction": prediction, "error_kind": error,
                    "has_thinking": bool((response.get("message") or {}).get("thinking")), "done_reason": response.get("done_reason")}
            rows.append(cell)
            cells.append(cell)
        conditions[condition] = {"expected": 4, "recorded": 4, "valid_final": sum(r["status"] == "ok" for r in cells),
            "answered": sum(r["status"] == "ok" and r["prediction"] != "abstain" for r in cells),
            "correct_of_fixed_four": sum(r["status"] == "ok" and r["prediction"] == r["label"] for r in cells),
            "verdict_counts": dict(Counter(r["prediction"] for r in cells)),
            "error_counts": dict(Counter(r["error_kind"] for r in cells if r["error_kind"]))}
    if required != set(manifest["files"]):
        raise ValueError("Required diagnostic inventory differs from manifest")
    by_cell = {(r["condition"], r["sample_id"]): r for r in rows}
    comparisons = {}
    for condition in CONDITIONS[1:]:
        pairs = [(by_cell[CONDITIONS[0], i], by_cell[condition, i]) for i in selected]
        valid = [(a, b) for a, b in pairs if a["status"] == b["status"] == "ok"]
        comparisons[condition] = {"expected_pairs": 4, "paired_valid_final": len(valid),
            "three_state_flips": sum(a["prediction"] != b["prediction"] for a, b in valid), "unpaired_or_invalid": 4 - len(valid)}
    calculated = {"plan_sha256": plan["plan_sha256"], "expected_calls": 16, "recorded_calls": 16,
                  "conditions": conditions, "versus_baseline_short": comparisons, "rows": rows, "limitations": LIMITATIONS}
    if saved_report != calculated:
        raise ValueError("Saved report differs from independent raw-response recomputation")
    return {"verified": True, "plan_sha256": plan["plan_sha256"], "files": len(required), "conditions": conditions}


def export(data, run, pilot, bundle):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from robotbench import vlm_diagnostic as diagnostic
    from robotbench.data import load_samples, model_input
    from robotbench.models import prepare_panel
    data, run, pilot, bundle = map(Path, (data, run, pilot, bundle))
    plan = read(run / "plan.json")
    diagnostic.validate_plan(plan)
    if plan["code_sha256"] != diagnostic.source_hashes():
        raise ValueError("Current source differs from actual diagnostic protocol")
    if sha(pilot) != plan["pilot_plan_sha256"]:
        raise ValueError("Original pilot bytes changed")
    if any(sha(data / (key + ".jsonl")) != plan[key + "_sha256"] for key in ("inputs", "labels")):
        raise ValueError("Source dataset identity changed")
    if bundle.exists():
        existing = verify(bundle)
        if existing["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Different existing bundle; choose a new destination")
        original_files = ["plan.json", "report.json", *[f"records/{s['sample_id']}.{c}.json"
                          for s in plan["samples"] for c in CONDITIONS]]
        if any(sha(run / name) != sha(bundle / name) for name in original_files):
            raise ValueError("Existing bundle differs from current raw run bytes")
        return existing
    samples = {s["sample_id"]: s for s in load_samples(data)}
    bundle.parent.mkdir(parents=True, exist_ok=True)
    # Short stage name avoids Windows path-length failures; failed exports can
    # be retried without leaving a partially published destination directory.
    stage = Path(tempfile.mkdtemp(prefix=".vd-", dir=bundle.parent)).resolve()
    intended_parent = bundle.parent.resolve()
    if stage.parent != intended_parent or not stage.name.startswith(".vd-"):
        raise ValueError("Temporary export directory escaped the intended parent")
    try:
        def copy(source, relative):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in ("plan.json", "report.json"):
            copy(run / name, name)
        copy(pilot, "pilot-plan.json")
        for key in ("inputs", "labels"):
            copy(data / (key + ".jsonl"), "source-" + key + ".jsonl")
        for name, checksum in plan["code_sha256"].items():
            source = repo / name if name == "core.py" else repo / "robotbench" / name
            if sha(source) != checksum:
                raise ValueError("Frozen source unavailable")
            copy(source, "runtime/" + name)
        for entry in plan["samples"]:
            identifier = entry["sample_id"]
            inputs = model_input(samples[identifier], data, condition="ordered", view="all")
            panel, audit = prepare_panel(inputs)
            if audit != entry["panel"] or diagnostic._clean(inputs) != entry["input"]:
                raise ValueError("Reconstructed frozen panel differs")
            (stage / "panels").mkdir(exist_ok=True)
            (stage / "panels" / (identifier + ".png")).write_bytes(panel)
            for frame in inputs["frames"]:
                copy(frame["path"], "images/" + frame["sha256"] + ".png")
            for condition in CONDITIONS:
                name = identifier + "." + condition + ".json"
                copy(run / "records" / name, "records/" + name)
        copy(repo / "licenses" / "Apache-2.0.txt", "LICENSE-DATA.txt")
        copy(Path(__file__), "verify.py")
        (stage / "ATTRIBUTION.txt").write_text("Guardian UR5-Fail observations and author labels\nDataset: paulpacaud/Guardian-FailCoT-OOD-datasets\nRevision: " + plan["dataset_revision"] + "\nhttps://huggingface.co/datasets/paulpacaud/Guardian-FailCoT-OOD-datasets\nData license: Apache-2.0 (LICENSE-DATA.txt). Source images unchanged; panels compose these images with phase/camera headers.\n", encoding="utf-8")
        (stage / "README.txt").write_text("Post hoc local VLM protocol diagnostic: four fixed development samples, sixteen calls.\n\nRun offline verification with Python's standard library:\n  python verify.py verify .\nThe verifier does not execute runtime/ or require Ollama/ML packages. Inspect code before running it.\n\nplan.json binds exact inputs, author labels, image/panel hashes, code, model digests and server version.\nrecords/ preserves every complete raw response including thinking-only/truncated results.\nreport.json is independently recomputed from those responses, with all four samples per condition.\npanels/ contains exact-byte reconstructions verified against the pre-inference plan; images/ contains source PNGs.\nFull source input/label rows and the old pilot plan permit reproducing the stable label-stratified selection.\nSource dataset files are hash-pinned copies, not a claim of authentic independent authorship.\nChecksums detect corruption/mixing, not an attacker rewriting and re-signing the entire bundle.\n\nbaseline_short differs from the original 48-call evidence schema and token budget.\nThis is not a generalization estimate or a model ranking. No confidence/significance claim is made.\nAll runtime files are retained for audit only. No old benchmark evidence was edited.\n", encoding="utf-8")
        files = {p.relative_to(stage).as_posix(): sha(p) for p in stage.rglob("*") if p.is_file()}
        manifest = {"version": 1, "plan_sha256": plan["plan_sha256"], "files": dict(sorted(files.items())),
                    "excluded_envelope_files": ["manifest.json", "manifest.sha256"]}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(sha(stage / "manifest.json") + "\n", encoding="utf-8")
        result = verify(stage)
        if bundle.exists():
            verify(bundle)
            if read(bundle / "manifest.json") != manifest:
                raise ValueError("Different existing bundle; choose a new destination")
        else:
            stage.rename(bundle)
        return result
    finally:
        if stage.exists():
            if stage.resolve().parent != intended_parent or not stage.name.startswith(".vd-"):
                raise ValueError("Refusing to remove an unexpected temporary directory")
            shutil.rmtree(stage)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify")
    check.add_argument("bundle", type=Path)
    create = sub.add_parser("export")
    create.add_argument("--data", type=Path, default=Path("data/robotbench"))
    create.add_argument("--run", type=Path, default=Path("data/vlm-diagnostic"))
    create.add_argument("--pilot", type=Path, default=Path("data/robot-run/vlm-plan.json"))
    create.add_argument("--bundle", type=Path, default=Path("artifacts/robot-vlm-diagnostic"))
    args = parser.parse_args()
    result = verify(args.bundle) if args.command == "verify" else export(args.data, args.run, args.pilot, args.bundle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
