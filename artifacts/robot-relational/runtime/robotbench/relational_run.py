"""Freeze, run and export the exploratory nested relational-kernel experiment.

python -m robotbench.relational_run plan
python -m robotbench.relational_run run
python -m robotbench.relational_run export

No image encoding, download, Ollama inference or changes to previous results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from robotbench.data import load_samples, sha256_file
from robotbench.diagnostics import counterfactual_pairs
from robotbench.metrics import evaluate, compare_paired


METHODS = ("kernel_vision", "kernel_additive", "kernel_after", "kernel_relational")
SOURCES = {"inputs": "data/robotbench/inputs.jsonl", "labels": "data/robotbench/labels.jsonl",
           "features": "data/robot-probe/features.json", "primary_cv": "data/robot-probe/cv.json",
           "coarse_plan": "data/robot-sensitivity/plan.json", "coarse_cv": "data/robot-sensitivity/cv.json"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex[:12] + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def frozen_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Refusing to overwrite different frozen bytes: " + str(path))
    if not path.exists():
        path.write_bytes(content)


def runtime_sources():
    root = Path(__file__).resolve().parents[1]
    names = ["robotbench/relational.py", "robotbench/relational_run.py", "robotbench/data.py",
             "robotbench/metrics.py", "robotbench/diagnostics.py", "requirements-robot.txt", "requirements-robot-lock.txt"]
    return {name: root / name for name in names if (root / name).is_file()}


def coarse_samples(samples, mapping):
    reverse = {member: group for group, members in mapping.items() for member in members}
    if len(reverse) != sum(map(len, mapping.values())) or set(reverse) != {s["task_group"] for s in samples}:
        raise ValueError("Coarse mapping must cover the original groups exactly once")
    return [{**s, "original_task_group": s["task_group"], "task_group": reverse[s["task_group"]]} for s in samples]


def make_plan(output):
    from robotbench.relational import design_spec, _checked, FeatureSpace
    samples = load_samples(Path(SOURCES["inputs"]).parent)
    features = read(SOURCES["features"])
    _checked(samples, features)
    FeatureSpace([s["sample_id"] for s in samples], features)  # validation, no fit
    coarse_plan = read(SOURCES["coarse_plan"])
    hashes = {key: sha256_file(Path(path)) for key, path in SOURCES.items()}
    expected = {"inputs_sha256": hashes["inputs"], "labels_sha256": hashes["labels"], "features_sha256": hashes["features"]}
    if any(coarse_plan["sources"][key] != value for key, value in expected.items()):
        raise ValueError("Coarse-group source identities do not match")
    for key, value in expected.items():
        if read(SOURCES["primary_cv"])[key] != value:
            raise ValueError("Primary predictions belong to different inputs/features")
    if read(SOURCES["coarse_cv"])["plan_sha256"] != coarse_plan["plan_sha256"]:
        raise ValueError("Coarse reference predictions belong to another plan")
    coarse = coarse_samples(samples, coarse_plan["groups"])
    sources = runtime_sources()
    specification = {
        "schema_version": 1, "exploratory": True,
        "prior_exposure": "All 140 labels, original 16-group linear probe scores, four-group sensitivity scores, and the 3/17 counterfactual result were inspected before this method design. These are not independent confirmation results.",
        "design": design_spec(), "sources": {k: {"path": SOURCES[k], "sha256": v} for k, v in hashes.items()},
        "runtime_source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "feature_identity": features["identity"], "samples": len(samples),
        "protocols": {"base_taskvar": {"groups": dict(sorted(Counter(s["task_group"] for s in samples).items()))},
                      "coarse_semantic": {"groups": dict(sorted(Counter(s["task_group"] for s in coarse).items())), "mapping": coarse_plan["groups"]}},
        "scoring": {"bootstrap_iterations": 2000, "seed": 0, "confidence": 0.95, "denominator": "Every one of the 140 samples for every method, including abstentions.",
                    "headline": ["accuracy", "balanced_accuracy", "macro_f1", "coverage", "failure_recall"],
                    "paired": ["kernel_vision to kernel_relational", "kernel_additive to kernel_relational", "kernel_after to kernel_relational", "legacy_after to kernel_relational", "legacy_transition to kernel_relational"],
                    "sensitivity": "base_taskvar to coarse_semantic for each of all four methods; bootstrap by the four coarse groups",
                    "counterfactual": "All eligible exact ordered same-observation / different-subtask / opposite-label pairs; post-prediction diagnostic only, no selection or thresholds."},
        "limits": ["One fixed exploratory method design and its fixed ablations; no outer-score tuning or selective reporting.",
                   "Only 16 base taskvar groups or four unequal coarse groups, sharing robot/environment/primitive actions.",
                   "Bootstrap intervals condition on fixed OOF predictions, ignore overlapping-fold training variance, and do not justify significance claims.",
                   "Kernel and legacy linear comparisons differ in function class and C selection; they do not isolate temporal reasoning.",
                   "Counterfactual pairs share observations and samples, so their 17-pair count is not an independent sample size."]}
    plan_path = output / "plan.json"
    if plan_path.exists():
        previous = read(plan_path)
        if {k: v for k, v in previous.items() if k not in {"created_at", "plan_sha256"}} != specification:
            raise ValueError("The existing plan is frozen; choose a new output directory")
        check_plan(previous)
        return previous
    specification["created_at"] = datetime.now(timezone.utc).isoformat()
    specification["plan_sha256"] = digest(specification)
    for name, path in sources.items():
        frozen_write(output / "runtime" / name, path.read_bytes())
    atomic(plan_path, specification)
    return specification


def check_plan(plan):
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Frozen plan checksum mismatch")


def evaluate_run(plan, samples, results, references):
    options = {k: plan["scoring"][k] for k in ("bootstrap_iterations", "seed", "confidence")}
    coarse = coarse_samples(samples, plan["protocols"]["coarse_semantic"]["mapping"])
    summaries = {}
    for protocol, rows in (("base_taskvar", samples), ("coarse_semantic", coarse)):
        cv = results[protocol]
        if not cv["complete_outer_coverage"] or cv["evaluated_samples"] != len(samples):
            raise ValueError("Incomplete outer predictions cannot enter the full experiment report")
        if set(cv["predictions"]) != set(METHODS):
            raise ValueError("All four prespecified methods must be present")
        for predictions in cv["predictions"].values():
            if len(predictions) != len(rows) or {p["sample_id"] for p in predictions} != {s["sample_id"] for s in rows}:
                raise ValueError("Each method must contain every sample exactly once")
        metrics = {m: evaluate(rows, cv["predictions"][m], **options) for m in METHODS}
        comparisons = {m: cv["predictions"][m] for m in METHODS if m != "kernel_relational"}
        comparisons.update({"legacy_after": references[protocol]["predictions"]["after"],
                            "legacy_transition": references[protocol]["predictions"]["transition"]})
        summaries[protocol] = {"metrics": metrics,
                               "failure_recall": {m: metrics[m]["per_class"]["failure"]["recall"] for m in METHODS},
                               "paired_to_relational": {m: compare_paired(rows, values, cv["predictions"]["kernel_relational"], **options) for m, values in comparisons.items()},
                               "counterfactual": {m: counterfactual_pairs(rows, values) for m, values in {**cv["predictions"], **{k: v for k, v in comparisons.items() if k.startswith("legacy_")}}.items()}}
    return {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "exploratory": True,
            "sources": plan["sources"], "cv_content_sha256": {k: digest(v) for k, v in results.items()},
            "protocols": summaries,
            "paired_base_to_coarse": {m: compare_paired(coarse, results["base_taskvar"]["predictions"][m], results["coarse_semantic"]["predictions"][m], **options) for m in METHODS},
            "limits": plan["limits"]}


def run_plan(output):
    from robotbench.relational import nested_cross_validate
    if not (output / "plan.json").is_file():
        raise ValueError("Run the separate plan command before any fitting")
    plan = make_plan(output)  # fail if source data/code/spec changed
    for name, checksum in plan["runtime_source_sha256"].items():
        if sha256_file(output / "runtime" / name) != checksum:
            raise ValueError("Frozen runtime was modified")
    samples = load_samples(Path(SOURCES["inputs"]).parent)
    features = read(SOURCES["features"])
    coarse = coarse_samples(samples, plan["protocols"]["coarse_semantic"]["mapping"])
    results = {}
    receipts_path = output / "receipts.json"
    receipts = read(receipts_path) if receipts_path.exists() else {"plan_sha256": plan["plan_sha256"], "files": {}}
    if receipts["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Saved receipts refer to another plan")
    for protocol, rows in (("base_taskvar", samples), ("coarse_semantic", coarse)):
        filename = "cv-" + protocol + ".json"
        path = output / filename
        if path.exists():
            if receipts["files"].get(filename) != sha256_file(path):
                raise ValueError("Cached CV checksum differs or has no completion receipt")
            cv = read(path)
            if cv["plan_sha256"] != plan["plan_sha256"]:
                raise ValueError("Saved CV plan differs")
        else:
            began = time.perf_counter()
            progress = lambda event: print(json.dumps({"stage": "nested_cv", "protocol": protocol, **event}), flush=True)
            cv = nested_cross_validate(rows, features, progress=progress)
            cv.update(plan_sha256=plan["plan_sha256"], grouping_protocol=protocol,
                      completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - began)
            atomic(path, cv)
            receipts["files"][filename] = sha256_file(path)
            atomic(receipts_path, receipts)
        results[protocol] = cv
    references = {"base_taskvar": read(SOURCES["primary_cv"]), "coarse_semantic": read(SOURCES["coarse_cv"])}
    summary = evaluate_run(plan, samples, results, references)
    if (output / "evaluation.json").exists() and read(output / "evaluation.json") != summary:
        raise ValueError("Saved evaluation differs from the complete recomputation")
    atomic(output / "evaluation.json", summary)
    receipts["files"]["evaluation.json"] = sha256_file(output / "evaluation.json")
    atomic(receipts_path, receipts)
    return summary


def export_bundle(output, bundle):
    plan, receipts = read(output / "plan.json"), read(output / "receipts.json")
    check_plan(plan)
    if receipts["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Completion receipt plan mismatch")
    expected = {"cv-base_taskvar.json", "cv-coarse_semantic.json", "evaluation.json"}
    if set(receipts["files"]) != expected:
        raise ValueError("Both protocols and the final evaluation must be complete before export")
    sources = {"plan.json": output / "plan.json", "receipts.json": output / "receipts.json"}
    for name, checksum in receipts["files"].items():
        if sha256_file(output / name) != checksum:
            raise ValueError("Completed output checksum mismatch")
        sources[name] = output / name
    for name, checksum in plan["runtime_source_sha256"].items():
        path = output / "runtime" / name
        if sha256_file(path) != checksum:
            raise ValueError("Frozen runtime checksum mismatch")
        sources["runtime/" + name] = path
    for key in ("primary_cv", "coarse_cv", "coarse_plan"):
        source = plan["sources"][key]
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("Legacy reference bytes changed")
        sources["references/" + key + ".json"] = Path(source["path"])
    payload = {}
    for name, source in sources.items():
        content = source.read_bytes()
        frozen_write(bundle / name, content)
        payload[name] = hashlib.sha256(content).hexdigest()
    notes = """Exploratory instruction-conditioned action-effect kernel experiment

Both 16-base-taskvar and the previously frozen four-coarse-group protocols are
reported for all four methods. This follows prior inspection of all 140 rows.
It is not independent confirmation, a trajectory benchmark, or a new VLM.

plan.json was saved before fitting. runtime/ preserves the actual source.
cv-*.json include every inner split, validation prediction, chosen C, kernel
bandwidth and outer prediction. evaluation.json retains every fixed comparison
and the same-observation/different-instruction diagnostic. Legacy reference
predictions are copied byte-for-byte under references/.

No images or feature matrix are duplicated. Source inputs, labels and features
are the exact files from artifacts/robot-benchmark/source-inputs.jsonl,
source-labels.jsonl and features.json; expected SHA256 values are in plan.json
and the manifest. Source dataset/encoder licenses remain in the main bundle.

Repository reproduction (with the declared packages and prepared source data):
python -m robotbench.relational_run --output data/robot-relational-new plan
python -m robotbench.relational_run --output data/robot-relational-new run
python -m robotbench.relational_run --output data/robot-relational-new --bundle artifacts/robot-relational-new export

The frozen runtime snapshot must be used to reproduce this exact code revision;
current code changes are rejected when resuming an existing plan. Inspect and
trust Python code before executing it. Checksums are not authenticity signatures.

Bootstrap intervals resample fixed OOF outputs and do not include overlapping
fold training variance. Four unequal coarse groups permit only exploratory
intervals. Identical-image counterfactual pairs are dependent observations.
"""
    content = notes.encode()
    frozen_write(bundle / "README.txt", content)
    payload["README.txt"] = hashlib.sha256(content).hexdigest()
    manifest = {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "files": dict(sorted(payload.items())),
                "requires_primary_bundle": {"source-inputs.jsonl": plan["sources"]["inputs"]["sha256"],
                                            "source-labels.jsonl": plan["sources"]["labels"]["sha256"],
                                            "features.json": plan["sources"]["features"]["sha256"]},
                "excluded_envelope_files": ["bundle-manifest.json", "bundle-manifest.sha256"]}
    if (bundle / "bundle-manifest.json").exists() and read(bundle / "bundle-manifest.json") != manifest:
        raise ValueError("Existing bundle manifest differs")
    atomic(bundle / "bundle-manifest.json", manifest)
    frozen_write(bundle / "bundle-manifest.sha256", (sha256_file(bundle / "bundle-manifest.json") + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "export"))
    parser.add_argument("--output", type=Path, default=Path("data/robot-relational"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/robot-relational"))
    args = parser.parse_args()
    if args.command == "export":
        result = export_bundle(args.output, args.bundle)
        print(json.dumps({"bundle": str(args.bundle), "files": len(result["files"]), "plan_sha256": result["plan_sha256"]}), flush=True)
        return
    from filelock import FileLock
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        if args.command == "plan":
            result = make_plan(args.output)
            print(json.dumps({"plan_sha256": result["plan_sha256"], "created_at": result["created_at"], "fitted": False, "samples": result["samples"]}), flush=True)
        else:
            result = run_plan(args.output)
            print(json.dumps({"plan_sha256": result["plan_sha256"], "accuracy": {p: {m: s["metrics"][m]["accuracy"] for m in METHODS} for p, s in result["protocols"].items()}}), flush=True)


if __name__ == "__main__":
    main()
