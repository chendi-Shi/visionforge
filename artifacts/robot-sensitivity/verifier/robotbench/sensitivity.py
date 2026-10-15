"""Post-hoc coarse-group stress test using already frozen CLIP features.

Create a timestamped plan first, then run it in a separate command. This never
replaces the primary 16-group result and cannot select groups or hyperparameters
from scores. No image encoder or Ollama call is made.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from robotbench.data import load_samples, REVISION, sha256_file as _sha256
from robotbench.metrics import evaluate, compare_paired


GROUPS = {
    "cup_stacking": ["real_stack_cup", "real_stack_cups", "real_stack_cup_spatial"],
    "drawer_storage": ["real_put_item_in_drawer"],
    "open_receptacle_placement": [
        "real_healthiest_banana_in_plate", "real_healthiest_in_plate", "real_put_banana_in_plate",
        "real_put_carrot_in_container", "real_put_carrot_in_plate", "real_put_food_in_box",
        "real_put_fruit_in_box", "real_put_item_spatial_in_box", "real_put_item_spatial_in_plate"],
    "tabletop_pick_tool": ["real_pick_tool", "real_pick_tool_gondola", "real_sport"],
}
REASONS = {
    "cup_stacking": "Merge cup stacking, unstacking, plural naming, and spatial-reference variants.",
    "drawer_storage": "Drawer insertion/removal has a distinct storage setup from open receptacles.",
    "open_receptacle_placement": "Plate, box, container, food selection, and spatial variants have overlapping task texts; keep them together.",
    "tabletop_pick_tool": "Tool/pick variants and sport records concern free tabletop selection/manipulation; sport texts include pick up a fruit and put the ball next to the carrot.",
}
METHODS = ("train_majority", "text_only", "after", "transition")
FIXED_CONFIG = {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000,
                "random_state": 0, "scaler": "StandardScaler fitted inside each train fold", "thread_limit": 2}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cross_validate(samples, features):
    # Offline verification never imports the classifier or image/model stack.
    from robotbench.probe import cross_validate as fit
    return fit(samples, features)


def assign_groups(samples):
    mapping = {}
    for group, members in GROUPS.items():
        for member in members:
            if member in mapping:
                raise ValueError("An original task group has multiple coarse assignments")
            mapping[member] = group
    if set(mapping) != {sample["task_group"] for sample in samples} or len(GROUPS) < 3:
        raise ValueError("The fixed mapping must cover every original group exactly once")
    return [{**sample, "original_task_group": sample["task_group"],
             "task_group": mapping[sample["task_group"]]} for sample in samples]


def _runtime_sources():
    """Collect local Python imports statically; never import or execute a snapshot."""
    source_root = Path(__file__).resolve().parents[1]
    pending, found = ["robotbench.sensitivity"], {}

    def module_path(module):
        base = source_root.joinpath(*module.split("."))
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if path.is_file():
                return path
        return None

    while pending:
        module = pending.pop()
        path = module_path(module)
        if path is None:
            continue
        relative = path.relative_to(source_root).as_posix()
        if relative in found:
            continue
        found[relative] = path
        if "." in module:
            pending.append(module.rsplit(".", 1)[0])
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".") if package else []
                    prefix = parts[:len(parts) - node.level + 1]
                    base = ".".join(prefix + ([base] if base else []))
                if base:
                    pending.append(base)
                    pending.extend(base + "." + alias.name for alias in node.names if alias.name != "*")
    for name in ("requirements.txt", "requirements-search.txt", "requirements-robot.txt"):
        path = source_root / name
        if path.is_file():
            found[name] = path
    return dict(sorted(found.items()))


def _snapshot(output, sources):
    hashes = {}
    for relative, source in sources.items():
        content = source.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        target = output / "runtime" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _sha256(target) != checksum:
            raise ValueError("Runtime snapshot differs; use a new output directory")
        if not target.exists():
            target.write_bytes(content)
        hashes[relative] = checksum
    return hashes


def _inputs(data, feature_path, reference_path):
    samples = load_samples(data)
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = digest([{key: sample[key] for key in ("sample_id", "frames", "subtask_instruction")} for sample in samples])
    if features.get("input_sha256") != expected:
        raise ValueError("Frozen features do not match the dataset")
    hashes = {"inputs_sha256": _sha256(data / "inputs.jsonl"), "labels_sha256": _sha256(data / "labels.jsonl"),
              "features_sha256": _sha256(feature_path), "reference_cv_sha256": _sha256(reference_path)}
    if any(reference.get(key) != hashes[key] for key in ("inputs_sha256", "labels_sha256", "features_sha256")):
        raise ValueError("Primary CV is not bound to these exact data and features")
    if reference.get("config") != FIXED_CONFIG or set(reference.get("predictions", {})) != set(METHODS):
        raise ValueError("Primary experiment methods or fixed hyperparameters differ")
    if (reference.get("code_sha256") != _sha256(Path(__file__).with_name("probe.py"))
            or reference.get("models_code_sha256") != _sha256(Path(__file__).with_name("models.py"))):
        raise ValueError("Probe or feature construction code changed since the primary experiment")
    return samples, features, reference, hashes


def make_plan(data, feature_path, reference_path, output):
    samples, features, _, hashes = _inputs(data, feature_path, reference_path)
    coarse = assign_groups(samples)
    sources = _runtime_sources()
    source_hashes = {relative: _sha256(path) for relative, path in sources.items()}
    specification = {
        "schema_version": 1, "dataset_revision": REVISION, "post_hoc": True,
        "purpose": "One prespecified coarse-group stress test after discovering semantic overlap among base taskvar groups; never replace or select against the primary result.",
        "mapping_basis": "Original taskvar names and full task instruction texts only; mapping is not chosen from label counts or model scores.",
        "groups": GROUPS, "group_reasons": REASONS, "group_counts": dict(sorted(Counter(s["task_group"] for s in coarse).items())),
        "samples": len(samples), "sample_assignments": [{"sample_id": s["sample_id"], "original_task_group": s["original_task_group"],
                                                          "coarse_group": s["task_group"]} for s in coarse],
        "sources": hashes, "runtime_source_sha256": source_hashes, "feature_identity": features["identity"],
        "hyperparameters": FIXED_CONFIG, "methods": list(METHODS),
        "scoring": {"denominator": "all 140 samples for each method, retaining errors and abstentions",
                    "metrics": ["accuracy", "balanced_accuracy", "macro_f1", "coverage", "selective_accuracy", "task_macro_accuracy", "failure_recall"],
                    "bootstrap_iterations": 2000, "bootstrap_seed": 0, "confidence": 0.95, "cluster": "coarse_group",
                    "paired": ["after to transition", "primary to coarse for each of all four methods"],
                    "original_taskvar_point_metrics": True},
        "limits": ["Post-hoc stress test, not preregistered confirmation or a replacement for the primary 16-group result.",
                   "Only four highly unequal groups; cluster intervals are exploratory and support no significance claim.",
                   "All groups still share primitive actions; this does not guarantee action-, semantic-task-, or environment-disjoint generalization.",
                   "No new encoder/VLM inference, no threshold or hyperparameter tuning, and no regrouping after scores."]}
    path = output / "plan.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k not in {"created_at", "plan_sha256"}} != specification:
            raise ValueError("Frozen sensitivity plan changed; choose a new directory")
        if digest({k: v for k, v in previous.items() if k != "plan_sha256"}) != previous.get("plan_sha256"):
            raise ValueError("Plan checksum mismatch")
        _snapshot(output, sources)
        return previous
    specification["created_at"] = datetime.now(timezone.utc).isoformat()
    specification["plan_sha256"] = digest(specification)
    _snapshot(output, sources)
    _atomic_json(path, specification)
    return specification


def _evaluation(plan, samples, coarse, reference, cv, cv_sha256, evaluate_fn=evaluate, paired_fn=compare_paired):
    scoring = plan["scoring"]
    options = {"bootstrap_iterations": scoring["bootstrap_iterations"], "seed": scoring["bootstrap_seed"], "confidence": scoring["confidence"]}
    methods = plan["methods"]
    metrics = {method: evaluate_fn(coarse, cv["predictions"][method], **options) for method in methods}
    return {"schema_version": 1, "post_hoc": True, "plan_sha256": plan["plan_sha256"], "sources": plan["sources"],
            "cv_sha256": cv_sha256, "samples": len(samples), "coarse_groups": len(plan["groups"]), "limits": plan["limits"],
            "metrics": metrics, "failure_recall": {method: metrics[method]["per_class"]["failure"]["recall"] for method in methods},
            "original_taskvar_point_metrics": {method: evaluate_fn(samples, cv["predictions"][method], bootstrap_iterations=0) for method in methods},
            "paired_after_to_transition": paired_fn(coarse, cv["predictions"]["after"], cv["predictions"]["transition"], **options),
            "paired_primary_to_coarse": {method: paired_fn(coarse, reference["predictions"][method], cv["predictions"][method], **options) for method in methods}}


def execute_plan(data, feature_path, reference_path, output):
    if not (output / "plan.json").is_file():
        raise ValueError("Create and inspect plan.json in a separate plan command before fitting")
    plan = make_plan(data, feature_path, reference_path, output)
    samples, features, reference, hashes = _inputs(data, feature_path, reference_path)
    coarse = assign_groups(samples)
    cv_path = output / "cv.json"
    if cv_path.exists():
        cv = json.loads(cv_path.read_text(encoding="utf-8"))
        if cv.get("plan_sha256") != plan["plan_sha256"] or cv.get("sources") != hashes:
            raise ValueError("Saved sensitivity result belongs to a different plan or source")
        prior_metrics = output / "evaluation.json"
        if prior_metrics.exists() and json.loads(prior_metrics.read_text(encoding="utf-8")).get("cv_sha256") != _sha256(cv_path):
            raise ValueError("Saved CV differs from its previous evaluation checksum")
    else:
        began = time.perf_counter()
        cv = cross_validate(coarse, features)
        if cv["config"] != plan["hyperparameters"]:
            raise ValueError("Probe implementation changed its planned hyperparameters")
        cv.update(protocol="post-hoc leave-one-coarse-taskvar-group-out stress test; all primary results retained",
                  plan_sha256=plan["plan_sha256"], sources=hashes,
                  elapsed_seconds=time.perf_counter() - began,
                  completed_at=datetime.now(timezone.utc).isoformat())
        _atomic_json(cv_path, cv)
    _validate_folds(cv, coarse, plan["methods"])
    result = _evaluation(plan, samples, coarse, reference, cv, _sha256(cv_path))
    _atomic_json(output / "evaluation.json", result)
    return result


def _safe_file(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative or ".." in Path(relative).parts:
        raise ValueError("Unsafe bundle path")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Bundle path escaped its directory")
    return result


def _validate_folds(cv, samples, methods):
    by_id = {sample["sample_id"]: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("Duplicate source sample IDs")
    groups = {sample["task_group"] for sample in samples}
    folds = {fold["test_task_group"]: fold for fold in cv["folds"]}
    if len(folds) != len(cv["folds"]) or set(folds) != groups:
        raise ValueError("Fold groups do not exactly cover the planned groups")
    for group, fold in folds.items():
        test = {identifier for identifier, sample in by_id.items() if sample["task_group"] == group}
        hashes = {frame["sha256"] for identifier in test for frame in by_id[identifier]["frames"]}
        purged = {identifier for identifier, sample in by_id.items() if identifier not in test
                  and hashes.intersection(frame["sha256"] for frame in sample["frames"])}
        train = set(by_id) - test - purged
        for key, wanted in (("test_sample_ids", test), ("train_sample_ids", train), ("purged_shared_image_ids", purged)):
            if len(fold[key]) != len(wanted) or set(fold[key]) != wanted:
                raise ValueError(f"Fold {key} differs from the source-derived split")
        if not train or fold["train_class_counts"] != dict(Counter(by_id[i]["label"] for i in train)):
            raise ValueError("Fold training class counts differ from source labels")
        if {by_id[i]["episode_group"] for i in train} & {by_id[i]["episode_group"] for i in test}:
            raise ValueError("Episode identity crosses a train/test fold")
    if set(cv["predictions"]) != set(methods):
        raise ValueError("Prediction methods differ from the plan")
    for method, rows in cv["predictions"].items():
        if len(rows) != len(samples) or {row["sample_id"] for row in rows} != set(by_id):
            raise ValueError("Each method must contain every sample ID exactly once")
        for row in rows:
            sample = by_id[row["sample_id"]]
            fold = folds[sample["task_group"]]
            if (row.get("method") != method or row.get("held_out_group") != sample["task_group"]
                    or row.get("fold_train_n") != len(fold["train_sample_ids"])):
                raise ValueError("Prediction fold metadata differs from the source-derived split")


MAIN_FILES = {"source-inputs.jsonl": "inputs_sha256", "source-labels.jsonl": "labels_sha256",
              "features.json": "features_sha256", "cv.json": "reference_cv_sha256"}
MANIFEST = "bundle-manifest.json"
MANIFEST_CHECKSUM = "bundle-manifest.sha256"


def export_bundle(output, bundle):
    """Copy the actual frozen run; preserve its original computational sources."""
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Frozen plan checksum mismatch")
    sources = {name: output / name for name in ("plan.json", "cv.json", "evaluation.json")}
    for relative, checksum in plan["runtime_source_sha256"].items():
        source = _safe_file(output, "runtime/" + relative)
        if _sha256(source) != checksum:
            raise ValueError("Frozen runtime source checksum mismatch")
        sources["runtime/" + relative] = source
    # The frozen computational snapshot remains untouched. This later verifier
    # is separately hashed, and its standalone entry point imports no ML stack.
    for name in ("sensitivity.py", "data.py", "metrics.py"):
        sources["verifier/robotbench/" + name] = Path(__file__).with_name(name)
    bundle.mkdir(parents=True, exist_ok=True)
    payload = {}

    def save(relative, content):
        target = _safe_file(bundle, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Bundle already contains different bytes; use a new directory")
        if not target.exists():
            target.write_bytes(content)
        payload[relative] = hashlib.sha256(content).hexdigest()

    for relative, source in sources.items():
        save(relative, source.read_bytes())
    save("verifier/robotbench/__init__.py", b"\"\"\"Explicit standalone verification code; runtime/ is never executed.\"\"\"\n")
    entry = '''"""Verify only: no fitting, encoder, Ollama, network, or image assets."""
import argparse
from pathlib import Path
import runpy
import sys
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--main-bundle", type=Path, default=root.parent / "robot-benchmark")
args = parser.parse_args()
sys.path.insert(0, str(root / "verifier"))
sys.dont_write_bytecode = True
sys.argv = [str(root / "verify.py"), "--bundle", str(root), "--main-bundle", str(args.main_bundle), "verify"]
runpy.run_module("robotbench.sensitivity", run_name="__main__")
'''
    save("verify.py", entry.encode("utf-8"))
    readme = """VisionForge: post-hoc four-group sensitivity experiment

This bundle preserves plan.json, cv.json, evaluation.json, and the source files
used during the actual fit. verifier/ is a later, separately
hashed export/verification implementation; runtime/ is the original snapshot.

It depends on the primary robot-benchmark bundle for exact copies of:
source-inputs.jsonl, source-labels.jsonl, features.json, and primary cv.json.
No 510-image duplicate is included, and verification does not require images.

Standalone standard-library verification (Python 3.12+):
python artifacts/robot-sensitivity/verify.py --main-bundle artifacts/robot-benchmark

Repository CLI equivalent:
python -m robotbench.sensitivity --bundle artifacts/robot-sensitivity --main-bundle artifacts/robot-benchmark verify

Verification checks payload and source hashes, planned assignments, exact fold
membership and purging, full prediction coverage, and recalculates every saved
metric and paired interval using the trusted verifier's metrics.py, whose SHA
must match the frozen planned version. It never executes runtime/ source. It does not refit a
classifier or invoke any encoder, model, network, or Ollama service.

This is one post-hoc stress test, not a replacement for the primary result.
Only four unequal clusters exist. Intervals resample fixed out-of-fold outputs;
training folds overlap and model-training variance is not covered. Do not use
these intervals for significance or deployment claims.

bundle-manifest.json hashes every payload file. Its own SHA256 is stored in
bundle-manifest.sha256; those two envelope files are excluded from the payload
map to avoid circular self-hashes. A checksum is not an authenticity signature.
Trust/review the verifier code before executing a downloaded standalone tool;
the repository CLI uses its already installed code and checks the data bundle.
"""
    save("README.txt", readme.encode("utf-8"))
    manifest = {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "files": dict(sorted(payload.items())),
                "requires_primary_bundle": {name: plan["sources"][key] for name, key in MAIN_FILES.items()},
                "excluded_envelope_files": [MANIFEST, MANIFEST_CHECKSUM]}
    manifest_path = bundle / MANIFEST
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing bundle manifest differs")
    _atomic_json(manifest_path, manifest)
    (bundle / MANIFEST_CHECKSUM).write_text(_sha256(manifest_path) + "\n", encoding="ascii")
    return manifest


def verify_bundle(bundle, main_bundle):
    """Read-only full verification with the trusted, version-matched evaluator."""
    manifest_path = bundle / MANIFEST
    if (bundle / MANIFEST_CHECKSUM).read_text(encoding="ascii").strip() != _sha256(manifest_path):
        raise ValueError("Bundle manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = set(manifest["files"]) | {MANIFEST, MANIFEST_CHECKSUM}
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Bundle file inventory differs from its manifest")
    for relative, checksum in manifest["files"].items():
        if _sha256(_safe_file(bundle, relative)) != checksum:
            raise ValueError("Bundle payload checksum mismatch: " + relative)
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    if (digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]
            or manifest["plan_sha256"] != plan["plan_sha256"]):
        raise ValueError("Frozen plan identity mismatch")
    for name, key in MAIN_FILES.items():
        if (manifest["requires_primary_bundle"].get(name) != plan["sources"][key]
                or _sha256(main_bundle / name) != plan["sources"][key]):
            raise ValueError("Primary bundle source checksum mismatch: " + name)
    for relative, checksum in plan["runtime_source_sha256"].items():
        if _sha256(_safe_file(bundle, "runtime/" + relative)) != checksum:
            raise ValueError("Frozen runtime differs from the planned source")
    read_rows = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    inputs, label_rows = read_rows(main_bundle / "source-inputs.jsonl"), read_rows(main_bundle / "source-labels.jsonl")
    labels = {row["sample_id"]: row for row in label_rows}
    if (len(labels) != len(label_rows) or len({s["sample_id"] for s in inputs}) != len(inputs)
            or set(labels) != {s["sample_id"] for s in inputs} or len(inputs) != plan["samples"]):
        raise ValueError("Source sample IDs differ or are duplicated")
    samples = [{**row, **labels[row["sample_id"]]} for row in inputs]
    mapping = {member: group for group, members in plan["groups"].items() for member in members}
    if len(mapping) != sum(map(len, plan["groups"].values())) or set(mapping) != {s["task_group"] for s in samples}:
        raise ValueError("Plan mapping does not exactly cover the original groups")
    coarse = [{**s, "original_task_group": s["task_group"], "task_group": mapping[s["task_group"]]} for s in samples]
    assignments = [{"sample_id": s["sample_id"], "original_task_group": s["original_task_group"], "coarse_group": s["task_group"]} for s in coarse]
    if assignments != plan["sample_assignments"] or dict(Counter(s["task_group"] for s in coarse)) != plan["group_counts"]:
        raise ValueError("Plan assignments or group counts differ from source inputs")
    features = json.loads((main_bundle / "features.json").read_text(encoding="utf-8"))
    feature_input = digest([{key: s[key] for key in ("sample_id", "frames", "subtask_instruction")} for s in samples])
    if (features["input_sha256"] != feature_input or features["identity"] != plan["feature_identity"]
            or len(features["rows"]) != len(samples) or {r["sample_id"] for r in features["rows"]} != set(labels)):
        raise ValueError("Frozen feature input or encoder identity mismatch")
    cv = json.loads((bundle / "cv.json").read_text(encoding="utf-8"))
    reference = json.loads((main_bundle / "cv.json").read_text(encoding="utf-8"))
    if (cv.get("plan_sha256") != plan["plan_sha256"] or cv.get("sources") != plan["sources"]
            or cv.get("config") != plan["hyperparameters"] or reference.get("config") != plan["hyperparameters"]):
        raise ValueError("CV plan, source, or hyperparameter identity mismatch")
    for result in (cv, reference):
        if (result["code_sha256"] != plan["runtime_source_sha256"]["robotbench/probe.py"]
                or result["models_code_sha256"] != plan["runtime_source_sha256"]["robotbench/models.py"]):
            raise ValueError("CV implementation identity mismatch")
    if any(reference.get(key) != plan["sources"][key] for key in ("inputs_sha256", "labels_sha256", "features_sha256")):
        raise ValueError("Primary CV is bound to different data/features")
    _validate_folds(cv, coarse, plan["methods"])
    _validate_folds(reference, samples, plan["methods"])
    trusted_metric_path = Path(evaluate.__code__.co_filename)
    if _sha256(trusted_metric_path) != plan["runtime_source_sha256"]["robotbench/metrics.py"]:
        raise ValueError("Trusted evaluator version differs from the frozen metrics; use the matching trusted revision")
    result = _evaluation(plan, samples, coarse, reference, cv, _sha256(bundle / "cv.json"))
    saved = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    if result != saved:
        raise ValueError("Recomputed metrics or paired intervals differ from the saved evaluation")
    return {"verified": True, "plan_sha256": plan["plan_sha256"], "payload_files": len(manifest["files"]),
            "samples": len(samples), "methods": plan["methods"], "fit_calls": 0, "encoder_calls": 0, "ollama_calls": 0,
            "verification_scope": "Exact frozen predictions, sources, folds and metric recomputation; no classifier refit or image re-encoding."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/robotbench"))
    parser.add_argument("--features", type=Path, default=Path("data/robot-probe/features.json"))
    parser.add_argument("--reference-cv", type=Path, default=Path("data/robot-probe/cv.json"))
    parser.add_argument("--output", type=Path, default=Path("data/robot-sensitivity"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/robot-sensitivity"))
    parser.add_argument("--main-bundle", type=Path, default=Path("artifacts/robot-benchmark"))
    parser.add_argument("command", choices=("plan", "run", "export", "verify"))
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_bundle(args.bundle, args.main_bundle)), flush=True)
        return
    if args.command == "export":
        manifest = export_bundle(args.output, args.bundle)
        print(json.dumps({"bundle": str(args.bundle), "payload_files": len(manifest["files"]),
                          "plan_sha256": manifest["plan_sha256"]}), flush=True)
        return
    from filelock import FileLock
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        if args.command == "plan":
            plan = make_plan(args.data, args.features, args.reference_cv, args.output)
            print(json.dumps({"plan_sha256": plan["plan_sha256"], "created_at": plan["created_at"],
                              "groups": plan["group_counts"], "samples": plan["samples"], "fitted": False}), flush=True)
        else:
            result = execute_plan(args.data, args.features, args.reference_cv, args.output)
            print(json.dumps({"post_hoc": True, "plan_sha256": result["plan_sha256"],
                              "accuracy": {m: result["metrics"][m]["accuracy"] for m in METHODS},
                              "failure_recall": result["failure_recall"]}), flush=True)


if __name__ == "__main__":
    main()
