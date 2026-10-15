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

from filelock import FileLock

from robotbench.data import load_samples, REVISION
from robotbench.metrics import evaluate, compare_paired
from robotbench.probe import cross_validate, _hash
from video_media import _atomic_json, _sha256


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
    expected = _hash([{key: sample[key] for key in ("sample_id", "frames", "subtask_instruction")} for sample in samples])
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
    scoring = plan["scoring"]
    options = {"bootstrap_iterations": scoring["bootstrap_iterations"], "seed": scoring["bootstrap_seed"], "confidence": scoring["confidence"]}
    metrics = {method: evaluate(coarse, cv["predictions"][method], **options) for method in METHODS}
    result = {"schema_version": 1, "post_hoc": True, "plan_sha256": plan["plan_sha256"], "sources": hashes,
              "cv_sha256": _sha256(cv_path), "samples": len(samples), "coarse_groups": len(GROUPS), "limits": plan["limits"],
              "metrics": metrics, "failure_recall": {method: metrics[method]["per_class"]["failure"]["recall"] for method in METHODS},
              "original_taskvar_point_metrics": {method: evaluate(samples, cv["predictions"][method], bootstrap_iterations=0) for method in METHODS},
              "paired_after_to_transition": compare_paired(coarse, cv["predictions"]["after"], cv["predictions"]["transition"], **options),
              "paired_primary_to_coarse": {method: compare_paired(coarse, reference["predictions"][method], cv["predictions"][method], **options) for method in METHODS}}
    _atomic_json(output / "evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/robotbench"))
    parser.add_argument("--features", type=Path, default=Path("data/robot-probe/features.json"))
    parser.add_argument("--reference-cv", type=Path, default=Path("data/robot-probe/cv.json"))
    parser.add_argument("--output", type=Path, default=Path("data/robot-sensitivity"))
    parser.add_argument("command", choices=("plan", "run"))
    args = parser.parse_args()
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
