"""Exploratory instruction-conditioned kernels over frozen CLIP embeddings.

This is a small sklearn baseline, not a new foundation model or evidence of
causal/action understanding. Its design follows inspection of this dataset's
earlier results; nested CV does not make that design choice independent.

Hypothesis: an instruction and a visible effect must be compatible jointly.
The additive control can only add separate instruction/effect functions. The
product kernel can represent their interaction. Both can overfit 140 rows.
All kernels have unit diagonal. Bandwidths and C selection use training rows
only, including in inner folds. Zero visual change is retained, never removed.

Reuse: sklearn SVC(kernel='precomputed'), sklearn pairwise RBF kernels.
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://scikit-learn.org/stable/modules/metrics.html#rbf-kernel
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
"""
from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import warnings

import numpy as np


METHODS = ("kernel_vision", "kernel_additive", "kernel_after", "kernel_relational")
C_VALUES = (0.1, 1.0, 10.0)
LABELS = ("failure", "success")


def design_spec():
    """JSON-serializable protocol for the caller to freeze before any real fit."""
    return {
        "schema_version": 1, "exploratory_after_prior_results": True,
        "hypothesis": "Instruction-conditioned visual-effect similarity predicts subtask outcome better than additive instruction/effect functions.",
        "features": "Existing frozen, pooled before/after/subtask CLIP only; no task group, failure metadata, IDs or labels enter kernels.",
        "blocks": {"text": "unit subtask CLIP", "after": "unit after CLIP", "delta": "unit(after) minus unit(before), retaining zero and magnitude"},
        "bandwidth": "gamma = reciprocal median strictly positive squared pairwise distance among this fit's training rows; gamma=1 for a constant block",
        "kernels": {"kernel_vision": "Ka * Kd", "kernel_additive": "(Kt + Ka * Kd) / 2",
                    "kernel_after": "Kt * Ka", "kernel_relational": "Kt * Ka * Kd"},
        "normalization": "Each block RBF and each combined kernel has unit diagonal; no dataset-wide learned normalization.",
        "C_values": list(C_VALUES), "selection": "Inner leave-one-group-out class-balanced group recall; ties choose smaller C; methods are all reported, never selected against outer scores.",
        "selection_metric": "For each class, mean recall over validation groups containing that class; then mean over the two classes. Missing/failed predictions count wrong. This is not ordinary pooled balanced accuracy.",
        "split": "Outer and inner leave-one-group-out; purge training rows sharing any image SHA or episode with validation/test rows at each level.",
        "classifier": {"implementation": "sklearn.svm.SVC", "kernel": "precomputed", "class_weight": "balanced",
                       "probability": False, "max_iter": 10000, "tol": 0.001, "cache_size_MB": 64, "thread_limit": 2},
        "failures": "Empty training or solver failure yields abstentions without dropping validation/test rows; one-class training uses its sole class.",
        "limits": ["Only two pooled states and subtask text, no trajectory, action vector, object localization or overall-task reference resolution.",
                   "Product kernels change the function class; an improvement is not proof of temporal reasoning or causality.",
                   "Nested selection guards C selection only. Method design followed already-seen dataset results and remains exploratory.",
                   "Identical observations can receive different outputs for different embeddings, but CLIP may not encode the relevant instruction distinction.",
                   "Counterfactual pair labels are never used to select hyperparameters or construct a training feature."]}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _input_digest(samples):
    return _digest([{k: s[k] for k in ("sample_id", "frames", "subtask_instruction")} for s in samples])


class FeatureSpace:
    """Read only an explicit embedding allowlist; labels are not accepted here."""

    def __init__(self, identifiers, features):
        rows = features["rows"]
        by_id = {r["sample_id"]: r for r in rows}
        if len(set(identifiers)) != len(identifiers) or len(by_id) != len(rows) or set(by_id) != set(identifiers):
            raise ValueError("Features must contain every unique sample ID exactly once")
        if any(set(row) != {"sample_id", "before", "after", "text"} for row in rows):
            raise ValueError("Only before/after/text embeddings and their lookup ID may enter the kernel")
        dimension = features["identity"]["dimension"]
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Invalid encoder dimension")
        blocks = {}
        for key in ("before", "after", "text"):
            matrix = np.asarray([by_id[i][key] for i in identifiers], dtype=np.float64)
            if matrix.shape != (len(identifiers), dimension) or not np.isfinite(matrix).all():
                raise ValueError("Invalid embedding shape or non-finite values")
            norms = np.linalg.norm(matrix, axis=1)
            if np.any(norms <= 1e-12):
                raise ValueError("Zero source CLIP embedding")
            blocks[key] = matrix / norms[:, None]
        self.blocks = {"text": blocks["text"], "after": blocks["after"], "delta": blocks["after"] - blocks["before"]}
        # Distances are fixed row-local arithmetic, not a fitted global transform.
        # No test distribution statistics enter the training median below.
        self.distances = {}
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            for key, block in self.blocks.items():
                norms = np.sum(block * block, axis=1)
                distance = np.maximum(norms[:, None] + norms[None, :] - 2 * block @ block.T, 0)
                np.fill_diagonal(distance, 0)
                self.distances[key] = distance

    def kernels(self, train, test):
        if not len(train):
            raise ValueError("Cannot estimate a kernel from empty training rows")
        fit, query, bandwidths = {}, {}, {}
        for key, distances in self.distances.items():
            training = distances[np.ix_(train, train)]
            candidates = training[np.triu_indices(len(train), k=1)]
            positive = candidates[candidates > 1e-12]
            median = float(np.median(positive)) if len(positive) else None
            gamma = 1 / median if median is not None else 1.0
            fit[key] = np.exp(-gamma * training)
            query[key] = np.exp(-gamma * distances[np.ix_(test, train)])
            bandwidths[key] = {"gamma": gamma, "positive_squared_distance_median": median,
                               "positive_training_pairs": len(positive), "constant_block": median is None}
        return fit, query, bandwidths


def combine_kernel(blocks, method):
    text, after, delta = (blocks[k] for k in ("text", "after", "delta"))
    visual = after * delta
    if method == "kernel_vision":
        return visual
    if method == "kernel_additive":
        return (text + visual) / 2
    if method == "kernel_after":
        return text * after
    if method == "kernel_relational":
        return text * visual
    raise ValueError("Unknown kernel method")


def purged_group_split(samples, available, group):
    """Group membership uses metadata only for splitting, never as a feature."""
    test = [int(i) for i in available if samples[i]["task_group"] == group]
    if not test:
        raise ValueError("Requested group is absent")
    pixels = {f["sha256"] for i in test for f in samples[i]["frames"]}
    episodes = {samples[i]["episode_group"] for i in test}
    train, purged = [], []
    for i in available:
        if samples[i]["task_group"] == group:
            continue
        overlap = pixels.intersection(f["sha256"] for f in samples[i]["frames"])
        same_episode = samples[i]["episode_group"] in episodes
        if overlap or same_episode:
            purged.append({"sample_id": samples[i]["sample_id"], "shared_image": bool(overlap), "shared_episode": same_episode})
        else:
            train.append(int(i))
    return train, test, purged


def class_balanced_group_recall(samples, indices, predictions):
    """Equal class weight; equal group weight within each observed class."""
    if len(indices) != len(predictions) or not indices:
        raise ValueError("One prediction is required per validation row")
    per_class = {}
    for label in LABELS:
        groups = sorted({samples[i]["task_group"] for i in indices if samples[i]["label"] == label})
        recalls = []
        for group in groups:
            correct = [prediction == label for i, prediction in zip(indices, predictions)
                       if samples[i]["label"] == label and samples[i]["task_group"] == group]
            recalls.append(sum(correct) / len(correct))
        if recalls:
            per_class[label] = sum(recalls) / len(recalls)
    if set(per_class) != set(LABELS):
        return None  # Undefined selection target; fixed C is used, never test labels.
    return sum(per_class.values()) / len(LABELS)


def _fit_predict(train_labels, training, query, c_value):
    """No hidden calibration CV; margin values are explicitly uncalibrated."""
    counts = Counter(train_labels)
    if not counts:
        return ["abstain"] * len(query), [None] * len(query), {"status": "empty_training", "warnings": []}
    if len(counts) == 1:
        return [next(iter(counts))] * len(query), [None] * len(query), {"status": "single_class", "warnings": []}
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    classifier = SVC(kernel="precomputed", C=c_value, class_weight="balanced", probability=False,
                     max_iter=10000, tol=0.001, cache_size=64, random_state=0)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            classifier.fit(training, train_labels)
            if classifier.fit_status_ != 0:
                return ["abstain"] * len(query), [None] * len(query), {"status": "solver_failed", "warnings": [str(w.message) for w in caught]}
            predictions = classifier.predict(query).tolist()
            margin = classifier.decision_function(query)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            return ["abstain"] * len(query), [None] * len(query), {"status": "fit_error", "error": f"{type(exc).__name__}: {exc}", "warnings": [str(w.message) for w in caught]}
        if list(classifier.classes_) != list(LABELS):
            raise AssertionError("Unexpected class order for signed success margins")
    return predictions, margin.tolist(), {"status": "ok", "warnings": [str(w.message) for w in caught]}


def _checked(samples, features):
    if not samples or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Nonempty unique sample IDs required")
    for sample in samples:
        if sample.get("label") not in LABELS or not sample.get("task_group") or not sample.get("episode_group"):
            raise ValueError("Valid labels and split identities required")
        if not sample.get("frames") or any(not re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "")) for f in sample["frames"]):
            raise ValueError("Valid image identities are required for leakage purging")
    if features.get("input_sha256") != _input_digest(samples):
        raise ValueError("Frozen feature source does not match these inputs")
    if len({s["task_group"] for s in samples}) < 3:
        raise ValueError("Nested leave-group-out needs at least three groups")


def nested_cross_validate(samples, features, *, methods=METHODS, c_values=C_VALUES, outer_groups=None, progress=None):
    """Prototype runner. The caller must freeze a new protocol before real use.

    ``outer_groups`` supports a predeclared subset/single-fold integrity test;
    returning incomplete OOF coverage is explicit in the output, never hidden.
    """
    _checked(samples, features)
    methods, c_values = tuple(methods), tuple(sorted(c_values))
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Invalid or repeated kernel methods")
    if not c_values or len(set(c_values)) != len(c_values) or any(not np.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError("C candidates must be finite, distinct and positive")
    identifiers = [s["sample_id"] for s in samples]
    space = FeatureSpace(identifiers, features)
    groups = sorted({s["task_group"] for s in samples})
    selected_groups = groups if outer_groups is None else list(outer_groups)
    if not selected_groups or len(set(selected_groups)) != len(selected_groups) or not set(selected_groups) <= set(groups):
        raise ValueError("Invalid outer groups")
    output = {m: [] for m in methods}
    folds = []
    for group in selected_groups:
        train, test, purged = purged_group_split(samples, range(len(samples)), group)
        inner_groups = sorted({samples[i]["task_group"] for i in train})
        selection = {m: {c: {"indices": [], "predictions": [], "fit_statuses": []} for c in c_values} for m in methods}
        inner_audits = []
        for inner_group in inner_groups:
            fit, validation, inner_purged = purged_group_split(samples, train, inner_group)
            if fit:
                fit_blocks, query_blocks, bandwidths = space.kernels(fit, validation)
            else:
                bandwidths = {}
            audit = {"validation_group": inner_group, "train_sample_ids": [identifiers[i] for i in fit],
                     "validation_sample_ids": [identifiers[i] for i in validation], "purged": inner_purged,
                     "bandwidths": bandwidths, "train_class_counts": dict(Counter(samples[i]["label"] for i in fit))}
            inner_audits.append(audit)
            for method in methods:
                training = combine_kernel(fit_blocks, method) if fit else np.empty((0, 0))
                query = combine_kernel(query_blocks, method) if fit else np.empty((len(validation), 0))
                for c_value in c_values:
                    predictions, _, status = _fit_predict([samples[i]["label"] for i in fit], training, query, c_value)
                    candidate = selection[method][c_value]
                    candidate["indices"].extend(validation)
                    candidate["predictions"].extend(predictions)
                    candidate["fit_statuses"].append({"validation_group": inner_group, **status})
        fitted, queried, outer_bandwidths = space.kernels(train, test) if train else ({}, {}, {})
        fold = {"test_task_group": group, "train_sample_ids": [identifiers[i] for i in train],
                "test_sample_ids": [identifiers[i] for i in test], "purged": purged, "inner_folds": inner_audits,
                "bandwidths": outer_bandwidths, "train_class_counts": dict(Counter(samples[i]["label"] for i in train)), "methods": {}}
        for method in methods:
            candidates = []
            for c_value, candidate in selection[method].items():
                if len(candidate["indices"]) != len(train) or set(candidate["indices"]) != set(train):
                    raise AssertionError("Inner validation must cover every outer training row exactly once")
                score = class_balanced_group_recall(samples, candidate["indices"], candidate["predictions"]) if train else None
                candidates.append({"C": c_value, "class_balanced_group_recall": score,
                                   "validation_sample_ids": [identifiers[i] for i in candidate["indices"]],
                                   "predictions": candidate["predictions"], "fit_statuses": candidate["fit_statuses"]})
            valid = [candidate for candidate in candidates if candidate["class_balanced_group_recall"] is not None]
            if valid and len(inner_groups) >= 2:
                chosen = max(valid, key=lambda item: (round(item["class_balanced_group_recall"], 12), -item["C"]))["C"]
                selection_status = "nested_selected"
            else:
                chosen = min(c_values, key=lambda c: (abs(np.log10(c)), c))
                selection_status = "undefined_inner_target_fixed_nearest_C_1"
            training = combine_kernel(fitted, method) if train else np.empty((0, 0))
            query = combine_kernel(queried, method) if train else np.empty((len(test), 0))
            predictions, margins, status = _fit_predict([samples[i]["label"] for i in train], training, query, chosen)
            fold["methods"][method] = {"selected_C": chosen, "selection_status": selection_status, "candidates": candidates, "outer_fit": status}
            output[method].extend({"sample_id": identifiers[i], "method": method, "prediction": prediction,
                                   "status": "ok" if prediction in LABELS else "failed", "error": None if prediction in LABELS else status["status"],
                                   "margin_success": margin, "score_note": "Uncalibrated signed SVM margin; not a probability",
                                   "held_out_group": group, "fold_train_n": len(train), "selected_C": chosen}
                                  for i, prediction, margin in zip(test, predictions, margins))
        folds.append(fold)
        if progress is not None:
            progress({"completed_outer_folds": len(folds), "total_outer_folds": len(selected_groups), "held_out_group": group})
    return {"protocol": design_spec(), "executed_methods": list(methods), "executed_C_values": list(c_values),
            "samples": len(samples), "evaluated_samples": sum(len(f["test_sample_ids"]) for f in folds),
            "complete_outer_coverage": set(selected_groups) == set(groups), "input_sha256": features["input_sha256"],
            "features_sha256": _digest(features), "feature_identity": features["identity"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "packages": {p: version(p) for p in ("scikit-learn", "numpy", "threadpoolctl")}, "folds": folds, "predictions": output}
