"""Failure-verification metrics with fixed denominators and task-group uncertainty.

The label is the outcome of a robot subtask, not a failure onset timestamp.
Missing, malformed and failed predictions abstain; they never disappear from
headline denominators. Bootstrap intervals resample whole task families and are
descriptive, especially when only a few families are evaluated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random


LABELS = ("success", "failure")
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "coverage", "selective_accuracy", "task_macro_accuracy")
FORBIDDEN_INPUT_KEYS = frozenset({"label", "labels", "failure_category", "failure_mode", "execution_reward",
    "gt_failure_reason", "gt_failure_step", "ground_truth", "groundtruth", "answer", "answers", "conversations",
    "task_group", "episode_group", "split", "source_filename"})


def _checked_samples(samples):
    rows, seen = [], set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each sample must be an object.")
        identifier = sample.get("sample_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("sample_id must be nonempty and unique.")
        if sample.get("label") not in LABELS:
            raise ValueError(f"Invalid ground-truth label for {identifier}.")
        if not isinstance(sample.get("task_group"), str) or not sample["task_group"].strip():
            raise ValueError(f"Missing task_group for {identifier}.")
        seen.add(identifier)
        rows.append(sample)
    if not rows:
        raise ValueError("Cannot evaluate an empty sample set.")
    return rows


def _predictions(samples, predictions):
    expected = {sample["sample_id"] for sample in samples}
    indexed = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Each prediction must be an object with sample_id.")
        identifier = prediction.get("sample_id")
        if identifier not in expected:
            raise ValueError(f"Prediction ID is not in the selected evaluation set: {identifier}")
        if identifier in indexed:
            raise ValueError(f"Duplicate prediction ID: {identifier}")
        indexed[identifier] = prediction
    rows = []
    for sample in samples:
        prediction = indexed.get(sample["sample_id"])
        if prediction is None:
            label, reason = "abstain", "missing_prediction"
        elif prediction.get("status", "ok") not in {"ok", "completed"}:
            label, reason = "abstain", "failed_prediction"
        elif prediction.get("prediction") in LABELS:
            label, reason = prediction["prediction"], None
        elif prediction.get("prediction") == "abstain":
            label, reason = "abstain", "explicit_abstention"
        else:
            label, reason = "abstain", "invalid_prediction"
        rows.append({"sample_id": sample["sample_id"], "task_group": sample["task_group"],
                     "label": sample["label"], "prediction": label, "abstention_reason": reason,
                     "failure_category": sample.get("failure_category") or "unspecified"})
    return rows


def _point(rows):
    count = len(rows)
    correct = sum(row["prediction"] == row["label"] for row in rows)
    answered = sum(row["prediction"] in LABELS for row in rows)
    matrix = {label: {pred: 0 for pred in (*LABELS, "abstain")} for label in LABELS}
    for row in rows:
        matrix[row["label"]][row["prediction"]] += 1
    per_class = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        false_negative = support - true_positive  # Includes abstentions.
        false_positive = sum(matrix[other][label] for other in LABELS if other != label)
        denominator = 2 * true_positive + false_positive + false_negative
        per_class[label] = {"support": support, "true_positive": true_positive,
            "false_positive": false_positive, "false_negative": false_negative,
            "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
            "recall": true_positive / support if support else None,
            "f1": 2 * true_positive / denominator if denominator else 0.0}
    per_task = defaultdict(list)
    for row in rows:
        per_task[row["task_group"]].append(row)
    task_accuracy = [sum(row["prediction"] == row["label"] for row in group) / len(group) for group in per_task.values()]
    recalls = [per_class[label]["recall"] for label in LABELS]
    return {"n": count, "answered": answered, "abstained": count - answered, "correct": correct,
            "accuracy": correct / count if count else None,
            "balanced_accuracy": sum(recalls) / 2 if all(value is not None for value in recalls) else None,
            "macro_f1": sum(per_class[label]["f1"] for label in LABELS) / 2,
            "coverage": answered / count if count else None,
            "selective_accuracy": correct / answered if answered else None,
            "task_macro_accuracy": sum(task_accuracy) / len(task_accuracy) if task_accuracy else None,
            "confusion_matrix": matrix, "per_class": per_class,
            "abstention_reasons": dict(sorted(Counter(row["abstention_reason"] for row in rows if row["abstention_reason"]).items()))}


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _bootstrap_options(iterations, confidence):
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 0 <= iterations <= 100000:
        raise ValueError("bootstrap_iterations must be an integer between 0 and 100000.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")


def _bootstrap(rows, *, iterations, seed, confidence, paired=None):
    _bootstrap_options(iterations, confidence)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["task_group"]].append(index)
    names = sorted(groups)
    warning = ("Fewer than 10 independent task groups: percentile intervals are exploratory and may be unstable. "
               "They do not establish generalization to new environments or robots.") if len(names) < 10 else (
               "Task-group percentile intervals describe uncertainty within this benchmark; they do not certify deployment performance.")
    output = {"method": "paired_task_cluster_percentile" if paired is not None else "task_cluster_percentile",
              "group_field": "task_group", "n_groups": len(names), "iterations": iterations, "seed": seed,
              "confidence": confidence, "warning": warning, "metrics": {}}
    if iterations == 0 or len(names) < 2:
        output["status"] = "disabled" if iterations == 0 else "insufficient_task_groups"
        for key in METRIC_KEYS:
            output["metrics"][key] = {"lower": None, "upper": None, "valid_iterations": 0, "undefined_iterations": 0}
        return output
    values = {key: [] for key in METRIC_KEYS}
    generator = random.Random(seed)
    for _ in range(iterations):
        selected = [generator.choice(names) for _ in names]
        # Keep duplicate sampled clusters distinct for task-macro averaging.
        expanded, expanded_b = [], []
        for occurrence, name in enumerate(selected):
            for index in groups[name]:
                expanded.append({**rows[index], "task_group": str(occurrence)})
                if paired is not None:
                    expanded_b.append({**paired[index], "task_group": str(occurrence)})
        point_a = _point(expanded)
        point_b = _point(expanded_b) if paired is not None else None
        for key in METRIC_KEYS:
            value = point_a[key]
            if point_b is not None:
                value = point_b[key] - value if value is not None and point_b[key] is not None else None
            if value is not None:
                values[key].append(value)
    alpha = (1 - confidence) / 2
    output["status"] = "computed"
    for key in METRIC_KEYS:
        valid = values[key]
        output["metrics"][key] = {"lower": _quantile(valid, alpha) if valid else None,
                                  "upper": _quantile(valid, 1 - alpha) if valid else None,
                                  "valid_iterations": len(valid), "undefined_iterations": iterations - len(valid)}
    return output


def evaluate(samples, predictions, *, bootstrap_iterations=2000, seed=0, confidence=0.95):
    """Evaluate every selected sample; ground truth never enters model inference."""
    samples = _checked_samples(samples)
    rows = _predictions(samples, predictions)
    result = _point(rows)
    result["definitions"] = {
        "positive_label": "failure", "denominator": "All selected samples, including missing/failed/invalid predictions.",
        "macro_f1": "Mean F1 over the fixed labels success and failure; abstentions are false negatives for their true class; undefined class F1 is zero.",
        "balanced_accuracy": "Mean of success recall and failure recall; undefined when either ground-truth class is absent.",
        "selective_accuracy": "Correct / answered only; secondary to accuracy and must be read together with coverage.",
        "task_macro_accuracy": "Equal-weight mean of per-task-group accuracies.",
        "scope": "Before/after subtask outcome verification; no temporal onset localization is measured."}
    result["per_task"] = {group: _point([row for row in rows if row["task_group"] == group]) for group in sorted({row["task_group"] for row in rows})}
    categories = sorted({row["failure_category"] for row in rows if row["label"] == "failure"})
    result["per_failure_category"] = {}
    for category in categories:
        subset = [row for row in rows if row["label"] == "failure" and row["failure_category"] == category]
        detected = sum(row["prediction"] == "failure" for row in subset)
        answered = sum(row["prediction"] in LABELS for row in subset)
        result["per_failure_category"][category] = {"n": len(subset), "detected": detected,
            "failure_recall": detected / len(subset), "answered": answered, "abstained": len(subset) - answered,
            "coverage": answered / len(subset)}
    result["bootstrap"] = _bootstrap(rows, iterations=bootstrap_iterations, seed=seed, confidence=confidence)
    result["sample_ids"] = [row["sample_id"] for row in rows]
    return result


def compare_paired(samples, predictions_a, predictions_b, *, bootstrap_iterations=2000, seed=0, confidence=0.95):
    """Compare B minus A on identical samples and identical bootstrap draws."""
    samples = _checked_samples(samples)
    rows_a, rows_b = _predictions(samples, predictions_a), _predictions(samples, predictions_b)
    a, b = _point(rows_a), _point(rows_b)
    table = Counter()
    jointly_answered = 0
    for left, right in zip(rows_a, rows_b):
        correct_a, correct_b = left["prediction"] == left["label"], right["prediction"] == right["label"]
        table["both_correct" if correct_a and correct_b else "a_only_correct" if correct_a else "b_only_correct" if correct_b else "both_wrong_or_abstain"] += 1
        jointly_answered += left["prediction"] in LABELS and right["prediction"] in LABELS
    return {"n": len(samples), "direction": "B minus A", "a": a, "b": b,
            "delta": {key: b[key] - a[key] if a[key] is not None and b[key] is not None else None for key in METRIC_KEYS},
            "paired_outcomes": {key: table[key] for key in ("both_correct", "a_only_correct", "b_only_correct", "both_wrong_or_abstain")},
            "jointly_answered": jointly_answered,
            "denominator_note": "Every selected ID is matched; missing outputs are abstentions, not dropped pairs. No independent-sample significance claim is made.",
            "bootstrap_delta": _bootstrap(rows_a, paired=rows_b, iterations=bootstrap_iterations, seed=seed, confidence=confidence),
            "sample_ids": [sample["sample_id"] for sample in samples]}


def _forbidden_fields(value, prefix="model_input"):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_INPUT_KEYS:
                found.append(path)
            found.extend(_forbidden_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_fields(child, f"{prefix}[{index}]"))
    return found


def audit_split(samples, *, required_splits=("dev", "test")):
    """Audit frozen manifest groups, duplicate image hashes and inference fields.

    This checks the supplied manifest, not unknown model pretraining membership.
    It cannot prove that arbitrary natural-language prompt text contains no hints.
    """
    samples = _checked_samples(samples)
    issues, split_counts, grouped = [], Counter(), {field: defaultdict(set) for field in ("task_group", "episode_group", "image_sha256")}
    for sample in samples:
        identifier, split = sample["sample_id"], sample.get("split")
        if not isinstance(split, str) or not split.strip():
            issues.append({"kind": "missing_split", "sample_id": identifier})
            continue
        split_counts[split] += 1
        grouped["task_group"][sample["task_group"]].add(split)
        episode = sample.get("episode_group")
        if not isinstance(episode, str) or not episode.strip():
            issues.append({"kind": "missing_episode_group", "sample_id": identifier})
        else:
            grouped["episode_group"][episode].add(split)
        model_input = sample.get("model_input")
        if not isinstance(model_input, dict):
            issues.append({"kind": "missing_model_input", "sample_id": identifier})
            continue
        for field in _forbidden_fields(model_input):
            issues.append({"kind": "forbidden_inference_field", "sample_id": identifier, "field": field})
        frames = model_input.get("frames")
        if not isinstance(frames, list) or not frames:
            issues.append({"kind": "missing_inference_frames", "sample_id": identifier})
            frames = []
        for frame in frames:
            digest = frame.get("sha256") if isinstance(frame, dict) else None
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                issues.append({"kind": "missing_or_invalid_image_sha256", "sample_id": identifier})
            else:
                grouped["image_sha256"][digest].add(split)
    for split in required_splits:
        if not split_counts[split]:
            issues.append({"kind": "missing_required_split", "split": split})
    for field, groups in grouped.items():
        for group, splits in sorted(groups.items()):
            if len(splits) > 1:
                issues.append({"kind": "cross_split_overlap", "field": field, "value": group, "splits": sorted(splits)})
    return {"ok": not issues, "n": len(samples), "split_counts": dict(sorted(split_counts.items())),
            "n_task_groups": len(grouped["task_group"]), "n_episode_groups": len(grouped["episode_group"]),
            "issues": issues,
            "limits": "Audits supplied dev/test task, episode and exact-image identity separation, not undisclosed model pretraining or semantic prompt leakage."}
