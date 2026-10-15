"""Standard-library offline failure detection/localization evaluation.

Ground truth is accepted here only, after predictions exist. Every selected
episode remains in every denominator. Sampler oracle coverage is a timestamp
resolution diagnostic; it does not establish visual observability of failure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random

LABELS = ("success", "failure")
TOLERANCES = (1.0, 2.0, 5.0)
CONTROL_METHODS = ("clock_midpoint", "state_peak")


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _ratio(a, b):
    return a / b if b else None


def episode_summaries(episodes):
    """Small portable input summary: measured time, never fps-derived time."""
    result, seen = [], set()
    for row in episodes:
        identifier = row.get("sample_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Episode sample IDs must be nonempty and unique")
        times = row.get("timestamp")
        if (not isinstance(times, (list, tuple)) or len(times) < 2 or any(not _finite(t) or t < 0 for t in times)
                or any(a >= b for a, b in zip(times, times[1:]))):
            raise ValueError("Measured timestamps must be finite, nonnegative and strictly increasing")
        if not isinstance(row.get("task_id"), str) or not row["task_id"]:
            raise ValueError("Episode task_id required for cluster evaluation")
        duration = row.get("duration_seconds", times[-1])
        if not _finite(duration) or duration < times[-1]:
            # The adapter's video duration and the measured clock can differ by
            # fractions of a frame; the measured timeline always remains valid.
            duration = times[-1]
        result.append({"sample_id": identifier, "task_id": row["task_id"], "split": row.get("split"),
                       "timestamp": list(times), "duration_seconds": duration})
        seen.add(identifier)
    if not result:
        raise ValueError("Cannot evaluate an empty episode set")
    return result


def _checked_labels(labels, episodes):
    by_id = {row["sample_id"]: row for row in episode_summaries(episodes)}
    seen, rows = set(), []
    for label in labels:
        identifier = label.get("sample_id")
        if identifier in seen or identifier not in by_id:
            raise ValueError("Label IDs must uniquely match the selected episodes")
        episode = by_id[identifier]
        if label.get("label") not in LABELS or label.get("task_id") != episode["task_id"]:
            raise ValueError("Invalid label or mismatched task identity")
        if label.get("split") != episode["split"]:
            raise ValueError("Labels and input split identities differ")
        onset = label.get("onset_seconds")
        if label["label"] == "failure":
            if not _finite(onset) or not 0 <= onset <= episode["duration_seconds"]:
                raise ValueError("Failure labels require an in-range author onset")
        elif onset is not None:
            raise ValueError("Success labels must not invent an onset")
        rows.append({**episode, "label": label["label"], "true_onset_seconds": onset})
        seen.add(identifier)
    if seen != set(by_id):
        raise ValueError("Missing labels cannot be silently excluded")
    return rows


def _index_records(samples, records):
    expected = {row["sample_id"] for row in samples}
    indexed = {}
    for row in records:
        identifier = row.get("sample_id")
        if identifier not in expected or identifier in indexed:
            raise ValueError("Unknown or duplicate prediction sample ID")
        indexed[identifier] = row
    return indexed


def _selection_times(record, timeline):
    selection = record.get("selection") or {}
    times, indices = selection.get("timestamps"), selection.get("indices")
    if (not isinstance(times, list) or not isinstance(indices, list) or not times or len(times) != len(indices)
            or selection.get("budget", len(times)) != len(times)):
        return None
    if (any(type(i) is not int or not 0 <= i < len(timeline) for i in indices)
            or any(a >= b for a, b in zip(indices, indices[1:]))):
        return None
    if any(not _finite(t) or abs(t - timeline[i]) > 1e-7 for i, t in zip(indices, times)):
        return None
    return times


def _normalize(samples, records):
    indexed = _index_records(samples, records)
    rows = []
    for sample in samples:
        record = indexed.get(sample["sample_id"])
        prediction, reason = "abstain", None
        if record is None:
            reason = "missing_prediction"
        elif record.get("status", "ok") not in {"ok", "completed"}:
            reason = "failed_prediction"
        elif record.get("prediction") in LABELS:
            prediction = record["prediction"]
        elif record.get("prediction") == "abstain":
            reason = "explicit_abstention"
        else:
            reason = "invalid_prediction"
        selection = _selection_times(record, sample["timestamp"]) if record else None
        onset, onset_issue = None, None
        if prediction == "failure":
            proposed = record.get("onset_seconds")
            if not _finite(proposed) or not sample["timestamp"][0] <= proposed <= sample["timestamp"][-1]:
                onset_issue = "missing_or_out_of_range_onset"
            elif record.get("method") in CONTROL_METHODS and record.get("control") is True:
                onset = float(proposed)
            else:
                frame = record.get("frame_id")
                if selection is None or type(frame) is not int or not 0 <= frame < len(selection):
                    onset_issue = "invalid_evidence_frame"
                elif abs(proposed - selection[frame]) > 1e-7:
                    onset_issue = "onset_does_not_match_measured_evidence_time"
                else:
                    onset = float(proposed)
        failure = sample["label"] == "failure"
        error = abs(onset - sample["true_onset_seconds"]) if failure and onset is not None else None
        nearest = min(abs(t - sample["true_onset_seconds"]) for t in selection) if failure and selection else None
        rows.append({"sample_id": sample["sample_id"], "task_id": sample["task_id"], "label": sample["label"],
                     "true_onset_seconds": sample["true_onset_seconds"], "prediction": prediction,
                     "onset_seconds": onset, "absolute_onset_error_seconds": error,
                     "abstention_reason": reason, "onset_issue": onset_issue,
                     "selection_valid": selection is not None, "nearest_sample_error_seconds": nearest})
    return rows


def _point(rows, tolerances):
    n = len(rows)
    confusion = {label: {prediction: 0 for prediction in (*LABELS, "abstain")} for label in LABELS}
    for row in rows:
        confusion[row["label"]][row["prediction"]] += 1
    supports = {label: sum(confusion[label].values()) for label in LABELS}
    recalls = {label: _ratio(confusion[label][label], supports[label]) for label in LABELS}
    f1 = {}
    for label in LABELS:
        tp = confusion[label][label]
        fn = supports[label] - tp
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        f1[label] = _ratio(2 * tp, 2 * tp + fn + fp) or 0.0
    failures = [row for row in rows if row["label"] == "failure"]
    errors = [row["absolute_onset_error_seconds"] for row in failures if row["absolute_onset_error_seconds"] is not None]
    oracle = [row["nearest_sample_error_seconds"] for row in failures if row["nearest_sample_error_seconds"] is not None]
    answered = sum(row["prediction"] in LABELS for row in rows)
    correct = sum(row["prediction"] == row["label"] for row in rows)
    metrics = {"n": n, "answered": answered, "abstained": n - answered, "correct": correct,
               "accuracy": _ratio(correct, n), "coverage": _ratio(answered, n), "abstention_rate": _ratio(n - answered, n),
               "selective_accuracy": _ratio(correct, answered), "macro_f1": sum(f1.values()) / 2,
               "balanced_accuracy": sum(recalls.values()) / 2 if all(x is not None for x in recalls.values()) else None,
               "failure_recall": recalls["failure"], "success_false_alarm_rate": _ratio(confusion["success"]["failure"], supports["success"]),
               "failure_n": len(failures), "success_n": supports["success"], "detected_failure_n": confusion["failure"]["failure"],
               "valid_localized_failure_n": len(errors), "valid_onset_coverage": _ratio(len(errors), len(failures)),
               "conditional_onset_mae_seconds": _ratio(sum(errors), len(errors)),
               "sampler_valid_failure_n": len(oracle), "sampler_valid_coverage": _ratio(len(oracle), len(failures)),
               "confusion_matrix": confusion,
               "abstention_reasons": dict(sorted(Counter(r["abstention_reason"] for r in rows if r["abstention_reason"]).items())),
               "invalid_onset_reasons": dict(sorted(Counter(r["onset_issue"] for r in rows if r["onset_issue"]).items()))}
    for tolerance in tolerances:
        suffix = f"{tolerance:g}s"
        metrics["hit_at_" + suffix] = _ratio(sum(error <= tolerance for error in errors), len(failures))
        metrics["sampler_oracle_at_" + suffix] = _ratio(sum(error <= tolerance for error in oracle), len(failures))
    return metrics


def _keys(tolerances):
    return ("accuracy", "balanced_accuracy", "macro_f1", "coverage", "abstention_rate", "failure_recall", "success_false_alarm_rate",
            "valid_onset_coverage", "conditional_onset_mae_seconds", "sampler_valid_coverage",
            *("hit_at_" + f"{t:g}s" for t in tolerances), *("sampler_oracle_at_" + f"{t:g}s" for t in tolerances))


def _quantile(values, q):
    values = sorted(values)
    point = (len(values) - 1) * q
    low, high = math.floor(point), math.ceil(point)
    return values[low] + (values[high] - values[low]) * (point - low)


def _bootstrap(rows, tolerances, iterations, seed, confidence, paired=None):
    if type(iterations) is not int or not 0 <= iterations <= 100000 or not _finite(confidence) or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["task_id"]].append(index)
    names = sorted(groups)
    keys = _keys(tolerances)
    result = {"method": "paired_task_cluster_percentile" if paired is not None else "task_cluster_percentile",
              "group_field": "task_id", "n_groups": len(names), "iterations": iterations, "seed": seed, "confidence": confidence,
              "warning": "Small label-stratified local subset; task clusters may be few. Intervals are descriptive, conditional on fixed predictions, and do not establish deployment or significance.",
              "metrics": {}}
    if iterations == 0 or len(names) < 2:
        result["status"] = "disabled" if iterations == 0 else "insufficient_task_groups"
        result["metrics"] = {key: {"lower": None, "upper": None, "valid_iterations": 0, "undefined_iterations": 0} for key in keys}
        return result
    generator, values = random.Random(seed), {key: [] for key in keys}
    for _ in range(iterations):
        indices = [i for group in (generator.choice(names) for _ in names) for i in groups[group]]
        a = _point([rows[i] for i in indices], tolerances)
        b = _point([paired[i] for i in indices], tolerances) if paired is not None else None
        for key in keys:
            value = a[key]
            if b is not None:
                value = b[key] - value if value is not None and b[key] is not None else None
            if value is not None:
                values[key].append(value)
    result["status"] = "computed"
    alpha = (1 - confidence) / 2
    for key in keys:
        valid = values[key]
        result["metrics"][key] = {"lower": _quantile(valid, alpha) if valid else None,
                                  "upper": _quantile(valid, 1 - alpha) if valid else None,
                                  "valid_iterations": len(valid), "undefined_iterations": iterations - len(valid)}
    return result


def _tolerances(values):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(not _finite(v) or v <= 0 for v in values):
        raise ValueError("Localization tolerances must be distinct positive numbers")
    return tuple(sorted(float(v) for v in values))


def evaluate(labels, records, episodes, *, tolerances=TOLERANCES, bootstrap_iterations=2000, seed=1337, confidence=0.95):
    tolerances = _tolerances(tolerances)
    rows = _normalize(_checked_labels(labels, episodes), records)
    result = _point(rows, tolerances)
    result["sample_ids"] = [row["sample_id"] for row in rows]
    result["rows"] = rows
    result["per_task"] = {task: _point([r for r in rows if r["task_id"] == task], tolerances) for task in sorted({r["task_id"] for r in rows})}
    result["bootstrap"] = _bootstrap(rows, tolerances, bootstrap_iterations, seed, confidence)
    result["definitions"] = {
        "denominator": "All selected labeled episodes, including missing, failed and malformed predictions.",
        "hit_at_tolerance": "Correctly predicted failure AND valid evidence onset within tolerance / all ground-truth failures.",
        "conditional_onset_mae_seconds": "Mean absolute onset error among ground-truth failures predicted failure with a valid onset; always read with valid_onset_coverage.",
        "valid_onset_coverage": "Ground-truth failures detected with a valid onset / all ground-truth failures.",
        "success_false_alarm_rate": "Predicted failure / all true successes; abstentions remain in the denominator.",
        "sampler_oracle": "Nearest selected measured timestamp within tolerance / all true failures, regardless of model verdict. Missing/invalid selections count uncovered. This measures sampling resolution, not visual observability.",
        "invalid_onset": "A binary failure verdict with an invalid onset remains a detected failure but cannot count as localized or a hit.",
        "scope": "Offline retrospective diagnosis using whole-recording sampling, not streaming detection latency."}
    return result


def compare_paired(labels, records_a, records_b, episodes, *, tolerances=TOLERANCES, bootstrap_iterations=2000, seed=1337, confidence=0.95):
    tolerances = _tolerances(tolerances)
    samples = _checked_labels(labels, episodes)
    a, b = _normalize(samples, records_a), _normalize(samples, records_b)
    point_a, point_b = _point(a, tolerances), _point(b, tolerances)
    keys = _keys(tolerances)
    joint_errors = [(left["absolute_onset_error_seconds"], right["absolute_onset_error_seconds"]) for left, right in zip(a, b)
                    if left["absolute_onset_error_seconds"] is not None and right["absolute_onset_error_seconds"] is not None]
    return {"n": len(samples), "direction": "B minus A", "sample_ids": [s["sample_id"] for s in samples],
            "a": point_a, "b": point_b,
            "delta": {k: point_b[k] - point_a[k] if point_a[k] is not None and point_b[k] is not None else None for k in keys},
            "joint_valid_localized_failure_n": len(joint_errors),
            "joint_conditional_mae_delta_seconds": _ratio(sum(right - left for left, right in joint_errors), len(joint_errors)),
            "bootstrap_delta": _bootstrap(a, tolerances, bootstrap_iterations, seed, confidence, paired=b),
            "limits": "All IDs are paired including missing records. Conditional MAE deltas can involve different detected subsets; the joint-valid delta and coverage are also reported. Lower MAE is better, higher hit rate is better."}


def controls(episodes, sampler_records):
    """Label-free all-failure controls; never pass labels to this function.

    Clock uses the midpoint of first/last measured timestamps. State peak uses
    the earliest maximum of existing whole-recording event scores (not a learned
    failure score). Both deliberately predict failure on every available episode.
    """
    summaries = episode_summaries(episodes)
    by_id = defaultdict(list)
    expected = {row["sample_id"] for row in summaries}
    for record in sampler_records:
        if record.get("sample_id") not in expected:
            raise ValueError("Control source contains an unknown episode")
        by_id[record["sample_id"]].append(record)
    result = {method: [] for method in CONTROL_METHODS}
    for episode in summaries:
        identifier, times = episode["sample_id"], episode["timestamp"]
        common = {"sample_id": identifier, "task_id": episode["task_id"], "control": True,
                  "status": "ok", "prediction": "failure", "error": None}
        result["clock_midpoint"].append({**common, "method": "clock_midpoint", "onset_seconds": (times[0] + times[-1]) / 2,
                                         "basis": "Label-free whole-recording measured-clock midpoint; always predicts failure."})
        candidates = []
        for record in by_id[identifier]:
            scores = (record.get("selection") or {}).get("scores")
            if isinstance(scores, list) and len(scores) == len(times) and all(_finite(s) and s >= 0 for s in scores):
                candidates.append(scores)
        if candidates and any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError("Sampler records disagree on the same state-event score series")
        if not candidates:
            result["state_peak"].append({**common, "method": "state_peak", "status": "failed", "prediction": "abstain",
                                         "onset_seconds": None, "error": "missing_or_invalid_state_scores"})
            continue
        index = max(range(len(times)), key=lambda i: (candidates[0][i], -i))
        result["state_peak"].append({**common, "method": "state_peak", "onset_seconds": times[index], "source_frame_index": index,
                                     "peak_score": candidates[0][index], "all_scores_zero": max(candidates[0]) == 0,
                                     "basis": "Label-free largest whole-recording motion-change score, earliest tie; always predicts failure."})
    return result


def evaluate_all(labels, records, episodes, *, methods=("uniform", "state_event"), bootstrap_iterations=2000, seed=1337):
    allowed = set(methods)
    predictions = {method: [] for method in methods}
    for record in records:
        if record.get("method") not in allowed:
            raise ValueError("Unexpected method in the selected sampler record set")
        predictions[record["method"]].append(record)
    predictions.update(controls(episodes, records))
    options = {"bootstrap_iterations": bootstrap_iterations, "seed": seed}
    evaluated = {method: evaluate(labels, rows, episodes, **options) for method, rows in predictions.items()}
    pairs = {"uniform_to_state_event": compare_paired(labels, predictions["uniform"], predictions["state_event"], episodes, **options)}
    for control in CONTROL_METHODS:
        pairs[control + "_to_state_event"] = compare_paired(labels, predictions[control], predictions["state_event"], episodes, **options)
    return {"schema_version": 1, "seed": seed, "bootstrap_iterations": bootstrap_iterations, "methods": evaluated,
            "paired": pairs, "control_predictions": {m: predictions[m] for m in CONTROL_METHODS},
            "limits": ["Controls are deliberately label-free and always predict failure; they are not oracle predictors.",
                       "Sampler oracle coverage is computed separately using annotations after inference.",
                       "All selected episodes remain in every method's denominators; no missing output is dropped."]}
