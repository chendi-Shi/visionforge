"""Post-prediction instruction sensitivity on identical observation pairs.

This is an analysis of fixed outputs, not a new model, selection rule, or tuning
objective. Pairs share samples and images: their counts are not independent trials.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
import json
import re

from robotbench.metrics import LABELS, _checked_samples, _predictions


def _observation_key(sample):
    frames = sample.get("frames")
    if not isinstance(frames, list) or len(frames) != 6:
        raise ValueError("Counterfactual diagnostics require all six ordered UR5 frames.")
    if [frame.get("phase") for frame in frames] != ["before"] * 3 + ["after"] * 3:
        raise ValueError("Expected three before frames followed by three after frames.")
    views = [str(frame.get("view")) for frame in frames]
    if len(set(views[:3])) != 3 or views[:3] != views[3:]:
        raise ValueError("Camera slots must correspond across phases.")
    hashes = [frame.get("sha256") for frame in frames]
    if any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes):
        raise ValueError("Every observation requires a valid image SHA256.")
    return tuple(zip(views, hashes))


def _group_id(observations):
    return hashlib.sha256(json.dumps(observations, separators=(",", ":")).encode()).hexdigest()


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def counterfactual_pairs(samples, predictions):
    """Score different-subtask/opposite-label pairs with identical ordered pixels.

    Same task AND subtask with opposite labels is an annotation/input conflict,
    excluded from this instruction-change score and explicitly reported. Same
    subtask but different overall task is also reported separately because it
    does not meet the prespecified different-subtask diagnostic.
    """
    samples = _checked_samples(samples)
    normalized = {row["sample_id"]: row for row in _predictions(samples, predictions)}
    groups = defaultdict(list)
    for sample in samples:
        for field in ("task_instruction", "subtask_instruction"):
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError(f"Missing {field} for counterfactual diagnostics.")
        groups[_observation_key(sample)].append(sample)
    eligible, excluded, conflicts = [], [], []
    same_label_pairs = 0
    for observations, group in sorted(groups.items()):
        observation_id = _group_id(observations)
        for left, right in combinations(sorted(group, key=lambda row: row["sample_id"]), 2):
            if left["label"] == right["label"]:
                same_label_pairs += 1
                continue
            ids = [left["sample_id"], right["sample_id"]]
            labels = [left["label"], right["label"]]
            common = {"observation_group": observation_id, "sample_ids": ids, "labels": labels}
            if left["subtask_instruction"] == right["subtask_instruction"]:
                if left["task_instruction"] == right["task_instruction"]:
                    input_id = hashlib.sha256(json.dumps([observation_id, left["task_instruction"], left["subtask_instruction"]], ensure_ascii=False).encode()).hexdigest()
                    item = {**common, "reason": "identical_input_opposite_labels", "input_sha256": input_id}
                    conflicts.append(item)
                else:
                    item = {**common, "reason": "same_subtask_different_overall_task"}
                excluded.append(item)
                continue
            outcomes = [normalized[identifier] for identifier in ids]
            guesses = [row["prediction"] for row in outcomes]
            answered = [guess in LABELS for guess in guesses]
            correct = [guess == label for guess, label in zip(guesses, labels)]
            eligible.append({**common, "predictions": guesses,
                             "abstention_reasons": [row["abstention_reason"] for row in outcomes],
                             "both_answered": all(answered), "answered_count": sum(answered),
                             "both_correct": all(correct), "correct_count": sum(correct),
                             "same_binary_prediction": all(answered) and guesses[0] == guesses[1],
                             "opposite_binary_predictions": all(answered) and guesses[0] != guesses[1],
                             "same_prediction_including_abstention": guesses[0] == guesses[1]})
    conflicted_ids = {identifier for pair in conflicts for identifier in pair["sample_ids"]}
    for pair in eligible:
        pair["involves_conflicted_input"] = bool(set(pair["sample_ids"]) & conflicted_ids)
    count = len(eligible)
    answered = sum(pair["both_answered"] for pair in eligible)
    correct = sum(pair["both_correct"] for pair in eligible)
    same = sum(pair["same_binary_prediction"] for pair in eligible)
    same_all = sum(pair["same_prediction_including_abstention"] for pair in eligible)
    return {
        "protocol": "post-prediction identical-observations / different-subtask / opposite-label diagnostic",
        "status": "computed" if count else "no_eligible_pairs",
        "n_samples": len(samples), "observation_groups": len(groups),
        "observation_groups_with_multiple_rows": sum(len(group) > 1 for group in groups.values()),
        "eligible_observation_groups": len({pair["observation_group"] for pair in eligible}),
        "eligible_samples": len({identifier for pair in eligible for identifier in pair["sample_ids"]}),
        "eligible_pairs": count, "both_correct": correct, "both_correct_rate": _ratio(correct, count),
        "eligible_pairs_involving_conflicted_input": sum(pair["involves_conflicted_input"] for pair in eligible),
        "both_answered": answered, "pair_coverage": _ratio(answered, count),
        "one_answered": sum(pair["answered_count"] == 1 for pair in eligible),
        "neither_answered": sum(pair["answered_count"] == 0 for pair in eligible),
        "selective_both_correct_rate": _ratio(correct, answered),
        "same_binary_prediction": same, "same_binary_prediction_rate": _ratio(same, count),
        "same_binary_given_both_answered": _ratio(same, answered),
        "opposite_binary_predictions": sum(pair["opposite_binary_predictions"] for pair in eligible),
        "same_prediction_including_abstention": same_all,
        "same_prediction_including_abstention_rate": _ratio(same_all, count),
        "same_label_pairs_not_scored": same_label_pairs,
        "opposite_label_pairs_total": count + len(excluded),
        "excluded_pairs": excluded,
        "identical_input_conflict_pairs": len(conflicts),
        "identical_input_conflict_samples": len({identifier for pair in conflicts for identifier in pair["sample_ids"]}),
        "identical_input_conflict_groups": len({pair["input_sha256"] for pair in conflicts}),
        "identical_input_conflict_observation_groups": len({pair["observation_group"] for pair in conflicts}),
        "identical_input_conflicts": conflicts,
        "same_subtask_different_task_pairs": sum(item["reason"] == "same_subtask_different_overall_task" for item in excluded),
        "pairs": eligible,
        "definitions": {
            "matching": "Exact ordered before/after image SHA256 and camera-slot identity; instruction equality uses exact strings.",
            "both_correct_rate": "Both original labels predicted correctly / all eligible pairs, including pairs with missing or failed outputs.",
            "same_binary_prediction": "Both outputs are success/failure and equal; two abstentions are counted separately, never as answered agreement.",
            "conflicts": "Opposite labels for exactly the same task, subtask and ordered observations are reported outside the instruction-change score; other pairs involving these samples remain explicitly flagged.",
        },
        "limits": ["Pairs can share samples and observations; pair count is not an independent sample size.",
                   "This diagnostic does not change selection, fitting, hyperparameters, or headline sample denominators.",
                   "No additional inference is run and no statistical significance or causal interpretation is claimed.",
                   "No eligible pairs yields null rates, not a zero or perfect score."]}
