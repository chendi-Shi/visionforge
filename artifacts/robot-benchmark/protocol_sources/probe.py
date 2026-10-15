"""Frozen CLIP features and task-disjoint exploratory linear probes.

These are cross-validation results on this small dataset, NOT official OOD
zero-shot scores. Hyperparameters are fixed; no test-label model selection.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np

from robotbench.models import _unit, feature_vector
from video_index import FastEmbedCLIP
from video_media import _atomic_json, _sha256


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def encode_samples(samples, root: Path, output: Path, *, embedder=None):
    """One cache per raw image hash or exact subtask, bound to pinned encoders."""
    started = time.perf_counter()
    engine = embedder or FastEmbedCLIP()
    identity = engine.identity()
    identity_hash = _hash(identity)
    # Windows' usual path limit also includes _atomic_json's temporary suffix.
    # Short names are locators only; full identities below reject collisions.
    cache = output / "embeddings" / identity_hash[:12]
    cache.mkdir(parents=True, exist_ok=True)
    identity_path = cache / "identity.json"
    cache_identity = {"identity": identity, "identity_sha256": identity_hash}
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != cache_identity:
            raise ValueError("Embedding cache namespace collision or identity change")
    else:
        _atomic_json(identity_path, cache_identity)

    def encoded(key, producer):
        file = cache / f"{_hash(key)[:24]}.json"
        if file.exists():
            saved = json.loads(file.read_text(encoding="utf-8"))
            if saved.get("key") != key or saved.get("identity") != identity or saved.get("identity_sha256") != identity_hash:
                raise ValueError("Embedding cache key collision or identity change")
            if saved.get("sha256") != _hash(saved["vector"]):
                raise ValueError("Corrupt embedding cache")
            vec = _unit(saved["vector"])
            if vec.shape != (identity["dimension"],):
                raise ValueError("Wrong cached embedding dimension")
            return vec
        vec = _unit(producer())
        if vec.shape != (identity["dimension"],):
            raise ValueError("Wrong embedding dimension")
        values = vec.tolist()
        _atomic_json(file, {"key": key, **cache_identity, "vector": values, "sha256": _hash(values)})
        return vec

    rows = []
    for i, sample in enumerate(samples):
        phases = {"before": [], "after": []}
        for frame in sample["frames"]:
            path = (root / frame["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or _sha256(path) != frame["sha256"]:
                raise ValueError("Missing, escaped or changed input image")
            vec = encoded("image-" + frame["sha256"], lambda: engine.embed_images([str(path)])[0])
            phases[frame["phase"]].append(vec)
        if any(len(values) != 3 for values in phases.values()):
            raise ValueError("UR5 requires three cameras per phase")
        text = sample["subtask_instruction"]
        goal = encoded("text-" + _hash(text), lambda: engine.embed_text(text))
        rows.append({"sample_id": sample["sample_id"], "before": _unit(np.mean(phases["before"], axis=0)).tolist(),
                     "after": _unit(np.mean(phases["after"], axis=0)).tolist(), "text": goal.tolist()})
        print(json.dumps({"stage": "features", "completed": i + 1, "total": len(samples), "sample_id": sample["sample_id"]}), flush=True)
    result = {"identity": identity, "input_sha256": _hash([{k: s[k] for k in ("sample_id", "frames", "subtask_instruction")} for s in samples]),
              "code_sha256": _sha256(Path(__file__)), "elapsed_seconds": time.perf_counter() - started,
              "pooling": "unit-normalized mean of three independently unit-normalized camera embeddings per phase",
              "text": "subtask_instruction only (CLIP 77-token limit; no silent truncation)", "rows": rows}
    _atomic_json(output / "features.json", result)
    return result


def cross_validate(samples, features):
    """Leave one base task family out; purge cross-fold shared images.

    Every sample receives one out-of-family prediction. Counterfactual goals
    sharing observations stay in the same family. Any residual cross-family
    shared image causes removal from TRAINING, never from the test denominator.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from threadpoolctl import threadpool_limits

    ids = [s["sample_id"] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample identity")
    by_id = {r["sample_id"]: r for r in features["rows"]}
    if len(by_id) != len(features["rows"]) or set(by_id) != set(ids):
        raise ValueError("Features must exactly match the evaluation sample set")
    labels = np.asarray([s["label"] for s in samples])
    if set(labels) != {"success", "failure"}:
        raise ValueError("Both classes are required")
    groups = np.asarray([s["task_group"] for s in samples])
    hashes = [{f["sha256"] for f in s["frames"]} for s in samples]
    matrix = {m: np.asarray([feature_vector(by_id[id]["before"], by_id[id]["after"], by_id[id]["text"], m) for id in ids])
              for m in ("text_only", "after", "transition")}
    predictions = {m: [] for m in ("train_majority", *matrix)}
    folds = []
    for train, test in LeaveOneGroupOut().split(matrix["after"], labels, groups):
        test_hashes = set().union(*(hashes[i] for i in test))
        purged = [int(i) for i in train if hashes[i] & test_hashes]
        train = np.asarray([i for i in train if not hashes[i] & test_hashes], dtype=int)
        if len(train) == 0:
            raise ValueError("Shared-image purge left no training rows")
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Task leakage")
        counts = Counter(labels[train])
        majority = sorted(counts, key=lambda k: (-counts[k], k))[0]
        audit = {"test_task_group": str(groups[test[0]]), "train_sample_ids": [ids[i] for i in train],
                 "test_sample_ids": [ids[i] for i in test], "purged_shared_image_ids": [ids[i] for i in purged],
                 "train_class_counts": dict(counts), "warnings": {}}
        for method in predictions:
            began = time.perf_counter()
            if method == "train_majority" or len(counts) < 2:
                predicted = [majority] * len(test)
                scores = [counts.get("success", 0) / len(train)] * len(test)
            else:
                # Fit both scaling and the classifier solely on this fold's train rows.
                classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                                                                  max_iter=2000, random_state=0))
                with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    classifier.fit(matrix[method][train], labels[train])
                    predicted = classifier.predict(matrix[method][test]).tolist()
                    scores = classifier.predict_proba(matrix[method][test])[:, list(classifier.classes_).index("success")].tolist()
                audit["warnings"][method] = [str(w.message) for w in caught]
            elapsed = time.perf_counter() - began
            predictions[method].extend({"sample_id": ids[i], "prediction": str(p), "status": "ok", "method": method,
                                        "score_success": float(score), "score_note": "uncalibrated classifier output",
                                        "held_out_group": str(groups[i]), "fold_train_n": len(train),
                                        "fold_elapsed_seconds": elapsed, "single_class_fallback": len(counts) < 2}
                                       for i, p, score in zip(test, predicted, scores))
        folds.append(audit)
    return {"protocol": "exploratory leave-one-task-family-out CV; not official zero-shot benchmark",
            "config": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000,
                       "random_state": 0, "scaler": "StandardScaler fitted inside each train fold", "thread_limit": 2},
            "packages": {k: version(k) for k in ("scikit-learn", "scipy", "numpy", "threadpoolctl")},
            "code_sha256": _sha256(Path(__file__)), "models_code_sha256": _sha256(Path(__file__).with_name("models.py")),
            "folds": folds, "predictions": predictions}
