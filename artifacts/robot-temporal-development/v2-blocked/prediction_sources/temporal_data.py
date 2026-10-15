"""Read the pinned CC-BY-4.0 FailTime data without importing upstream code.

Author labels are exported separately. load_inputs never opens labels, source
annotations or the selection plan. All timestamp coordinates remain episode
relative. SO-101 arm poses use separate base frames, not one world frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET

DATASET = "ChangUoA/FailTime"
REVISION = "07aea9142616cf1f8047512bd53a4d9fcd5e7cee"
ANNOTATION_BYTES = 42940
ANNOTATION_SHA256 = "14b1de4d92c2eb194215ca1608955f4704e86cde4a34a78628da1a10f3561e41"
SELECTION_SEED = "visionforge-failtime-temporal-v1"
DEVELOPMENT_SAMPLES = ("uncap_marker_draw_recap_episode_43", "block_assembly_episode_09", "block_assembly_episode_12")
TASK_GIT_BLOBS = {
    "balls_to_egg_carton": "a1be21d24c38283c74d638dde2b7e1e6b3d34545",
    "balls_to_rack": "7bd8e755b71edcfe180c4ce1cd90d74296c88f55",
    "block_assembly": "69417753d590bca3de9a7e5da5807852a5fb228f",
    "blocks_to_bowls": "5d261b9746dc9d7aecc2c0b4b81384df5d617a23",
    "bottle_to_bowl_and_cover": "6cadc77c82bacf9524b40bdc3c0cebc535c10e03",
    "cube_stack": "050588ecd7deef3ac6d497bb03a76d89b877ba77",
    "empty_bowl_insert_block": "530dd2e8581319430f4708e05a978d5bc85c696a",
    "handover": "10921686e7b36163391017f726b12f01235d44e8",
    "pour_ball_transfer_cup": "b57d759614e3d2c99c452792e8f3001bc4ce48aa",
    "stabilize_and_insert_block": "d9b2d285e0f63d84239990ddc22b0fd7f963ff17",
    "stack_cup_pyramid": "d885dbfea31bc914eb027358bf1ace7f6a753e74",
    "uncap_marker_draw_recap": "f2381f721ee7156d98711eaf49b081f8f9a9a1be",
    "yellow_red_blocks_to_bowls": "fab19259ff7fe896490c122739038b2dd6819398",
}
INPUT_FIELDS = {"sample_id", "task_id", "robot", "split", "task_description", "substages",
                "source_sha256", "timestamp", "state_eef", "videos", "head_view",
                "video_sha256", "fps", "duration_seconds", "eef_metadata"}
STORED_FIELDS = (INPUT_FIELDS - {"timestamp", "state_eef"}) | {"states_path", "states_sha256"}
FORBIDDEN = {"label", "verdict", "onset", "onset_seconds", "reason", "avoidance", "failure_types", "annotation"}
EEF_FIELDS = {"frame", "layout", "gripper", "joint_units", "source_field", "fk_model", "fk_site", "joint_map", "qpos_layout_14d", "qpos_layout_18d"}


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def _relative(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe dataset-relative path")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("Unsafe dataset-relative path")
    root = Path(root).resolve()
    result = (root / name).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Path escapes dataset directory")
    return result


def _safe_name(value):
    if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Invalid source identifier")
    return value


def validate_plan(plan):
    claimed = plan.get("plan_sha256")
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != claimed:
        raise ValueError("Selection plan digest mismatch")
    if plan.get("dataset") != DATASET or plan.get("revision") != REVISION:
        raise ValueError("Unsupported dataset revision")
    if not isinstance(plan.get("samples"), list) or not plan["samples"]:
        raise ValueError("Plan requires samples")
    seen, sources, split_tasks = set(), set(), {"development": set(), "test": set()}
    for item in plan["samples"]:
        for key in ("sample_id", "source_sample", "task_id"):
            _safe_name(item[key])
        if item["sample_id"] in seen or item["source_sample"] in sources:
            raise ValueError("Duplicate sample ID or episode")
        seen.add(item["sample_id"])
        sources.add(item["source_sample"])
        if item["split"] not in split_tasks:
            raise ValueError("Unknown split")
        split_tasks[item["split"]].add(item["task_id"])
        if item["relative_hdf5"] != f"failtime_long/episodes/{item['source_sample']}.hdf5":
            raise ValueError("Source path does not match sample")
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ValueError("Invalid source size")
        if len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"]):
            raise ValueError("Invalid source SHA")
        label = item["annotation"]
        if label["sample"] != item["source_sample"] or label["task_id"] != item["task_id"] or label["robot"] != item["robot"]:
            raise ValueError("Annotation identity mismatch")
        if label["verdict"] == "fail":
            onset = label.get("onset")
            if isinstance(onset, bool) or not isinstance(onset, (int, float)) or not math.isfinite(onset) or onset < 0:
                raise ValueError("Failure requires a finite author onset")
        elif label["verdict"] != "success" or "onset" in label:
            raise ValueError("Success must not have an invented onset")
    if split_tasks["development"] & split_tasks["test"]:
        raise ValueError("Development task group leaks into test")
    return plan


def _source_url(relative):
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{relative}"


def _download_verified(url, path, size, expected_sha, local=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != size or sha256_file(path) != expected_sha:
            raise ValueError(f"Existing source differs from frozen plan: {path.name}")
        return
    # Stable partial file permits resume after a dropped connection/process.
    # Bytes are promoted only after the complete pinned SHA has been checked.
    tmp = path.with_name(path.name + "." + expected_sha[:12] + ".part")
    if local is not None and Path(local).is_file() and Path(local).stat().st_size == size and sha256_file(local) == expected_sha:
        shutil.copyfile(local, tmp)
    failures = 0
    while (tmp.stat().st_size if tmp.exists() else 0) < size:
        offset = tmp.stat().st_size if tmp.exists() else 0
        end = min(size - 1, offset + 8 * 1024 * 1024 - 1)
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                status = getattr(response, "status", 200)
                if status == 206:
                    if response.headers.get("Content-Range") != f"bytes {offset}-{end}/{size}":
                        raise ValueError("Server returned a different byte range")
                elif status != 200 or offset != 0 or end != size - 1:
                    raise ValueError("Server did not honor byte range")
                received = 0
                with tmp.open("ab") as stream:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if offset + received > end + 1:
                            raise ValueError("Download exceeds requested range")
                        stream.write(chunk)
                if received != end - offset + 1:
                    raise OSError("Short range response")
            failures = 0
        except (OSError, TimeoutError):
            progressed = tmp.exists() and tmp.stat().st_size > offset
            failures = 0 if progressed else failures + 1
            if failures >= 4:
                raise
            time.sleep(min(max(failures, 1), 3))
    if tmp.stat().st_size != size or sha256_file(tmp) != expected_sha:
        tmp.unlink(missing_ok=True)
        raise ValueError("Downloaded source hash/size mismatch")
    os.replace(tmp, path)


def _task_spec(path):
    root = ET.fromstring(Path(path).read_bytes())
    description = root.findtext("description", "").strip().replace("\\n", "\n")
    substages = [node.text.strip() for node in root.findall("./substages/substage") if node.text and node.text.strip()]
    if not description or not substages:
        raise ValueError("Task XML requires description and substages")
    return description, substages


def _verify_author_annotations(plan, root, plan_path):
    path = root / "source" / "test.json"
    _download_verified(_source_url("failtime_long/test.json"), path, ANNOTATION_BYTES,
                       ANNOTATION_SHA256, local=plan_path.parent / "failtime_long_test.json")
    published = {row["sample"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
    for sample in plan["samples"]:
        if sample["annotation"] != published.get(sample["source_sample"]):
            raise ValueError("Plan label differs from pinned author annotation")
    if "seed" in plan:
        if plan["seed"] != SELECTION_SEED:
            raise ValueError("Unexpected pilot selection seed")
        actual_development = [s["source_sample"] for s in plan["samples"] if s["split"] == "development"]
        actual_test = [s["source_sample"] for s in plan["samples"] if s["split"] == "test"]
        if actual_development != list(DEVELOPMENT_SAMPLES) or actual_test != select_test_samples(list(published.values())):
            raise ValueError("Frozen membership differs from the deterministic selection rule")


def select_test_samples(annotations, *, seed=SELECTION_SEED, development=DEVELOPMENT_SAMPLES, task_count=8):
    """Reproduce the frozen pilot using task/name hashes and verdict strata only."""
    by_name = {row["sample"]: row for row in annotations}
    if len(by_name) != len(annotations):
        raise ValueError("Duplicate author sample ID")
    excluded = {by_name[name]["task_id"] for name in development}
    def order(kind, name):
        return hashlib.sha256(f"{seed}:{kind}:{name}".encode()).hexdigest()
    tasks = sorted({row["task_id"] for row in annotations} - excluded, key=lambda name: order("task", name))
    if len(tasks) < task_count:
        raise ValueError("Insufficient task groups")
    selected = []
    for task in tasks[:task_count]:
        for verdict in ("fail", "success"):
            rows = [r for r in annotations if r["task_id"] == task and r["verdict"] == verdict]
            if not rows:
                raise ValueError("Selected task lacks a required verdict stratum")
            selected.append(min(rows, key=lambda row: order("sample", row["sample"]))["sample"])
    return selected


def _native(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _check_time(timestamp, indices, eef):
    import numpy as np
    if timestamp.ndim != 1 or len(timestamp) < 2 or not np.isfinite(timestamp).all():
        raise ValueError("Invalid timestamps")
    if timestamp[0] != 0 or not np.all(np.diff(timestamp) > 0):
        raise ValueError("Timestamps must increase from episode zero")
    if not np.array_equal(indices, np.arange(len(timestamp))):
        raise ValueError("Frame indices do not match timeline")
    if eef.shape != (len(timestamp), 16) or not np.isfinite(eef).all():
        raise ValueError("Invalid synchronized end-effector states")


def _video_metadata(path, n_frames, fps):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video cannot be opened")
        count, rate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if count != n_frames or abs(rate - fps) > 1e-4:
            raise ValueError("Video frame count/fps does not match states")
        # Feasibility samples were fully decoded; each new test asset receives
        # mechanical first/middle/last decode checks without visual inspection.
        for index in sorted({0, n_frames // 2, n_frames - 1}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Video frame {index} does not decode")
        return {"frame_count": count, "fps": rate, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def _extract_sample(source, root, sample, task_path):
    import h5py
    import numpy as np
    root = Path(root)
    folder = root / "observations" / sample["sample_id"]
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder / "complete.json"
    spec_sha = sha256_file(task_path)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("format") == 1 and cached.get("source_sha256") == sample["sha256"] and cached.get("task_sha256") == spec_sha:
            for name, expected in cached["payload_sha256"].items():
                if sha256_file(_relative(root, name)) != expected:
                    raise ValueError("Prepared observation cache was modified")
            return cached["input"]
    description, substages = _task_spec(task_path)
    with h5py.File(source, "r") as data:
        timestamp = np.asarray(data["timestamp"][:], dtype=np.float64).reshape(-1)
        indices = data["frame_index"][:].reshape(-1)
        eef = np.asarray(data["observation.state_eef"][:], dtype=np.float64)
        _check_time(timestamp, indices, eef)
        fps = float(data.attrs.get("fps", 1 / float(np.median(np.diff(timestamp)))))
        # Use actual camera rate when the legacy ALOHA container has no fps attr.
        if "fps" not in data.attrs:
            fps = float(round(fps))
        if not math.isfinite(fps) or fps <= 0 or np.max(np.abs(timestamp - np.arange(len(timestamp)) / fps)) > 1e-4:
            raise ValueError("Timestamp grid and frame rate disagree")
        n_frames = len(timestamp)
        fields = {"timestamp": timestamp, "frame_index": indices, "state_eef": eef}
        for key, name in [("observation.state", "state"), ("action", "action"), ("observation.effort", "effort")]:
            if key in data:
                values = data[key][:]
                if len(values) != n_frames or not np.isfinite(values).all():
                    raise ValueError("Robot state/action timeline mismatch")
                fields[name] = values
        states = folder / "states.npz"
        fd, temp = tempfile.mkstemp(suffix=".npz", dir=folder)
        os.close(fd)
        try:
            np.savez_compressed(temp, **fields)
            os.replace(temp, states)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        videos, video_hashes, video_checks = {}, {}, {}
        for key in sorted(data):
            if not key.startswith("observation.images."):
                continue
            view = _safe_name(key.removeprefix("observation.images."))
            group = data[key]
            camera_time = np.asarray(group["timestamp"][:], dtype=np.float64).reshape(-1)
            if camera_time.shape != timestamp.shape or not np.isfinite(camera_time).all() or np.max(np.abs(camera_time - timestamp)) > 1e-4:
                raise ValueError("Camera timeline disagrees with robot state timeline")
            path = folder / (view + ".mp4")
            _atomic_bytes(path, group["mp4_bytes"][:].tobytes())
            video_checks[view] = _video_metadata(path, n_frames, fps)
            videos[view] = path.relative_to(root).as_posix()
            video_hashes[view] = sha256_file(path)
        head = next((key for key in ("head", "cam_high", "cam_low") if key in videos), None)
        if len(videos) != 3 or head is None:
            raise ValueError("Expected three camera streams and a head/overview stream")
        record = {"sample_id": sample["sample_id"], "task_id": sample["task_id"], "robot": sample["robot"],
                  "split": sample["split"], "task_description": description, "substages": substages,
                  "source_sha256": sample["sha256"], "states_path": states.relative_to(root).as_posix(),
                  "states_sha256": sha256_file(states), "videos": videos, "video_sha256": video_hashes,
                  "head_view": head, "fps": fps, "duration_seconds": n_frames / fps,
                  "eef_metadata": {k: _native(v) for k, v in data["observation.state_eef"].attrs.items() if k in EEF_FIELDS}}
        assert set(record) == STORED_FIELDS and not (FORBIDDEN & set(record))
    payload = {record["states_path"]: record["states_sha256"]}
    payload.update({videos[key]: video_hashes[key] for key in videos})
    _write_json(cache_path, {"format": 1, "source_sha256": sample["sha256"], "task_sha256": spec_sha,
                            "payload_sha256": payload, "video_checks": video_checks, "input": record})
    return record


def prepare_dataset(plan_path, dest, *, split=None):
    """Download pinned files and materialize isolated inputs/labels.

    Optional split='development' lets a runner start development while the fixed
    test media downloads. Later calls merge completed records under the same plan.
    """
    plan_path, root = Path(plan_path).resolve(), Path(dest).resolve()
    plan = validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    root.mkdir(parents=True, exist_ok=True)
    _verify_author_annotations(plan, root, plan_path)
    destination_plan = root / "plan.json"
    if destination_plan.exists():
        previous = validate_plan(json.loads(destination_plan.read_text(encoding="utf-8")))
        if previous["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Output directory belongs to a different selection plan")
    else:
        _atomic_bytes(destination_plan, plan_path.read_bytes())
    if split not in (None, "development", "test"):
        raise ValueError("Unknown requested split")
    records = {}
    if (root / "inputs.jsonl").exists():
        records = {x["sample_id"]: x for x in _read_jsonl(root / "inputs.jsonl")}
    for sample in plan["samples"]:
        if split is not None and sample["split"] != split:
            continue
        source = root / "source" / "episodes" / (sample["source_sample"] + ".hdf5")
        _download_verified(_source_url(sample["relative_hdf5"]), source, sample["bytes"], sample["sha256"], local=plan_path.parent / source.name)
        task = root / "source" / "tasks" / (sample["task_id"] + ".xml")
        if not task.exists():
            with urllib.request.urlopen(_source_url(f"failtime_long/tasks/{sample['task_id']}.xml"), timeout=30) as response:
                body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise ValueError("Task XML exceeds size limit")
            _atomic_bytes(task, body)
        task_bytes = task.read_bytes()
        task_git_blob = hashlib.sha1(f"blob {len(task_bytes)}\0".encode() + task_bytes).hexdigest()
        if task_git_blob != TASK_GIT_BLOBS.get(sample["task_id"]):
            raise ValueError("Task XML differs from the pinned author version")
        record = _extract_sample(source, root, sample, task)
        label = sample["annotation"]
        if label["verdict"] == "fail" and label["onset"] >= record["duration_seconds"]:
            raise ValueError("Author onset lies outside the episode")
        records[sample["sample_id"]] = record
        # Checkpoint only completed samples. No inference code needs plan/labels.
        ordered = [records[s["sample_id"]] for s in plan["samples"] if s["sample_id"] in records]
        _atomic_bytes(root / "inputs.jsonl", b"".join(_canonical(row) + b"\n" for row in ordered))
        print(f"Prepared {sample['split']} {sample['sample_id']} ({len(ordered)}/{len(plan['samples'])})", flush=True)
    labels = []
    for sample in plan["samples"]:
        if sample["sample_id"] not in records:
            continue
        annotation = sample["annotation"]
        labels.append({"sample_id": sample["sample_id"], "task_id": sample["task_id"], "split": sample["split"],
                       "label": "failure" if annotation["verdict"] == "fail" else "success",
                       "onset_seconds": annotation.get("onset"), "failure_types": annotation.get("type", [])})
    _atomic_bytes(root / "labels.jsonl", b"".join(_canonical(row) + b"\n" for row in labels))
    _write_json(root / "manifest.json", {"format": 1, "dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0",
                "plan_sha256": plan["plan_sha256"], "samples_prepared": len(records), "samples_planned": len(plan["samples"]),
                "complete": len(records) == len(plan["samples"]), "inputs_sha256": sha256_file(root / "inputs.jsonl"),
                "labels_sha256": sha256_file(root / "labels.jsonl"), "limitations": plan.get("limitations", [])})
    return root / "manifest.json"


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_inputs(dest, *, split=None):
    """Canonical prediction allowlist; never reads selection plan or labels.

    timestamp/state_eef return plain lists for callers that serialize prompts.
    Paths are absolute only for local media loading, not prompt text.
    """
    import numpy as np
    root = Path(dest).resolve()
    rows = _read_jsonl(root / "inputs.jsonl")
    if len({row.get("sample_id") for row in rows}) != len(rows):
        raise ValueError("Duplicate prepared input ID")
    if any(set(row) != STORED_FIELDS or FORBIDDEN & set(row) for row in rows):
        raise ValueError("Unexpected or label-bearing input fields")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(root / "inputs.jsonl") != manifest.get("inputs_sha256"):
        raise ValueError("Prepared input metadata changed")
    result = []
    for row in rows:
        if split is not None and row["split"] != split:
            continue
        states = _relative(root, row["states_path"])
        if sha256_file(states) != row["states_sha256"]:
            raise ValueError("Prepared states changed")
        with np.load(states, allow_pickle=False) as data:
            time, eef = data["timestamp"], data["state_eef"]
            _check_time(time, data["frame_index"], eef)
            record = {k: v for k, v in row.items() if k not in {"states_path", "states_sha256"}}
            record["timestamp"], record["state_eef"] = time.tolist(), eef.tolist()
        videos = {}
        for view, name in row["videos"].items():
            path = _relative(root, name)
            if sha256_file(path) != row["video_sha256"][view]:
                raise ValueError("Prepared video changed")
            videos[view] = str(path)
        record["videos"] = videos
        if record["head_view"] not in videos:
            raise ValueError("Head view missing")
        assert set(record) == INPUT_FIELDS
        result.append(record)
    return result


def load_labels(dest, *, split=None):
    """Evaluation-only labels; these must not enter a prediction prompt."""
    records = _read_jsonl(Path(dest) / "labels.jsonl")
    return [row for row in records if split is None or row["split"] == split]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dest", type=Path, default=Path("data/temporalbench"))
    parser.add_argument("--split", choices=("development", "test"))
    args = parser.parse_args()
    print(prepare_dataset(args.plan, args.dest, split=args.split))
