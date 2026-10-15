"""Pinned UR5-Fail execution data, with ground truth separated from model input.

The official ordered images are authoritative. Their numerical filenames are
not timestamps and must never be sorted to infer a different transition.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

DATASET_ID = "paulpacaud/Guardian-FailCoT-OOD-datasets"
REVISION = "bd2b24224e4da833a8d42ca694d839622e54bb48"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}/tree/{REVISION}/ur5fail_test"
PINNED_FILES = {
    "ur5fail_test/metadata_execution.jsonl": {"size": 140467, "git_sha1": "0de4ad8df52f69e4c33477fdbf0b63b01f9c08fc"},
    "ur5fail_test/internVL_dataset_execution_vanilla.jsonl": {"size": 350449, "git_sha1": "77f4bc1e750d6b566ca332690cebcdd947ead813"},
    "ur5fail_test/records.tar.gz": {"size": 81685957, "sha256": "4d3ff623db50221281a053343f8aec82e438cd5a0358998743e158e564f228fa"},
}
VIEW_NAMES = {"0": "front", "1": "left", "2": "right"}
CATEGORY_NAMES = {
    "success": "success", "no gripper close": "no_gripper_close",
    "imprecise grasping/pushing": "imprecise_grasping_pushing",
    "wrong object state or placement": "wrong_state_or_placement",
    "wrong object manipulated": "wrong_object", "no progress": "no_progress",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_download(path, spec):
    path = Path(path)
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"Pinned download size mismatch: {path.name}")
    if "sha256" in spec and sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Pinned download SHA256 mismatch: {path.name}")
    if "git_sha1" in spec:
        content = path.read_bytes()
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if digest != spec["git_sha1"]:
            raise ValueError(f"Pinned Git blob mismatch: {path.name}")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path, rows):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _source_key(value):
    """Canonicalize archive lookup only; never expose original paths to models."""
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("Invalid source image path")
    parts = PurePosixPath(value).parts
    if PurePosixPath(value).is_absolute() or ".." in parts or any(":" in p for p in parts):
        raise ValueError("Unsafe source image path")
    if "records" not in parts:
        raise ValueError("Image path does not identify the records directory")
    return "/".join(parts[parts.index("records"):])


def _safe_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Prepared image escaped the dataset directory")
    return target


def _extract_needed(archive_path, required, output_root):
    """Read only named regular PNG members; never execute or extract tar paths."""
    images = Path(output_root) / "images"
    images.mkdir(parents=True, exist_ok=True)
    found = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.name.endswith(".png"):
                continue
            key = _source_key(member.name)
            if key not in required:
                continue
            if key in found or not member.isfile() or not 0 < member.size <= 8 * 1024 * 1024:
                raise ValueError("Duplicate, linked, or oversized source image")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Cannot read source image")
            with handle:
                content = handle.read(member.size + 1)
            if len(content) != member.size or not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Source image is not an intact PNG")
            digest = hashlib.sha256(content).hexdigest()
            relative = f"images/{digest}.png"
            path = _safe_path(output_root, relative)
            if path.exists() and sha256_file(path) != digest:
                raise ValueError("Previously prepared image changed")
            if not path.exists():
                path.write_bytes(content)
            found[key] = {"path": relative, "sha256": digest}
    missing = required - set(found)
    if missing:
        raise ValueError(f"Archive lacks {len(missing)} required image members")
    return found


def prepare_dataset(metadata_path, conversation_path, archive_path, output_root, *, expected_count=140):
    """Prepare separate inputs/labels plus private source provenance.

    Callers downloading upstream data must first verify_download against the
    pinned constants. The optional count exists for small offline fixtures.
    """
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata, conversations = _read_jsonl(metadata_path), _read_jsonl(conversation_path)
    if len(metadata) != expected_count or len(conversations) != expected_count:
        raise ValueError("Official metadata/conversation sample count differs")
    required, validated = set(), []
    for ordinal, (row, conversation) in enumerate(zip(metadata, conversations)):
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 6 or conversation.get("image") != images:
            raise ValueError("Expected matching ordered before/after three-view images")
        for index, source in enumerate(images):
            # The actual release mixes numerical views and left/right/wrist,
            # despite its generic conversation template naming front/left/right.
            # Keep positional slots rather than claiming a camera calibration.
            named_view = ("left", "right", "wrist")[index % 3]
            if not source.endswith((f"_img_viewpoint_{index % 3}.png", f"_img_viewpoint_{named_view}.png")):
                raise ValueError("Official viewpoint ordering changed")
            required.add(_source_key(source))
        label_value = row.get("execution_reward")
        if type(label_value) is not int or label_value not in (0, 1) or row.get("reward") != label_value or row.get("planning_reward") != 1:
            raise ValueError("Expected unambiguous binary execution labels")
        answer = next((x["value"] for x in conversation.get("conversations", []) if x.get("from") == "gpt"), "")
        match = re.search(r"<answer>\s*(True|False)\s*</answer>", answer)
        category_match = re.search(r"<category>\s*([^<]+)\s*</category>", answer)
        if not match or int(match.group(1) == "True") != label_value or not category_match:
            raise ValueError("Official label disagrees with evaluation answer")
        category = CATEGORY_NAMES.get(category_match.group(1).strip().split(":", 1)[0])
        if category is None or (category == "success") != bool(label_value):
            raise ValueError("Unsupported or inconsistent official failure category")
        for key in ("taskvar", "task_instruction", "detailed_subtask_name"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing official {key}")
        sample_id = "ur5-" + hashlib.sha256(f"{DATASET_ID}:{REVISION}:{ordinal}".encode()).hexdigest()[:20]
        validated.append((row, sample_id, "success" if label_value else "failure", category))
    assets = _extract_needed(archive_path, required, root)
    inputs, labels, provenance = [], [], []
    for ordinal, (row, sample_id, label, category) in enumerate(validated):
        frames = []
        for index, source in enumerate(row["images"]):
            phase, view = ("before" if index < 3 else "after"), str(index % 3)
            frames.append({"frame_id": f"{phase}_{view}", "phase": phase, "view": view, **assets[_source_key(source)]})
        inputs.append({"sample_id": sample_id, "task_instruction": row["task_instruction"],
                       "subtask_instruction": row["detailed_subtask_name"], "frames": frames})
        episode = _json([row["taskvar"], row.get("episode_id")])
        labels.append({"sample_id": sample_id, "task_group": row["taskvar"].split("+", 1)[0],
                       "episode_group": hashlib.sha256(episode.encode()).hexdigest()[:24],
                       "label": label, "failure_category": category})
        provenance.append({"sample_id": sample_id, "official_row": ordinal, "taskvar": row["taskvar"],
                           "episode_id": row.get("episode_id"), "source_images": row["images"],
                           "official_failure_mode": row.get("failure_mode"), "official_failure_reason": row.get("failure_reason")})
    _write_jsonl(root / "inputs.jsonl", inputs)
    _write_jsonl(root / "labels.jsonl", labels)
    _write_jsonl(root / "private-provenance.jsonl", provenance)
    summary = {"dataset": DATASET_ID, "revision": REVISION, "source": DATASET_URL, "license": "Apache-2.0 (publisher dataset card)",
               "samples": len(inputs), "success": sum(row["label"] == "success" for row in labels),
               "failure": sum(row["label"] == "failure" for row in labels), "task_groups": len({row["task_group"] for row in labels}),
               "episode_groups": len({row["episode_group"] for row in labels}), "source_images": len(assets),
               "unique_pixel_assets": len({asset["sha256"] for asset in assets.values()}),
               "frame_order": "before slots 0,1,2; after slots 0,1,2; official list order, not filename order",
               "viewpoint_note": "Official files mix numeric views 0/1/2 and left/right/wrist. Slots are positional; front is only a legacy alias for slot 0, not a calibrated camera claim.",
               "scope": "Real-robot subtask action-effect verification; no dense motion, timestamps, or failure-onset labels.",
               "files": {name: sha256_file(root / name) for name in ("inputs.jsonl", "labels.jsonl", "private-provenance.jsonl")}}
    (root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def load_samples(root):
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET_ID or manifest.get("revision") != REVISION:
        raise ValueError("Prepared benchmark revision differs from the pinned release")
    for name in ("inputs.jsonl", "labels.jsonl"):
        if sha256_file(root / name) != manifest["files"][name]:
            raise ValueError("Prepared benchmark inputs or labels changed")
    inputs, label_rows = _read_jsonl(root / "inputs.jsonl"), _read_jsonl(root / "labels.jsonl")
    labels = {row["sample_id"]: row for row in label_rows}
    if len(labels) != len(label_rows) or len({row["sample_id"] for row in inputs}) != len(inputs) or set(labels) != {row["sample_id"] for row in inputs}:
        raise ValueError("Prepared sample identities do not match")
    return [{**row, **labels[row["sample_id"]]} for row in inputs]


def model_input(sample, root, condition="ordered", view="front"):
    """Return only inference-safe text and opaque image paths; never labels.

    Reversed is an order-corruption ablation, not a separately ground-truthed
    reverse-action task. Pixel order is swapped and positional phase labels are
    reassigned so the original ordering is not leaked back through frame IDs.
    """
    if condition not in {"ordered", "after_only", "reversed"}:
        raise ValueError("Condition must be ordered, after_only, or reversed")
    # Aliases follow the upstream generic template; prefer explicit slots 0/1/2.
    aliases = {name: key for key, name in VIEW_NAMES.items()}
    view = aliases.get(str(view), str(view))
    if view not in {"all", "0", "1", "2"}:
        raise ValueError("View must be all, 0/front, 1/left, or 2/right")
    selected = [deepcopy(frame) for frame in sample["frames"] if view == "all" or frame["view"] == view]
    before = [frame for frame in selected if frame["phase"] == "before"]
    after = [frame for frame in selected if frame["phase"] == "after"]
    expected = 3 if view == "all" else 1
    if len(before) != expected or len(after) != expected:
        raise ValueError("Missing before/after viewpoints")
    if condition == "after_only":
        selected = after
    elif condition == "reversed":
        selected = after + before
        for index, frame in enumerate(selected):
            frame["phase"] = "before" if index < expected else "after"
            frame["frame_id"] = f"{frame['phase']}_{frame['view']}"
    else:
        selected = before + after
    safe_frames = []
    for frame in selected:
        path = _safe_path(root, frame["path"])
        if not re.fullmatch(r"images/[0-9a-f]{64}\.png", frame["path"]):
            raise ValueError("Image filename could expose source annotations")
        if sha256_file(path) != frame["sha256"]:
            raise ValueError("Prepared inference image SHA256 changed")
        safe_frames.append({key: frame[key] for key in ("frame_id", "phase", "view", "sha256")} | {"path": str(path)})
    return {"task": sample["task_instruction"], "subtask": sample["subtask_instruction"], "frames": safe_frames}
