"""Atomically export a complete robot experiment after offline verification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotbench.data import load_samples, sha256_file, DATASET_ID, REVISION
from robotbench.diagnostics import counterfactual_pairs
from robotbench.run import build_report
from tools.verify_robot_release import verify, check_public_metadata, CONDITIONS, _relative
from video_media import _atomic_json

REPOSITORY = Path(__file__).resolve().parents[1]


def _copy_checked(source, destination, expected=None):
    if not source.is_file():
        raise ValueError(f"Required release source is missing: {source.name}")
    before = sha256_file(source)
    if expected is not None and before != expected:
        raise ValueError(f"Release source identity differs: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != before or sha256_file(source) != before:
        raise ValueError(f"Release source changed during copy: {source.name}")


def export(data, run, probe_run, destination):
    data, run, probe_run, destination = (Path(x).resolve() for x in (data, run, probe_run, destination))
    if destination.exists():
        raise ValueError("Choose a new export destination; existing evidence is not overwritten")
    samples = load_samples(data)
    plan = json.loads((run / "vlm-plan.json").read_text(encoding="utf-8"))
    if tuple(plan["conditions"]) != CONDITIONS or len(plan["sample_ids"]) != len(set(plan["sample_ids"])):
        raise ValueError("Invalid frozen experiment plan")
    selected = set(plan["sample_ids"])
    if not selected <= {s["sample_id"] for s in samples}:
        raise ValueError("Frozen plan selects unknown samples")
    expected_predictions = {f"{identifier}.{condition}.json" for identifier in selected for condition in CONDITIONS}
    if any(_relative(name).name != name for name in expected_predictions):
        raise ValueError("Unsafe prediction filename")
    present = {path.name for path in (run / "predictions").glob("*.json")}
    if present != expected_predictions:
        raise ValueError("Only complete prespecified prediction sets may be exported")
    cv = json.loads((probe_run / "cv.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Short sibling staging paths also avoid Windows MAX_PATH failures. The
    # TemporaryDirectory context removes only its own verified child on failure;
    # the requested destination is created only after every check succeeds.
    with tempfile.TemporaryDirectory(prefix=".rb-", dir=destination.parent) as temporary:
        temporary = Path(temporary).resolve()
        if temporary.parent != destination.parent or not temporary.name.startswith(".rb-"):
            raise ValueError("Unexpected export staging directory")
        stage = temporary / "r"
        stage.mkdir()
        for name in ("cv.json", "features.json"):
            _copy_checked(probe_run / name, stage / name)
        _copy_checked(run / "vlm-plan.json", stage / "vlm-plan.json")
        for name in sorted(expected_predictions):
            _copy_checked(run / "predictions" / name, stage / "predictions" / name)
        for name, expected in plan["code_sha256"].items():
            if _relative(name).name != name:
                raise ValueError("Unsafe protocol source filename")
            _copy_checked(run / "protocol_sources" / name, stage / "protocol_sources" / name, expected)
        _copy_checked(REPOSITORY / "robotbench" / "probe.py", stage / "protocol_sources" / "probe.py", cv["code_sha256"])
        _copy_checked(stage / "protocol_sources" / "models.py", stage / "cv-models-check.py", cv["models_code_sha256"])
        (stage / "cv-models-check.py").unlink()
        evaluation_sources = {}
        for relative in ("robotbench/metrics.py", "robotbench/diagnostics.py", "robotbench/report.py", "robotbench/run.py",
                         "tools/export_robot_run.py", "tools/verify_robot_release.py"):
            name = "evaluation_sources/" + Path(relative).name
            _copy_checked(REPOSITORY / relative, stage / name)
            evaluation_sources[name] = sha256_file(stage / name)
        for original, public in (("inputs.jsonl", "source-inputs.jsonl"), ("labels.jsonl", "source-labels.jsonl"), ("manifest.json", "source-manifest.json")):
            _copy_checked(data / original, stage / public)
        _copy_checked(REPOSITORY / "licenses" / "Apache-2.0.txt", stage / "LICENSE-DATA.txt")
        report = build_report(samples, data, stage)
        vlm = report.get("vlm")
        if not report.get("cv") or not vlm or vlm["completed_calls"] != vlm["expected_calls"]:
            raise ValueError("Only complete prespecified experiments may be exported as a release")
        diagnostics = {name: counterfactual_pairs(samples, rows) for name, rows in cv["predictions"].items()}
        _atomic_json(stage / "diagnostics.json", diagnostics)
        public_samples = []
        for sample in samples:
            row = {key: sample[key] for key in ("sample_id", "task_group", "episode_group", "label", "failure_category", "task_instruction", "subtask_instruction")}
            row["frames"] = []
            for frame in sample["frames"]:
                asset = "assets/" + frame["sha256"] + ".png"
                included = (stage / asset).is_file()
                row["frames"].append({key: frame[key] for key in ("frame_id", "phase", "view", "sha256")} | {"path": asset if included else None, "asset_included": included})
            public_samples.append(row)
        by_id = {row["sample_id"]: row for row in public_samples}
        _atomic_json(stage / "samples.json", [by_id[identifier] for identifier in plan["sample_ids"]])
        _atomic_json(stage / "evaluation-samples.json", public_samples)
        attribution = (f"Guardian UR5-Fail execution data\nPublisher: {DATASET_ID}\nRevision: {REVISION}\n"
            f"Source: https://huggingface.co/datasets/{DATASET_ID}/tree/{REVISION}/ur5fail_test\n"
            "Dataset card license: Apache-2.0; full text: LICENSE-DATA.txt (https://www.apache.org/licenses/LICENSE-2.0.txt).\n"
            "Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models, arXiv:2512.01946.\n"
            "Only selected pilot image assets are packaged. Other frame paths are null; source hashes remain available.\n"
            "Original input/label/manifest bytes support offline score recomputation without model calls; download pinned source to rerun all images.\n"
            "140 correlated rows are not 140 independent robot rollouts.\n"
            "protocol_sources/ preserves inference and CV code; evaluation_sources/ preserves scoring/report code.\n"
            "The manifest lists every payload file, excluding itself and the ZIP; the ZIP contains payload plus manifest.\n")
        (stage / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8", newline="\n")
        check_public_metadata(stage)
        entries = [{"path": p.relative_to(stage).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                   for p in sorted(stage.rglob("*")) if p.is_file()]
        _atomic_json(stage / "release-manifest.json", {"format": 2, "files": entries, "dataset_revision": REVISION,
                     "evaluation_sources": evaluation_sources, "manifest_excludes": ["release-manifest.json", "robot-benchmark.zip"]})
        archive = stage / "robot-benchmark.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and path != archive:
                    bundle.write(path, path.relative_to(stage).as_posix())
        verification = verify(stage, require_archive=True)
        result = {"files": len(entries) + 1, "zip_bytes": archive.stat().st_size, "dataset_samples": len(samples),
                  "vlm_samples": len(selected), "vlm_calls": vlm["completed_calls"], "verification": verification}
        if destination.exists():
            raise ValueError("Export destination appeared during staging; existing evidence is preserved")
        os.rename(stage, destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/robotbench"))
    parser.add_argument("--run", type=Path, default=Path("data/robot-run"))
    parser.add_argument("--probe-run", type=Path, default=Path("data/robot-run"))
    parser.add_argument("--destination", type=Path, default=Path("artifacts/robot-benchmark"))
    args = parser.parse_args()
    print(json.dumps(export(args.data, args.run, args.probe_run, args.destination), ensure_ascii=False))
