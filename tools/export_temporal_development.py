"""Preserve development protocol evidence without media paths or new inference.

Run from the repository root. Verification reuses the temporal report's
read-only validators and evaluator; prediction source snapshots are never run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotbench import temporal_evaluation as evaluation
from robotbench import temporal_report as report


RUNS = {
    "v1": ("temporal-dev-minicpm", 6),
    "v2-ready": ("temporal-dev-minicpm-v2-ready", 6),
    "v2-blocked": ("temporal-dev-minicpm-v2", 0),
}
ENVELOPE = {"bundle-manifest.json", "bundle-manifest.sha256"}


def sample_metadata(inputs, labels):
    rows = report.sample_rows(inputs, labels, {r["sample_id"]: "" for r in inputs})
    return [{k: v for k, v in row.items() if k != "video_file"} for row in rows]


def verify(bundle):
    bundle = Path(bundle)
    manifest = report.read(bundle / "bundle-manifest.json")
    if (bundle / "bundle-manifest.sha256").read_text().strip() != report.sha(bundle / "bundle-manifest.json"):
        raise ValueError("Development manifest checksum mismatch")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | ENVELOPE:
        raise ValueError("Development bundle inventory differs")
    for name, checksum in manifest["files"].items():
        path = report.safe(bundle, name)
        if report.sha(path) != checksum:
            raise ValueError("Development payload changed: " + name)
        if path.suffix in {".json", ".jsonl"}:
            value = report.rows(path) if path.suffix == ".jsonl" else report.read(path)
            report.reject_local_paths(value)
    for name, path in (("temporal_report.py", Path(report.__file__)),
                       ("temporal_evaluation.py", Path(evaluation.__file__))):
        if report.sha(path) != manifest["evaluation_source_sha256"][name]:
            raise ValueError("Use the version-matched repository evaluator")
    inputs = report.read(bundle / "input-evidence.json")
    results = {}
    for name, (_, count) in RUNS.items():
        run = bundle / name
        plan, labels = report.load_frozen_labels(run)
        if plan["split"] != "development" or len(plan["inputs"]) != 3:
            raise ValueError("Only the three fixed development examples are allowed")
        report.validate_inputs(plan, inputs)
        records = report.load_records(run, inputs, plan, complete=bool(count))
        if len(records) != count:
            raise ValueError("Unexpected development invocation count")
        if sample_metadata(inputs, labels) != report.read(run / "samples.json"):
            raise ValueError("Development metadata differs from frozen data")
        if count and evaluation.evaluate_all(labels, records, inputs) != report.read(run / "evaluation.json"):
            raise ValueError("Development evaluation does not reproduce")
        summary = {
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": report.sha(run / "plan.json"),
            "protocol_version": plan["protocol"]["version"],
            "adapter_source_sha256": plan["protocol"]["adapter_source_sha256"],
            "prediction_source_sha256": plan["protocol"]["source_sha256"],
            "records": len(records),
            "status_counts": dict(Counter(r["status"] for r in records)),
            "prediction_counts": dict(Counter(r["prediction"] for r in records)),
        }
        if summary != manifest["runs"][name]:
            raise ValueError("Development summary differs")
        results[name] = summary
    return {"verified": True, "runs": results, "inference_calls": 0, "network_calls": 0,
            "payload_files": len(manifest["files"])}


def export(dataset, source_root, destination):
    from robotbench.temporal_data import load_inputs

    if destination.exists():
        return verify(destination)
    inputs = report.portable_inputs(load_inputs(dataset, split="development"))
    stage = destination.with_name("." + destination.name + ".staging")
    stage.mkdir(parents=True, exist_ok=True)
    report.save(stage / "input-evidence.json", report.json_bytes(inputs), frozen=True)
    summaries = {}
    copied_originals = {}

    def copy(source, relative):
        content = source.read_bytes()
        report.save(stage / relative, content, frozen=True)
        # Each entry records a byte-identical source copy, not a reconstruction.
        copied_originals[relative] = report.sha(source)

    for name, (directory, count) in RUNS.items():
        run = source_root / directory
        plan, labels = report.load_frozen_labels(run)
        report.validate_inputs(plan, inputs)
        records = report.load_records(run, inputs, plan, complete=bool(count))
        if plan["split"] != "development" or len(records) != count:
            raise ValueError("Unexpected development source")
        for relative in ["plan.json", *("dataset_snapshot/" + n for n in report.SNAPSHOTS),
                         *("prediction_sources/" + n for n in report.PREDICTION_SOURCES)]:
            copy(run / relative, name + "/" + relative)
        for record in records:
            stem = record["sample_id"] + "--" + record["method"]
            for relative in ("predictions/" + stem + ".json", "panels/" + stem + ".jpg"):
                copy(run / relative, name + "/" + relative)
        if count:
            if evaluation.evaluate_all(labels, records, inputs) != report.read(run / "evaluation.json"):
                raise ValueError("Original development evaluation does not reproduce")
            copy(run / "evaluation.json", name + "/evaluation.json")
        report.save(stage / name / "samples.json", report.json_bytes(sample_metadata(inputs, labels)), frozen=True)
        summaries[name] = {
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": report.sha(run / "plan.json"),
            "protocol_version": plan["protocol"]["version"],
            "adapter_source_sha256": plan["protocol"]["adapter_source_sha256"],
            "prediction_source_sha256": plan["protocol"]["source_sha256"],
            "records": len(records),
            "status_counts": dict(Counter(r["status"] for r in records)),
            "prediction_counts": dict(Counter(r["prediction"] for r in records)),
        }

    sources = {"temporal_report.py": Path(report.__file__), "temporal_evaluation.py": Path(evaluation.__file__)}
    for name, source in sources.items():
        copy(source, "evaluation_sources/" + name)
    readme = """# Development protocol evidence — excluded from test scores

These are the same three development episodes (two failures, one success),
from two task groups excluded entirely from the 16-episode test subset.
Each completed version has 3 episodes × 2 samplers, not six independent episodes.
All records, exact model-input JPEG panels, plans, source snapshots, annotation
snapshots and completed evaluations are copied byte for byte. samples.json is
rebuilt as metadata only: local file URLs and machine paths are omitted.
input-evidence.json preserves numeric inputs needed to verify frozen identities.
Original HDF5 files and full original videos are not duplicated.

- v1: six raw responses rejected by the strict independent class/frame parser.
  Failed records remain abstentions in the saved evaluation. Raw responses and
  errors remain available under predictions/; none was silently repaired.
- v2-blocked: the first joint-enum plan was stopped by source-identity checks
  before any inference. Its frozen plan/data/source snapshots are retained.
  There are zero prediction records and no invented evaluation for this plan.
- v2-ready: after the adapter source was stable, six responses were structurally
  valid and all predicted success. Each sampler is correct on 1/3 episodes,
  detects 0/2 failures and has Hit@2s 0/2. Format repair did not establish
  failure-detection ability. These findings refer to this model and protocol.

The v1 independent fields were replaced before test inference by one joint
decision enum. Its 12 failure-location choices and one success/abstain choice
are not generation-symmetric; changes in class distribution cannot be
attributed solely to visual reasoning. No test outputs were used for this change.

Verify from the version-matched repository (standard library only):

    python -B tools/export_temporal_development.py --verify --out artifacts/robot-temporal-development

The verifier checks file inventories/hashes, plan and source hashes, numeric
input identities, author-label reconstruction, raw output parsing, panel hashes,
and completed evaluations through the existing report validators. It does not
execute prediction_sources, call a model, decode videos or contact the network.
evaluation_sources are the evaluator/report versions used to validate this
export, separate from the original inference snapshots. Checksums establish
consistency, not authorship or independent reproduction of model inference.
"""
    attribution = """FailTime / selected development episodes and model-input panels
Paper: ProTracer: Proprioception-Guided Failure Diagnosis in Robot Manipulation
Authors: Chang Dong; Mehdi Hosseinzadeh; King Hang Wong; Lingqiao Liu;
Francois Fraysse; Feras Dayoub; Minh Hoai Nguyen (2026).
Dataset: ChangUoA/FailTime
Revision: 07aea9142616cf1f8047512bd53a4d9fcd5e7cee
Dataset card/citation: https://huggingface.co/datasets/ChangUoA/FailTime/blob/07aea9142616cf1f8047512bd53a4d9fcd5e7cee/README.md
License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
Terms: https://creativecommons.org/licenses/by/4.0/legalcode
Original episode identities, author annotations and source hashes are retained
in each dataset_snapshot/plan.json. Frames were selected, resized and composed
with frame/time labels for inference; those exact input JPEGs are unchanged in
this export. Metadata-only sample indices are reconstructed without local URLs.
No endorsement implied. No upstream ProTracer implementation was copied.
"""
    report.save(stage / "README.md", readme.encode(), frozen=True)
    report.save(stage / "DATA_ATTRIBUTION.txt", attribution.encode(), frozen=True)
    payload = {p.relative_to(stage).as_posix(): report.sha(p) for p in stage.rglob("*") if p.is_file() and p.name not in ENVELOPE}
    manifest = {
        "schema_version": 1, "scope": "development_only_not_test", "runs": summaries,
        "files": dict(sorted(payload.items())), "original_byte_copies": dict(sorted(copied_originals.items())),
        "evaluation_source_sha256": {name: report.sha(path) for name, path in sources.items()},
        "excluded_envelope_files": sorted(ENVELOPE),
    }
    report.save(stage / "bundle-manifest.json", report.json_bytes(manifest), frozen=True)
    report.save(stage / "bundle-manifest.sha256", (report.sha(stage / "bundle-manifest.json") + "\n").encode(), frozen=True)
    result = verify(stage)
    stage.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/temporalbench")
    parser.add_argument("--source-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/robot-temporal-development")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify(args.out) if args.verify else export(args.data, args.source_root, args.out)
    print(report.json_bytes(result).decode(), end="")


if __name__ == "__main__":
    main()
