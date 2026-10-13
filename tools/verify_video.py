"""Verify a stored video job and its portable evidence package; never runs a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_video import SegmentObservation, validate_evidence
from video_jobs import VideoEngine
from video_report import export_job

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_id")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--export-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paused-snapshot", type=Path)
    args = p.parse_args()
    engine = VideoEngine(args.data_dir)
    job = engine.get_job(args.job_id)
    if job["status"] != "completed":
        raise ValueError(f"Job must be completed for this verification, got {job['status']}")
    checks = {}
    checks["source_hash"] = sha(job["source_path"]) == job["video"]["source_sha256"]
    checks["preview_hash"] = sha(job["video"]["preview_path"]) == job["video"]["preview_sha256"]
    checks["all_segments_completed"] = all(s["status"] == "ok" for s in job["segments"])
    checks["evidence_hashes"] = all(sha(f["path"]) == f["sha256"] for s in job["segments"] for f in s["frames"])
    for segment in job["segments"]:
        structured = SegmentObservation.model_validate_json(segment["result"]["raw_response"]).model_dump()
        validate_evidence(structured, segment["frames"])
        if segment["result"]["digest"] != job["identity"]["digest"]:
            raise ValueError("Segment model digest does not match job identity")
    checks["schemas_and_references"] = True
    checks["model_digest_consistent"] = True
    if args.paused_snapshot:
        paused = json.loads(args.paused_snapshot.read_text(encoding="utf-8-sig"))
        if paused["id"] != job["id"] or paused["status"] != "paused":
            raise ValueError("Paused snapshot does not belong to this job")
        by_id = {s["id"]: s for s in job["segments"]}
        previous = [s for s in paused["segments"] if s["status"] == "ok"]
        checks["completed_segments_reused_after_pause"] = bool(previous) and all(
            by_id[s["id"]]["attempts"] == s["attempts"] and by_id[s["id"]]["result"] == s["result"]
            for s in previous)
    attribution_path = Path(__file__).resolve().parents[1] / "examples/video/source.json"
    if attribution_path.exists():
        attribution = json.loads(attribution_path.read_text(encoding="utf-8-sig"))
        if attribution.get("original_file", {}).get("sha256") == job["video"]["source_sha256"]:
            job["source_attribution"] = attribution
    paths = export_job(job, args.export_dir or engine.root / "exports")
    with zipfile.ZipFile(paths[0]) as bundle:
        manifest = json.loads(bundle.read("evidence.json"))
        checks["zip_preview_is_normalized_video"] = hashlib.sha256(bundle.read("video.mp4")).hexdigest() == job["video"]["preview_sha256"]
        checks["zip_contains_all_evidence"] = all(
            hashlib.sha256(bundle.read(f["path"])).hexdigest() == f["sha256"]
            for s in manifest["segments"] for f in s["frames"])
        checks["portable_manifest"] = "source_path" not in manifest and manifest["video"]["preview_path"] == "video.mp4"
        checks["report_and_csv_present"] = {"report.html", "timeline.csv"}.issubset(bundle.namelist())
    result = {
        "job_id": job["id"], "status": job["status"], "checks": checks,
        "all_checks_passed": all(checks.values()), "model": job["identity"],
        "source_sha256": job["video"]["source_sha256"], "preview_sha256": job["video"]["preview_sha256"],
        "video_seconds": job["video"]["duration_seconds"], "selection": job["selection"],
        "segments": [{"id": s["id"], "attempts": s["attempts"], "status": s["status"],
                      "sample_seconds": [f["time_seconds"] for f in s["frames"]],
                      "elapsed_seconds": s["result"]["elapsed_seconds"],
                      "metrics": s["result"]["metrics"]} for s in job["segments"]],
        "total_model_seconds": round(sum(s["result"]["elapsed_seconds"] for s in job["segments"]), 3),
        "exports": paths,
        "content_accuracy": "Not scored: schema validity is not semantic correctness. See the independent visual review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["all_checks_passed"], "output": str(args.output), "exports": paths}, ensure_ascii=False))
    return 0 if result["all_checks_passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())

