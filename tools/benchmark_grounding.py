"""Serial A/B grounding benchmark; explicit labels are separate from schema success.

Manifest: JSON list of {id,image,time_seconds,target,expected_verdict?,sha256?}.
Image paths are relative to the manifest directory. A nested `frame` object with
id/path/time_seconds/sha256 is also accepted. No label means no accuracy score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core
import core_grounding
from core_grounding import verify_frame


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def load_cases(path: Path, limit: int | None) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a non-empty JSON list")
    cases, ids = [], set()
    for row in rows[:limit]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Each case requires a unique string id")
        ids.add(row["id"])
        if not isinstance(row.get("target"), str) or not row["target"].strip():
            raise ValueError("Each case requires a target")
        expected = row.get("expected_verdict")
        if expected is not None and expected not in {"visible", "not_visible", "uncertain"}:
            raise ValueError("expected_verdict must be visible, not_visible, uncertain, or absent")
        frame = dict(row.get("frame") or {"id": row["id"], "path": row.get("image"), "time_seconds": row.get("time_seconds"), "sha256": row.get("sha256")})
        image = Path(frame.get("path") or "")
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise ValueError(f"Missing case image: {row['id']}")
        actual = hashlib.sha256(image.read_bytes()).hexdigest()
        frame["path"] = str(image)
        frame["sha256"] = frame.get("sha256") or actual
        cases.append({"id": row["id"], "frame": frame, "actual_image_sha256": actual,
                      "target": row["target"], "expected_verdict": expected})
    return cases


def summarize(records: list[dict], models: list[str]) -> dict:
    summary = {}
    for model in models:
        rows = [r for r in records if r["model_requested"] == model]
        valid = [r for r in rows if r["result"]["status"] == "ok"]
        labelled = [r for r in rows if r["expected_verdict"] is not None]
        correct = sum(r["exact_match"] is True for r in labelled)
        summary[model] = {
            "recorded_cases": len(rows), "schema_successes": len(valid),
            "schema_success_rate": len(valid) / len(rows) if rows else None,
            "labelled_cases": len(labelled), "verdict_matches": correct,
            "verdict_accuracy_all_labelled_attempts": correct / len(labelled) if labelled else None,
            "uncertain_verdicts": sum(r["result"]["structured"]["verdict"] == "uncertain" for r in valid),
            "false_visible_on_labelled_negative": sum(r["expected_verdict"] == "not_visible" and r["result"]["structured"]["verdict"] == "visible" for r in valid),
            "sum_recorded_call_seconds": round(sum(r["result"]["elapsed_seconds"] for r in rows), 6),
            "cached_cases": sum(r.get("reused", False) for r in rows),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/grounding-benchmark.json"))
    parser.add_argument("--models", nargs="+", default=["qwen3-vl:2b", "minicpm-v4.6:1b"])
    parser.add_argument("--max-edge", type=int, default=640)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started, records, errors, cache = time.perf_counter(), [], [], {}
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        if not 224 <= args.max_edge <= 768 or not 32 <= args.max_tokens <= 1024:
            raise ValueError("max-edge must be 224–768 and max-tokens 32–1024")
        cases = load_cases(args.manifest, args.limit)
        config = {"max_edge": args.max_edge, "max_tokens": args.max_tokens,
                  "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                  "grounding_sha256": hashlib.sha256(Path(core_grounding.__file__).read_bytes()).hexdigest()}
        if args.output.exists() and not args.force:
            old = json.loads(args.output.read_text(encoding="utf-8"))
            cache = {r["fingerprint"]: r for r in old.get("records", []) if isinstance(r, dict) and "fingerprint" in r}
        document = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config, "cases": cases, "models": args.models, "records": records, "model_errors": errors,
                    "metric_note": "Schema success is not accuracy. Verdict accuracy uses only explicitly supplied reference labels, counts failed attempts as incorrect, and does not grade the reason text. Missing model runs are listed separately. Manual visual review is required."}

        def checkpoint():
            document["session_wall_seconds"] = round(time.perf_counter() - started, 6)
            document["summary"] = summarize(records, args.models)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(temporary, args.output)

        try:
            for model in args.models:
                client = core._create_client()
                try:
                    resolved, model_digest = core._check_model(client, model)
                    if not model_digest:
                        raise ValueError("Local model digest is required")
                except Exception as exc:
                    errors.append({"model": model, "error": f"{type(exc).__name__}: {exc}", "unattempted_cases": len(cases)})
                    checkpoint()
                    continue
                finally:
                    client.close()
                for case in cases:
                    request = {"case": case, "model": resolved, "model_digest": model_digest, "config": config}
                    fingerprint = digest(request)
                    old = cache.get(fingerprint)
                    reusable = old and old.get("request") == request and old.get("result", {}).get("status") == "ok" and old.get("result", {}).get("digest") == model_digest and old.get("result_sha256") == digest(old["result"])
                    if reusable and not args.force:
                        expected = case["expected_verdict"]
                        matched = old["result"]["structured"]["verdict"] == expected if expected is not None else None
                        record = {**old, "model_requested": model, "expected_verdict": expected,
                                  "exact_match": matched, "reused": True}
                    else:
                        result = verify_frame(case["frame"], case["target"], model=resolved, max_edge=args.max_edge, max_tokens=args.max_tokens)
                        if result["digest"] != model_digest and result["status"] == "ok":
                            result = {**result, "status": "failed", "error": "Model digest changed during this benchmark; create a fresh run.", "structured": None}
                        expected = case["expected_verdict"]
                        matched = result["status"] == "ok" and result["structured"]["verdict"] == expected if expected is not None else None
                        record = {"case_id": case["id"], "model_requested": model, "fingerprint": fingerprint,
                                  "request": request, "result": result, "result_sha256": digest(result),
                                  "expected_verdict": expected, "exact_match": matched, "reused": False}
                    records.append(record)
                    checkpoint()
                    print(f"{model} / {case['id']}: {record['result']['status']}, reference_match={record['exact_match']}", flush=True)
        except KeyboardInterrupt:
            document["interrupted"] = True
            checkpoint()
            raise
        checkpoint()
        print(json.dumps(document["summary"], ensure_ascii=False, indent=2))
        return 1 if errors or any(r["result"]["status"] != "ok" for r in records) else 0
    except KeyboardInterrupt:
        print("Interrupted; completed results are saved for the same configuration.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
