"""Run a small, resumable image QA evaluation against the local Ollama model.

The bundled examples are synthetic smoke tests, not a public benchmark or an
estimate of real-world accuracy. Exact match is deliberately conservative.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import unicodedata
from typing import Any, Callable


SCHEMA_VERSION = 1
SCORING_VERSION = "nfkc-casefold-whitespace-exact-v1"
DISCLAIMER = (
    "Bundled examples are synthetic smoke tests, not a public benchmark or "
    "evidence of real-world generalization. Exact match only checks the supplied "
    "reference answers; review each image, question, and model response manually."
)


def normalize_answer(answer: str) -> str:
    """Normalize Unicode, case and whitespace, without guessing semantic meaning."""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    return " ".join(unicodedata.normalize("NFKC", answer).casefold().split())


def score_answer(answer: str, accepted_answers: list[str]) -> bool | None:
    """None denotes an unlabelled example; punctuation and extra prose still count."""
    if not accepted_answers:
        return None
    return normalize_answer(answer) in {normalize_answer(a) for a in accepted_answers}


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_request(item: dict[str, Any], image_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Everything that affects inference or grading participates in cache identity."""
    return {
        "id": item["id"],
        "image": item["image"],
        "image_sha256": file_sha256(image_path),
        "question": item["question"],
        "accepted_answers": item.get("accepted_answers", []),
        "config": config,
    }


def seal_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if key != "record_sha256"}
    result["record_sha256"] = _digest(result)
    return result


def is_reusable_record(record: Any, request: dict[str, Any]) -> bool:
    """Reject incomplete, mismatched, altered or malformed checkpoint entries."""
    if not isinstance(record, dict):
        return False
    try:
        result = record["result"]
        elapsed = record["wall_seconds"]
        return (
            record.get("complete") is True
            and record.get("call_succeeded") is True
            and record.get("request") == request
            and record.get("cache_key") == _digest(request)
            and record.get("record_sha256") == seal_record(record)["record_sha256"]
            and isinstance(result, dict)
            and result.get("status") == "ok"
            and result.get("digest") == request["config"]["model_digest"]
            and isinstance(result.get("answer"), str)
            and bool(result["answer"].strip())
            and isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and math.isfinite(elapsed)
            and elapsed >= 0
            and record.get("exact_match") == score_answer(result["answer"], request["accepted_answers"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def load_manifest(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must be a non-empty JSON list")
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {index} must be an object")
        for key in ("id", "image", "question"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Manifest item {index}: {key} must be a non-empty string")
        if item["id"] in seen:
            raise ValueError(f"Duplicate example id: {item['id']}")
        seen.add(item["id"])
        if Path(item["image"]).is_absolute():
            raise ValueError(f"{item['id']}: image must be relative to the manifest directory")
        accepted = item.get("accepted_answers", [])
        if not isinstance(accepted, list) or any(not isinstance(a, str) or not a.strip() for a in accepted):
            raise ValueError(f"{item['id']}: accepted_answers must be a list of non-empty strings")
    return items


def summarize(records: list[dict[str, Any]], selected_count: int) -> dict[str, Any]:
    labelled = [r for r in records if r["request"].get("accepted_answers")]
    successful_labelled = [r for r in labelled if r["call_succeeded"]]
    exact = sum(r["exact_match"] is True for r in labelled)
    successes = sum(r["call_succeeded"] for r in records)
    return {
        "selected_examples": selected_count,
        "recorded_examples": len(records),
        "pending_examples": selected_count - len(records),
        "call_successes": successes,
        "call_success_rate": successes / len(records) if records else None,
        "complete_answers": sum(r["complete"] for r in records),
        "labelled_examples": len(labelled),
        "exact_matches": exact,
        "exact_match_accuracy_all_labelled": exact / len(labelled) if labelled else None,
        "exact_match_accuracy_successful_calls": exact / len(successful_labelled) if successful_labelled else None,
        "cached_results": sum(r.get("reused", False) for r in records),
        "newly_attempted": sum(not r.get("reused", False) for r in records),
        "manual_review_required": True,
        "metric_notes": (
            "Call success requires status=ok and a non-empty answer; truncated/invalid outputs count "
            "as failures. It does not mean the answer is correct. Exact match matches the "
            "reference after Unicode/case/whitespace normalization. Errors and incomplete answers count "
            "as incorrect in all-labelled accuracy. Unlabelled examples are excluded."
        ),
    }


def _write_checkpoint(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_evaluation(
    manifest: Path,
    output: Path,
    *,
    model: str,
    model_digest: str,
    max_edge: int = 768,
    max_tokens: int = 32,
    limit: int = 3,
    force: bool = False,
    analyzer: Callable[..., dict[str, Any]],
    implementation_sha256: str,
) -> dict[str, Any]:
    if max_edge <= 0 or max_tokens <= 0 or limit <= 0:
        raise ValueError("max_edge, max_tokens and limit must be positive")
    if not isinstance(model_digest, str) or not model_digest.strip():
        raise ValueError("A verified local model digest is required for safe checkpoint reuse")
    started = time.perf_counter()
    items = load_manifest(manifest)[:limit]
    config = {
        "model": model,
        "model_digest": model_digest,
        "max_edge": max_edge,
        "max_tokens": max_tokens,
        "mode": "qa",
        "implementation_sha256": implementation_sha256,
        "scoring_version": SCORING_VERSION,
    }
    cache: dict[str, Any] = {}
    if output.exists() and not force:
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
            if previous.get("schema_version") == SCHEMA_VERSION and isinstance(previous.get("cache"), dict):
                cache = previous["cache"]
        except (OSError, ValueError, AttributeError) as exc:
            print(f"Ignoring unreadable checkpoint: {exc}", file=sys.stderr)
    records: list[dict[str, Any]] = []
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "disclaimer": DISCLAIMER,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.resolve()),
        "config": config,
        "results": records,
        "cache": cache,
        "interrupted": False,
    }

    def checkpoint() -> None:
        document["session_wall_seconds"] = round(time.perf_counter() - started, 6)
        document["summary"] = summarize(records, len(items))
        document["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_checkpoint(output, document)

    try:
        for item in items:
            image_path = (manifest.parent / item["image"]).resolve()
            attempted_at = time.perf_counter()
            request: dict[str, Any] = {
                "id": item["id"], "image": item["image"], "question": item["question"],
                "accepted_answers": item.get("accepted_answers", []), "config": config,
                "image_sha256": None,
            }
            try:
                request = make_request(item, image_path, config)
                key = _digest(request)
                cached = cache.get(key)
                if not force and is_reusable_record(cached, request):
                    record = seal_record({**cached, "reused": True})
                    records.append(record)
                    print(f"[{len(records)}/{len(items)}] {item['id']}: cached", flush=True)
                    checkpoint()
                    continue
                result = analyzer(image_path, item["question"], model=model, mode="qa", max_edge=max_edge, max_tokens=max_tokens)
                if not isinstance(result, dict) or result.get("status") not in {"ok", "truncated", "invalid_output", "error"}:
                    raise ValueError("Analyzer returned an invalid response/status")
                # Reject unserializable/NaN values here so the individual error is still recorded.
                _digest(result)
                if result.get("digest") != model_digest:
                    raise ValueError(
                        "Model digest mismatch: expected "
                        f"{model_digest!r}, received {result.get('digest')!r}; "
                        "the response will not be scored or cached"
                    )
                complete = result["status"] == "ok" and isinstance(result.get("answer"), str) and bool(result["answer"].strip())
                call_succeeded = complete
                expected = request["accepted_answers"]
                exact_match = score_answer(result["answer"], expected) if complete else (False if expected else None)
                record = {
                    "request": request, "cache_key": key, "result": result,
                    "call_succeeded": call_succeeded, "complete": complete,
                    "exact_match": exact_match, "reused": False,
                    "wall_seconds": round(time.perf_counter() - attempted_at, 6),
                }
            except Exception as exc:
                record = {
                    "request": request, "cache_key": _digest(request),
                    "result": {"status": "error", "answer": None, "error": {"type": type(exc).__name__, "message": str(exc)}},
                    "call_succeeded": False, "complete": False,
                    "exact_match": False if request["accepted_answers"] else None,
                    "reused": False, "wall_seconds": round(time.perf_counter() - attempted_at, 6),
                }
            record = seal_record(record)
            records.append(record)
            if record["complete"]:
                cache[record["cache_key"]] = record
            print(f"[{len(records)}/{len(items)}] {item['id']}: {record['result']['status']}, exact_match={record['exact_match']}", flush=True)
            checkpoint()
    except KeyboardInterrupt:
        document["interrupted"] = True
        checkpoint()
        raise
    checkpoint()
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("examples/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--model", help="Defaults to core.DEFAULT_MODEL")
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Ignore prior completed results and rerun selected examples")
    args = parser.parse_args()
    try:
        from core import analyze_image, DEFAULT_MODEL
        import core

        # Resolve the current local content identity before considering any cache.
        # A tag can point at different weights after `ollama pull`.
        client = core._create_client()
        try:
            model, model_digest = core._check_model(client, args.model or DEFAULT_MODEL)
        except ValueError:
            raise
        except Exception as exc:
            raise core._call_error(exc) from None
        finally:
            client.close()
        document = run_evaluation(
            args.manifest, args.output, model=model, model_digest=model_digest,
            max_edge=args.max_edge, max_tokens=args.max_tokens, limit=args.limit,
            force=args.force, analyzer=analyze_image,
            implementation_sha256=file_sha256(Path(core.__file__)),
        )
    except KeyboardInterrupt:
        print("Interrupted; completed answers are saved. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(document["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}\n{DISCLAIMER}")
    return 0 if document["summary"]["complete_answers"] == document["summary"]["selected_examples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
