"""Import, analyze, inspect and pause persistent local video jobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from video_jobs import VideoEngine, default_data_dir


def _background_worker(root: Path) -> None:
    """Each child tries the same FileLock; redundant children exit immediately."""
    with (root / "worker.log").open("ab") as log:
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--data-dir", str(root), "_worker"],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, **options)


def _wait(engine: VideoEngine, job_id: str) -> dict:
    previous = None
    while True:
        job = engine.get_job(job_id)
        state = (job["status"], job["message"])
        if state != previous:
            print(f"{job_id} {state[0]}: {state[1]}", file=sys.stderr, flush=True)
            previous = state
        if job["status"] in {"ready", "completed", "partial", "failed", "paused"}:
            return job
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("import", help="Copy a local video and prepare evidence frames; does not run the model")
    add.add_argument("source", type=Path)
    add.add_argument("--model", default="qwen3-vl:2b")
    add.add_argument("--focus", default="Describe visible events and changes using only the sampled frames.")
    add.add_argument("--max-segments", type=int, default=4)
    add.add_argument("--max-edge", type=int, default=448)
    add.add_argument("--max-tokens", type=int, default=384)
    add.add_argument("--wait", action="store_true")
    start = commands.add_parser("start", help="Explicitly start or resume segment inference")
    start.add_argument("job_id")
    start.add_argument("--wait", action="store_true")
    for command in ("status", "pause"):
        commands.add_parser(command).add_argument("job_id")
    commands.add_parser("list")
    commands.add_parser("_worker", help="Internal background worker; normally launched automatically")
    args = parser.parse_args()
    engine = VideoEngine(args.data_dir)
    try:
        if args.command == "_worker":
            if not engine.start():
                return 0
            while engine.worker_alive:
                time.sleep(1)
            return 0
        if args.command == "import":
            job_id = engine.create_job(args.source, args.model, args.focus, args.max_segments, args.max_edge, args.max_tokens)
        elif args.command == "start":
            job_id = args.job_id
            engine.analyze(job_id)
        elif args.command == "list":
            print(json.dumps(engine.list_jobs(), ensure_ascii=False, indent=2))
            return 0
        else:
            job = engine.pause(args.job_id) if args.command == "pause" else engine.get_job(args.job_id)
            print(json.dumps(job, ensure_ascii=False, indent=2))
            return 0
        if args.wait:
            engine.start()
            job = _wait(engine, job_id)
        else:
            _background_worker(engine.root)
            job = engine.get_job(job_id)
        print(json.dumps(job, ensure_ascii=False, indent=2))
        return 1 if job["status"] in {"partial", "failed"} else 0
    except KeyboardInterrupt:
        # Preserve explicit user control: don't leave queued analysis running on Ctrl+C.
        if args.command == "start":
            engine.pause(args.job_id)
        print("Interrupted; current progress is stored in the local database.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Video job failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.stop()


if __name__ == "__main__":
    raise SystemExit(main())
