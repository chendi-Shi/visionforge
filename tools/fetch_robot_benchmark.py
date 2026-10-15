"""Download only the pinned UR5-Fail split; keep its assets outside public Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotbench.data import DATASET_ID, PINNED_FILES, REVISION, prepare_dataset, verify_download


def download(relative, folder):
    spec = PINNED_FILES[relative]
    path = folder / Path(relative).name
    if path.exists():
        verify_download(path, spec)
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{relative}",
                                     headers={"User-Agent": "VisionForge-RobotBench/1.0"})
    read, reported = 0, 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                read += len(block)
                if read > spec["size"]:
                    raise ValueError("Download exceeds pinned size")
                output.write(block)
                if read - reported >= 16 * 1024 * 1024:
                    print(f"{path.name}: {read}/{spec['size']} bytes", flush=True)
                    reported = read
        verify_download(temporary, spec)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/robotbench"))
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    downloaded = {relative: download(relative, raw) for relative in PINNED_FILES}
    summary = prepare_dataset(downloaded["ur5fail_test/metadata_execution.jsonl"],
                              downloaded["ur5fail_test/internVL_dataset_execution_vanilla.jsonl"],
                              downloaded["ur5fail_test/records.tar.gz"], root)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
