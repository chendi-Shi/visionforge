"""Fetch the optional public sample, accepting only the recorded original bytes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


METADATA = Path(__file__).resolve().parents[1] / "examples" / "video" / "source.json"


def validate_file(path: str | Path, size_bytes: int, sha256: str) -> bool:
    """Return False if absent, True if verified; reject an existing wrong file."""
    path = Path(path)
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != size_bytes:
        raise ValueError(f"Existing sample has an unexpected size; refusing to overwrite: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError(f"Existing sample SHA256 mismatch; refusing to overwrite: {path.name}")
    return True


def fetch_sample(metadata_path: str | Path = METADATA) -> Path:
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata["original_file"]
    size_bytes, sha256 = expected["size_bytes"], expected["sha256"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("Sample metadata must specify a positive byte count.")
    if not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("Sample metadata must specify the expected SHA256.")
    filename = metadata["local_file"]
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Sample filename must stay inside the metadata directory.")
    destination = metadata_path.resolve().parent / filename
    if validate_file(destination, size_bytes, sha256):
        print(f"Verified existing sample; skipped download: {destination.name}")
        return destination
    url = metadata["original_file_url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org" or parsed.username or parsed.password:
        raise ValueError("The recorded original must use the official Wikimedia HTTPS download host.")
    temporary: Path | None = None
    try:
        request = Request(url, headers={"User-Agent": "VisionForge-public-sample/1.0 (SHA256-verified sample downloader)"})
        with urlopen(request, timeout=60) as response:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https" or final_url.hostname != "upload.wikimedia.org":
                raise ValueError("Sample download redirected outside the official HTTPS host.")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != size_bytes:
                raise ValueError("Download byte count differs from the recorded sample.")
            with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, prefix=filename + ".", suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                digest, written = hashlib.sha256(), 0
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > size_bytes:
                        raise ValueError("Download exceeded the expected byte count.")
                    digest.update(chunk)
                    handle.write(chunk)
                if written != size_bytes or digest.hexdigest() != sha256:
                    raise ValueError("Downloaded sample failed the expected size or SHA256 check.")
                handle.flush()
                os.fsync(handle.fileno())
        # A concurrently created file must also verify before anything is replaced.
        if validate_file(destination, size_bytes, sha256):
            return destination
        os.replace(temporary, destination)
        temporary = None
        print(f"Downloaded and verified {size_bytes:,} bytes: {destination.name}")
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    try:
        fetch_sample()
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Sample download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
