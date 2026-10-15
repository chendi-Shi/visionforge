"""Repair one pinned official Qwen3.5 blob via independently checked ranges.

Does not stop/restart Ollama or consume its inference engine. Use only after the
original pull client has stopped. Signed redirect URLs are never logged.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DIGEST = "a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50"
SIZE = 2648593216
CHUNK = 8 * 1024 * 1024
REGISTRY = "https://registry.ollama.ai/v2/library/qwen3.5"
R2_HOST = "dd20bb891979d25aebc8bec07b2b3bbc.r2.cloudflarestorage.com"


class CurlRangeResponse:
    """Range transport with bounded lifetime; signed redirects stay private."""
    def __init__(self, request, stage, r2_ip=None):
        self.directory = tempfile.TemporaryDirectory(prefix=".cr-", dir=stage)
        folder = Path(self.directory.name)
        if folder.resolve().parent != Path(stage).resolve():
            raise ValueError("Range temporary directory escaped download stage")
        body, headers = folder / "body", folder / "headers"
        try:
            command = [
                "curl.exe" if os.name == "nt" else "curl", "--silent", "--location",
                "--proto", "=https", "--proto-redir", "=https", "--max-time", "45",
                "--range", request.get_header("Range").removeprefix("bytes="),
                "--header", "Accept-Encoding: identity", "--output", str(body),
                "--dump-header", str(headers), "--write-out", "%{http_code}", request.full_url,
            ]
            if r2_ip:
                command[1:1] = ["--resolve", f"{R2_HOST}:443:{r2_ip}"]
            result = subprocess.run(command, capture_output=True, timeout=55)
            # curl writes every redirect header; only the final response binds
            # bytes. Neither stderr nor redirect URLs enter logs or receipts.
            self.status, self.headers = 0, {}
            for block in headers.read_text(encoding="iso-8859-1").split("\n\n"):
                lines = block.strip().splitlines()
                if not lines or not lines[0].startswith("HTTP/"):
                    continue
                self.status = int(lines[0].split()[1])
                self.headers = {name.strip().lower(): value.strip() for name, value in
                                (line.split(":", 1) for line in lines[1:] if ":" in line)}
            if self.status != 206 or result.stdout.strip() != b"206":
                raise OSError("curl range response was not HTTP 206")
            self.headers = {"Content-Range": self.headers.get("content-range")}
            self.returncode = result.returncode
            self.body = body.open("rb")
        except Exception:
            self.directory.cleanup()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.body.close()
        self.directory.cleanup()

    def read1(self, size):
        value = self.body.read1(size)
        if not value and self.returncode:
            raise OSError("curl range interrupted; validated partial bytes retained")
        return value

    read = read1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def save_model_manifest():
    """Preserve the API-addressed registry bytes and any local wrapper verbatim.

    Ollama 0.40.4 adds provenance layers to its manifests/... file while /tags
    retains the original registry digest. That original object is retained in
    blobs; removing layers then serializing JSON would not preserve its bytes.
    """
    import core
    with core._create_client() as client:
        name, model_digest, _ = core._inspect_model(client, "qwen3.5:4b")
    models = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))
    source = models / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "4b"
    local_raw = source.read_bytes()
    local_checksum = hashlib.sha256(local_raw).hexdigest()
    expected = model_digest.removeprefix("sha256:")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Registered model digest does not match exact manifest bytes")
    raw, origin = local_raw, "local_manifest"
    if local_checksum != expected:
        wrapper = json.loads(local_raw)
        manifest_type = "application/vnd.docker.distribution.manifest.v2+json"
        list_type = "application/vnd.ollama.manifest.list.v2+json"
        references = [layer for layer in wrapper.get("layers", [])
                      if layer.get("mediaType") == manifest_type and layer.get("digest") == "sha256:" + expected]
        if len(references) != 1:
            raise ValueError("Registered model digest does not match exact manifest bytes")
        raw = (models / "blobs" / ("sha256-" + expected)).read_bytes()
        if len(raw) != references[0].get("size") or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Retained registry manifest size/SHA256 differs from wrapper/API")
        original = json.loads(raw)
        runtime_layers = [layer for layer in wrapper["layers"]
                          if layer.get("mediaType") not in (manifest_type, list_type)]
        if (runtime_layers != original.get("layers")
                or any(wrapper.get(key) != original.get(key) for key in
                       ("schemaVersion", "mediaType", "config", "runner", "format"))):
            raise ValueError("Local wrapper changes registry model runtime configuration")
        lists = [layer for layer in wrapper["layers"] if layer.get("mediaType") == list_type]
        if len(lists) > 1 or sum(layer.get("mediaType") == manifest_type for layer in wrapper["layers"]) != 1:
            raise ValueError("Ambiguous local manifest provenance layers")
        for layer in lists:
            list_digest = layer.get("digest", "").removeprefix("sha256:")
            if len(list_digest) != 64 or any(c not in "0123456789abcdef" for c in list_digest):
                raise ValueError("Invalid retained manifest-list digest")
            list_raw = (models / "blobs" / ("sha256-" + list_digest)).read_bytes()
            if len(list_raw) != layer.get("size") or hashlib.sha256(list_raw).hexdigest() != list_digest:
                raise ValueError("Retained manifest-list size/SHA256 differs from wrapper")
            if not any(item.get("digest") == "sha256:" + expected
                       and item.get("runner") == original.get("runner")
                       and item.get("format") == original.get("format")
                       for item in json.loads(list_raw).get("manifests", [])):
                raise ValueError("Retained manifest-list does not select registered runtime")
        origin = "local_retained_registry_blob"
    checksum = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if not any(layer.get("digest") == "sha256:" + DIGEST and layer.get("size") == SIZE for layer in manifest["layers"]):
        raise ValueError("Registered model manifest does not contain the pinned repaired blob")
    target = ROOT / "data" / "qwen35-model-manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper_target = target.with_name("qwen35-model-manifest.local.json")
    for destination, content in ((target, raw), (wrapper_target, local_raw)):
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError("Refusing to replace a different saved model manifest")
    for destination, content in ((target, raw), (wrapper_target, local_raw)):
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
    result = {"model": name, "model_digest": model_digest, "manifest_sha256": checksum,
              "manifest_bytes": len(raw), "raw_bytes_preserved": True, "manifest_origin": origin,
              "local_wrapper_sha256": local_checksum, "local_wrapper_bytes": len(local_raw)}
    write_json(target.with_name("qwen35-manifest-provenance.json"), result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8, 16), default=2)
    parser.add_argument("--transport", choices=("urllib", "curl"), default="urllib")
    parser.add_argument("--r2-ip", help="Use one currently DNS-advertised official R2 address; TLS still verifies its hostname")
    args = parser.parse_args()
    if args.r2_ip:
        advertised = {row[4][0] for row in socket.getaddrinfo(R2_HOST, 443, type=socket.SOCK_STREAM)}
        if args.transport != "curl" or args.r2_ip not in advertised:
            raise ValueError("R2 address must be a current official DNS result and use curl transport")
    stage = ROOT / "data" / "model-download" / DIGEST[:12]
    stage.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(REGISTRY + "/manifests/4b", timeout=30) as response:
        manifest = json.load(response)
    expected = [layer for layer in manifest["layers"] if layer["digest"] == "sha256:" + DIGEST]
    if len(expected) != 1 or expected[0]["size"] != SIZE:
        raise ValueError("Official registry manifest no longer contains the pinned layer")
    write_json(stage / "manifest.json", manifest)
    blob_root = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models"))) / "blobs"
    blob = blob_root / ("sha256-" + DIGEST)
    if blob.exists():
        if blob.stat().st_size != SIZE or sha(blob) != DIGEST:
            raise ValueError("Existing destination blob is not the pinned content")
    else:
        def fetch(index):
            start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
            path, receipt = stage / f"{index:04d}.part", stage / f"{index:04d}.json"
            if path.exists() and receipt.exists():
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if saved.get("range") == [start, end] and path.stat().st_size == end - start + 1 and saved.get("sha256") == sha(path):
                    return index, end - start + 1, True
                raise ValueError("Cached range changed; refusing to treat it as downloaded")
            temp = stage / f"{index:04d}.tmp"
            progress = stage / f"{index:04d}.progress.json"
            total, checksum = 0, hashlib.sha256()
            if progress.exists():
                saved = json.loads(progress.read_text(encoding="utf-8"))
                size = saved.get("bytes")
                if (saved.get("range") != [start, end] or saved.get("blob") != DIGEST
                        or type(size) is not int or not 0 < size <= end - start + 1
                        or not temp.exists() or temp.stat().st_size < size):
                    raise ValueError("Cached range prefix identity/length changed")
                with temp.open("r+b") as prefix:
                    remaining = size
                    while remaining:
                        data = prefix.read(min(1024 * 1024, remaining))
                        checksum.update(data)
                        remaining -= len(data)
                    if checksum.hexdigest() != saved.get("sha256"):
                        raise ValueError("Cached range prefix SHA256 changed")
                    # A crash may leave uncheckpointed bytes. Only receipt-bound
                    # bytes are reusable, never a preallocated or inferred prefix.
                    prefix.truncate(size)
                total = size
            else:
                # Legacy temporary files have no authenticated progress receipt.
                # Ignore their apparent length and begin this range from zero.
                temp.write_bytes(b"")

            def checkpoint():
                if total:
                    with temp.open("r+b") as prefix:
                        prefix.flush()
                        os.fsync(prefix.fileno())
                    write_json(progress, {"blob": DIGEST, "range": [start, end],
                                          "bytes": total, "sha256": checksum.hexdigest()})

            for attempt in range(1, 13):
                try:
                    if total == end - start + 1:
                        break
                    requested_start = start + total
                    request = urllib.request.Request(REGISTRY + "/blobs/sha256:" + DIGEST,
                        headers={"Range": f"bytes={requested_start}-{end}", "Accept-Encoding": "identity"})
                    response = (CurlRangeResponse(request, stage, args.r2_ip) if args.transport == "curl"
                                else urllib.request.urlopen(request, timeout=60))
                    with response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {requested_start}-{end}/{SIZE}":
                            raise ValueError("Registry did not honor the exact requested byte range")
                        last_checkpoint = total
                        with temp.open("ab") as output:
                            # read1 retains short network reads; read(n) can lose
                            # its internal partial buffer on a socket timeout.
                            read = getattr(response, "read1", response.read)
                            while data := read(min(1024 * 1024, end - start + 2 - total)):
                                if total + len(data) > end - start + 1:
                                    raise ValueError("Range response exceeds expected length")
                                output.write(data)
                                checksum.update(data)
                                total += len(data)
                                if total - last_checkpoint >= 1024 * 1024:
                                    output.flush()
                                    checkpoint()
                                    last_checkpoint = total
                            output.flush()
                            os.fsync(output.fileno())
                        if total != end - start + 1:
                            raise ValueError("Incomplete range response")
                    break
                except Exception as exc:
                    if isinstance(exc, ValueError) and str(exc) in (
                            "Registry did not honor the exact requested byte range",
                            "Range response exceeds expected length"):
                        # A protocol violation must not become a resumable full
                        # prefix on a subsequent invocation.
                        progress.unlink(missing_ok=True)
                        temp.unlink(missing_ok=True)
                        raise RuntimeError("Range protocol validation failed") from None
                    checkpoint()
                    # urllib exceptions may contain signed CDN URLs. Log only type.
                    print(json.dumps({"stage": "range_retry", "chunk": index, "attempt": attempt,
                                      "error_type": type(exc).__name__, "saved_prefix_bytes": total}), flush=True)
                    if attempt == 12:
                        raise RuntimeError(f"Range {index} failed after twelve attempts ({type(exc).__name__})") from None
                    time.sleep(min(attempt * 2, 8))
            temp.replace(path)
            write_json(receipt, {"range": [start, end], "bytes": total, "sha256": checksum.hexdigest()})
            progress.unlink(missing_ok=True)
            return index, total, False

        completed = 0
        began = time.monotonic()
        count = (SIZE + CHUNK - 1) // CHUNK
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(fetch, i) for i in range(count)]):
                index, size, cached = future.result()
                completed += size
                print(json.dumps({"stage": "range_complete", "chunk": index, "cached": cached,
                                  "completed_bytes": completed, "total": SIZE,
                                  "elapsed_seconds": round(time.monotonic() - began, 1)}), flush=True)
        assembled = stage / "assembled.tmp"
        checksum = hashlib.sha256()
        with assembled.open("wb") as output:
            for index in range(count):
                with (stage / f"{index:04d}.part").open("rb") as source:
                    while data := source.read(1024 * 1024):
                        output.write(data)
                        checksum.update(data)
            output.flush()
            os.fsync(output.fileno())
        if assembled.stat().st_size != SIZE or checksum.hexdigest() != DIGEST:
            raise ValueError("Whole-blob size/SHA256 verification failed; not importing")
        # Re-read after flush before import: the object must match on disk as well.
        if sha(assembled) != DIGEST:
            raise ValueError("On-disk blob SHA256 changed")
        blob_root.mkdir(parents=True, exist_ok=True)
        if blob.exists():
            if sha(blob) != DIGEST:
                raise ValueError("Another writer produced a different destination blob")
            assembled.unlink()
        else:
            os.replace(assembled, blob)
        write_json(stage / "verified.json", {"sha256": DIGEST, "bytes": SIZE, "chunks": count,
                                             "source": REGISTRY, "imported": True})
        print(json.dumps({"stage": "verified_blob_import", "sha256": DIGEST, "bytes": SIZE}), flush=True)
    import core
    with core._create_client() as client:
        last = None
        for event in client.pull("qwen3.5:4b", stream=True):
            row = core._as_dict(event)
            key = (row.get("status"), row.get("digest"))
            if key != last:
                print(json.dumps({k: row.get(k) for k in ("status", "digest", "total", "completed")}), flush=True)
                last = key
        name, model_digest, info = core._inspect_model(client, "qwen3.5:4b")
    summary = {"model": name, "digest": model_digest, "capabilities": info.get("capabilities"), "details": info.get("details")}
    write_json(ROOT / "data" / "qwen35-model.json", summary)
    print(json.dumps({"stage": "model_ready", **summary}), flush=True)


if __name__ == "__main__":
    main()
