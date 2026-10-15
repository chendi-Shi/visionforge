"""Small offline range fixtures; never downloads a model or touches real cache."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import core
from tools import download_qwen35_blob as download


class Response(io.BytesIO):
    def __init__(self, content, *, status=200, headers=None):
        super().__init__(content)
        self.status, self.headers = status, headers or {}


class InterruptedResponse(Response):
    def read1(self, size=-1):
        if self.tell():
            raise TimeoutError("simulated interrupted range")
        return super().read1(min(size, 3))


class BlobDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="blob-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b"0123456789abcdefghijklmn"
        self.checksum = hashlib.sha256(self.content).hexdigest()
        self.ranges = []
        self.bad_range = False
        self.bad_body = False
        self.interrupt_once = False
        self.extra_body = False
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.client.pull.return_value = [{"status": "success"}]
        patches = [patch.object(download, "ROOT", self.root), patch.object(download, "DIGEST", self.checksum),
                   patch.object(download, "SIZE", len(self.content)), patch.object(download, "CHUNK", 8),
                   patch.dict(os.environ, {"OLLAMA_MODELS": str(self.root / "models")}),
                   patch("sys.argv", ["download", "--workers", "2"]), patch.object(download.time, "sleep"),
                   patch.object(download.urllib.request, "urlopen", side_effect=self.open),
                   patch.object(core, "_create_client", return_value=self.client),
                   patch.object(core, "_inspect_model", return_value=("qwen3.5:4b", "model-digest", {"capabilities": ["vision"]}))]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def open(self, request, timeout):
        if isinstance(request, str):
            return Response(json.dumps({"layers": [{"digest": "sha256:" + self.checksum, "size": len(self.content)}]}).encode())
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        self.ranges.append((start, end))
        payload = self.content[start:end + 1]
        if self.bad_body:
            payload = b"X" * len(payload)
        if self.extra_body:
            payload += b"X"
        cls = Response
        if self.interrupt_once and start == 0:
            self.interrupt_once = False
            cls = InterruptedResponse
        return cls(payload, status=206, headers={"Content-Range": f"bytes {start}-{end}/{len(self.content) + int(self.bad_range)}"})

    def run_silently(self):
        with redirect_stdout(io.StringIO()):
            download.main()

    def test_exact_ranges_whole_hash_and_manifest_finalize(self):
        self.run_silently()
        target = self.root / "models" / "blobs" / ("sha256-" + self.checksum)
        self.assertEqual(target.read_bytes(), self.content)
        self.assertEqual(sorted(self.ranges), [(0, 7), (8, 15), (16, 23)])
        self.client.pull.assert_called_once_with("qwen3.5:4b", stream=True)
        # Existing verified destination is recognized without any range fetch.
        self.ranges.clear()
        self.run_silently()
        self.assertEqual(self.ranges, [])

    def test_wrong_content_range_never_imports_or_finalizes(self):
        self.bad_range = True
        with self.assertRaises(RuntimeError):
            self.run_silently()
        self.assertFalse((self.root / "models" / "blobs").exists())
        self.client.pull.assert_not_called()

    def test_equal_length_corrupted_body_fails_whole_sha(self):
        self.bad_body = True
        with self.assertRaisesRegex(ValueError, "Whole-blob"):
            self.run_silently()
        self.assertFalse((self.root / "models" / "blobs").exists())
        self.client.pull.assert_not_called()

    def test_manifest_copy_preserves_original_bytes_and_checks_registered_digest(self):
        source = self.root / "models/manifests/registry.ollama.ai/library/qwen3.5/4b"
        source.parent.mkdir(parents=True)
        raw = ('{  "layers": [{"digest":"sha256:' + self.checksum + '", "size":24}] }\n').encode()
        source.write_bytes(raw)
        full_digest = hashlib.sha256(raw).hexdigest()
        with patch.object(core, "_inspect_model", return_value=("qwen3.5:4b", full_digest, {})):
            result = download.save_model_manifest()
        self.assertEqual((self.root / "data/qwen35-model-manifest.json").read_bytes(), raw)
        self.assertEqual(result["manifest_sha256"], full_digest)
        with self.assertRaisesRegex(ValueError, "exact manifest bytes"):
            download.save_model_manifest()

    def wrapped_manifest(self):
        original = {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                    "runner": "llamacpp", "format": "gguf", "config": {"digest": "sha256:" + "c" * 64},
                    "layers": [{"mediaType": "application/vnd.ollama.image.model",
                                "digest": "sha256:" + self.checksum, "size": 24}]}
        # Deliberate whitespace makes reconstruction/re-serialization detectable.
        raw = (json.dumps(original, indent=3) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        blobs = self.root / "models/blobs"
        blobs.mkdir(parents=True)
        (blobs / ("sha256-" + digest)).write_bytes(raw)
        list_raw = json.dumps({"manifests": [{"digest": "sha256:" + digest,
                                             "runner": "llamacpp", "format": "gguf"}]}).encode()
        list_digest = hashlib.sha256(list_raw).hexdigest()
        (blobs / ("sha256-" + list_digest)).write_bytes(list_raw)
        wrapper = dict(original)
        wrapper["layers"] = original["layers"] + [
            {"mediaType": "application/vnd.ollama.manifest.list.v2+json", "digest": "sha256:" + list_digest, "size": len(list_raw)},
            {"mediaType": original["mediaType"], "digest": "sha256:" + digest, "size": len(raw)}]
        path = self.root / "models/manifests/registry.ollama.ai/library/qwen3.5/4b"
        path.parent.mkdir(parents=True)
        path.write_bytes(json.dumps(wrapper).encode())
        return raw, digest, path, wrapper

    def test_wrapped_manifest_copies_original_blob_bytes_and_preserves_local_wrapper(self):
        raw, digest, source, _ = self.wrapped_manifest()
        local_raw = source.read_bytes()
        with patch.object(core, "_inspect_model", return_value=("qwen3.5:4b", digest, {})):
            result = download.save_model_manifest()
        self.assertEqual((self.root / "data/qwen35-model-manifest.json").read_bytes(), raw)
        self.assertEqual((self.root / "data/qwen35-model-manifest.local.json").read_bytes(), local_raw)
        self.assertEqual(result["manifest_origin"], "local_retained_registry_blob")
        self.assertEqual(result["manifest_sha256"], digest)
        self.assertNotEqual(result["local_wrapper_sha256"], digest)

    def test_wrapper_runtime_changes_cannot_hide_behind_valid_registry_digest(self):
        _, digest, source, wrapper = self.wrapped_manifest()
        wrapper["config"] = {"digest": "sha256:" + "d" * 64}
        source.write_bytes(json.dumps(wrapper).encode())
        with patch.object(core, "_inspect_model", return_value=("qwen3.5:4b", digest, {})):
            with self.assertRaisesRegex(ValueError, "runtime configuration"):
                download.save_model_manifest()
        self.assertFalse((self.root / "data/qwen35-model-manifest.json").exists())

    def test_retained_registry_blob_must_match_its_digest_before_copy(self):
        raw, digest, _, _ = self.wrapped_manifest()
        (self.root / "models/blobs" / ("sha256-" + digest)).write_bytes(raw.replace(b"gguf", b"GGUF"))
        with patch.object(core, "_inspect_model", return_value=("qwen3.5:4b", digest, {})):
            with self.assertRaisesRegex(ValueError, "manifest size/SHA256"):
                download.save_model_manifest()
        self.assertFalse((self.root / "data/qwen35-model-manifest.json").exists())

    def test_manifest_list_must_select_the_registered_runtime(self):
        _, digest, source, wrapper = self.wrapped_manifest()
        wrong_list = json.dumps({"manifests": [{"digest": "sha256:" + digest,
                                               "runner": "mlx", "format": "safetensors"}]}).encode()
        wrong_digest = hashlib.sha256(wrong_list).hexdigest()
        (self.root / "models/blobs" / ("sha256-" + wrong_digest)).write_bytes(wrong_list)
        wrapper["layers"][-2].update(digest="sha256:" + wrong_digest, size=len(wrong_list))
        source.write_bytes(json.dumps(wrapper).encode())
        with patch.object(core, "_inspect_model", return_value=("qwen3.5:4b", digest, {})):
            with self.assertRaisesRegex(ValueError, "does not select registered runtime"):
                download.save_model_manifest()

    def test_timeout_resumes_verified_prefix_inside_original_range(self):
        self.interrupt_once = True
        self.run_silently()
        self.assertIn((0, 7), self.ranges)
        self.assertIn((3, 7), self.ranges)
        target = self.root / "models" / "blobs" / ("sha256-" + self.checksum)
        self.assertEqual(target.read_bytes(), self.content)

    def test_restart_uses_only_receipt_bound_prefix_and_discards_tail(self):
        stage = self.root / "data/model-download" / self.checksum[:12]
        stage.mkdir(parents=True)
        (stage / "0000.tmp").write_bytes(self.content[:3] + b"uncheckpointed-tail")
        (stage / "0000.progress.json").write_text(json.dumps({
            "blob": self.checksum, "range": [0, 7], "bytes": 3,
            "sha256": hashlib.sha256(self.content[:3]).hexdigest()}))
        self.run_silently()
        self.assertNotIn((0, 7), self.ranges)
        self.assertIn((3, 7), self.ranges)
        self.assertFalse((stage / "0000.progress.json").exists())

    def test_unreceipted_preallocation_is_never_a_downloaded_prefix(self):
        stage = self.root / "data/model-download" / self.checksum[:12]
        stage.mkdir(parents=True)
        (stage / "0000.tmp").write_bytes(b"X" * 8)
        self.run_silently()
        self.assertIn((0, 7), self.ranges)

    def test_changed_prefix_rejected_before_import(self):
        stage = self.root / "data/model-download" / self.checksum[:12]
        stage.mkdir(parents=True)
        (stage / "0000.tmp").write_bytes(b"XXX")
        (stage / "0000.progress.json").write_text(json.dumps({
            "blob": self.checksum, "range": [0, 7], "bytes": 3,
            "sha256": hashlib.sha256(self.content[:3]).hexdigest()}))
        with self.assertRaisesRegex(ValueError, "prefix SHA256"):
            self.run_silently()
        self.client.pull.assert_not_called()

    def test_overlong_body_cannot_become_a_completed_prefix(self):
        self.extra_body = True
        with self.assertRaisesRegex(RuntimeError, "protocol validation"):
            self.run_silently()
        stage = self.root / "data/model-download" / self.checksum[:12]
        self.assertEqual(list(stage.glob("*.progress.json")), [])
        self.assertEqual(list(stage.glob("*.part")), [])
        self.client.pull.assert_not_called()

    def test_curl_transport_validates_last_headers_and_resumes_timeout(self):
        ranges = []
        def curl(args, **kwargs):
            a, b = map(int, args[args.index("--range") + 1].split("-"))
            ranges.append((a, b))
            header = Path(args[args.index("--dump-header") + 1])
            header.write_bytes((f"HTTP/1.1 307 Redirect\r\nLocation: https://example.invalid/signed\r\n\r\n"
                                f"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{b}/24\r\n\r\n").encode())
            interrupted = a == 0
            Path(args[args.index("--output") + 1]).write_bytes(self.content[a:a + 3] if interrupted else self.content[a:b + 1])
            return SimpleNamespace(returncode=28 if interrupted else 0, stdout=b"206", stderr=b"untrusted transport details")
        with patch("sys.argv", ["download", "--workers", "16", "--transport", "curl"]), patch.object(download.subprocess, "run", side_effect=curl):
            self.run_silently()
        self.assertIn((3, 7), ranges)
        self.assertEqual((self.root / "models/blobs" / ("sha256-" + self.checksum)).read_bytes(), self.content)
        self.assertEqual(list((self.root / "data/model-download" / self.checksum[:12]).glob(".cr-*")), [])

    def test_curl_wrong_final_range_not_hidden_by_valid_redirect_header(self):
        def curl(args, **kwargs):
            a, b = map(int, args[args.index("--range") + 1].split("-"))
            Path(args[args.index("--dump-header") + 1]).write_text(
                f"HTTP/1.1 307 Redirect\nContent-Range: bytes {a}-{b}/24\n\n"
                "HTTP/1.1 206 Partial Content\nContent-Range: bytes 0-7/999\n\n")
            Path(args[args.index("--output") + 1]).write_bytes(self.content[a:b + 1])
            return SimpleNamespace(returncode=0, stdout=b"206", stderr=b"")
        with patch("sys.argv", ["download", "--workers", "2", "--transport", "curl"]), patch.object(download.subprocess, "run", side_effect=curl):
            with self.assertRaisesRegex(RuntimeError, "protocol validation"):
                self.run_silently()
        self.client.pull.assert_not_called()

    def test_curl_selected_address_still_uses_official_tls_hostname(self):
        def curl(args, **kwargs):
            self.assertEqual(args[args.index("--resolve") + 1], download.R2_HOST + ":443:198.51.100.1")
            self.assertNotIn("--insecure", args)
            a, b = map(int, args[args.index("--range") + 1].split("-"))
            Path(args[args.index("--dump-header") + 1]).write_text(
                f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {a}-{b}/24\n\n")
            Path(args[args.index("--output") + 1]).write_bytes(self.content[a:b + 1])
            return SimpleNamespace(returncode=0, stdout=b"206", stderr=b"")
        addresses = [(2, 1, 6, "", ("198.51.100.1", 443))]
        argv = ["download", "--transport", "curl", "--r2-ip", "198.51.100.1"]
        with patch("sys.argv", argv), patch.object(download.socket, "getaddrinfo", return_value=addresses), patch.object(download.subprocess, "run", side_effect=curl):
            self.run_silently()
        self.client.pull.assert_called_once()

    def test_address_not_advertised_by_dns_rejected_before_transfer(self):
        argv = ["download", "--transport", "curl", "--r2-ip", "198.51.100.1"]
        with patch("sys.argv", argv), patch.object(download.socket, "getaddrinfo", return_value=[]):
            with self.assertRaisesRegex(ValueError, "official DNS"):
                self.run_silently()
        self.assertEqual(self.ranges, [])
        self.client.pull.assert_not_called()


if __name__ == "__main__":
    unittest.main()
