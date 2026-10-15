"""CPU CLIP search over sampled video pixels, independent of generated captions.

The input is video_media's fixed-12-fps preview. English text and RGB frames use
the matching Qdrant CLIP ONNX encoders. Scores are cosine similarities, not
probabilities or proof that an event occurred between the sampled frames.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Protocol
import uuid

import numpy as np

from video_media import _atomic_json, _open_capture, _preview_metadata, _read_frame, _sha256


INDEX_VERSION = 1
MAX_FRAMES = 900
MODEL_SPECS = {
    "text": {
        "name": "Qdrant/clip-ViT-B-32-text",
        "revision": "48ca1db27cb4063eb311ec2aa7f087a808112876",
        "model_sha256": "4dbe762b11e36488304471e439cde89da053ad7acaddbf9e096745d142ec8d8b",
        "files": ["model.onnx", "config.json", "merges.txt", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json", "vocab.json"],
    },
    "vision": {
        "name": "Qdrant/clip-ViT-B-32-vision",
        "revision": "e0c24ed0fa57fa3e4f97f30de74c51d944036ace",
        "model_sha256": "c68d3d9a200ddd2a8c8a5510b576d4c94d1ae383bf8b36dd8c084f94e1fb4d63",
        "files": ["model.onnx", "config.json", "preprocessor_config.json"],
    },
}


class VideoIndexError(ValueError):
    """A missing, stale, corrupt, or unsupported visual index."""


class Embedder(Protocol):
    """Injectable encoder pair; identity must describe both encoders and dimension."""

    def identity(self) -> dict: ...
    def embed_images(self, paths: list[str]): ...
    def embed_text(self, text: str): ...


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def model_cache_dir() -> Path:
    """Keep downloaded weights outside the repository, including on Windows."""
    base = Path(os.environ["LOCALAPPDATA"]) if os.environ.get("LOCALAPPDATA") else Path.home() / ".cache"
    return base / "VisionForge" / "models"


class FastEmbedCLIP:
    """Pinned public ONNX weights, CPU only, no PyTorch and no image upload."""

    def __init__(self) -> None:
        cache = model_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        # HF's optional chunk cache is separate from snapshot_download(cache_dir).
        # Set its process-local location before the first Xet download session.
        os.environ["HF_XET_CACHE"] = str(cache / "xet")
        try:
            from fastembed import ImageEmbedding, TextEmbedding
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise VideoIndexError("Visual search dependencies are missing; install requirements-search.txt.") from exc
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()
        for kind, spec in MODEL_SPECS.items():
            arguments = {"repo_id": spec["name"], "revision": spec["revision"],
                         "cache_dir": str(cache / "huggingface"), "allow_patterns": spec["files"],
                         "token": False, "max_workers": 1}
            try:
                directory = Path(snapshot_download(**arguments, local_files_only=True))
                if not all((directory / name).is_file() for name in spec["files"]):
                    raise FileNotFoundError("Incomplete cached model.")
            except (OSError, ValueError):
                try:
                    directory = Path(snapshot_download(**arguments))
                except Exception as exc:
                    raise VideoIndexError(
                        f"Could not download the pinned {kind} CLIP model; visual search is unavailable. "
                        "Retry when Hugging Face is reachable. No substitute embeddings were generated."
                    ) from exc
            self._paths[kind] = directory
        self._initial_identity = self._asset_identity()
        common = {"cache_dir": str(cache / "fastembed"), "threads": 2,
                  "providers": ["CPUExecutionProvider"], "local_files_only": True, "lazy_load": True}
        self._images = ImageEmbedding(MODEL_SPECS["vision"]["name"],
                                      specific_model_path=str(self._paths["vision"]), **common)
        self._texts = TextEmbedding(MODEL_SPECS["text"]["name"],
                                   specific_model_path=str(self._paths["text"]), **common)
        # Check the complete query before FastEmbed's CLIP tokenizer truncates it.
        from tokenizers import Tokenizer
        self._query_tokenizer = Tokenizer.from_file(str(self._paths["text"] / "tokenizer.json"))
        self._query_tokenizer.no_truncation()

    def _asset_identity(self) -> dict:
        assets = {}
        for kind, spec in MODEL_SPECS.items():
            files = []
            for name in spec["files"]:
                path = self._paths[kind] / name
                if not path.is_file():
                    raise VideoIndexError(f"The cached {kind} model is missing {name}; restore its pinned snapshot.")
                files.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
            if files[0]["sha256"] != spec["model_sha256"]:
                raise VideoIndexError(f"The cached {kind} ONNX weight checksum is wrong; refusing incompatible embeddings.")
            assets[kind] = {"name": spec["name"], "revision": spec["revision"], "files": files}
        return {"implementation": "FastEmbed CLIP ONNX CPU", "dimension": 512,
                "normalization": "explicit_float32_L2", "threads": 2,
                "providers": ["CPUExecutionProvider"], "models": assets,
                "packages": {name: version(name) for name in
                             ("fastembed", "onnxruntime", "tokenizers", "numpy", "Pillow")}}

    def identity(self) -> dict:
        current = self._asset_identity()
        if current != self._initial_identity:
            raise VideoIndexError("CLIP model files changed after loading; restart the application before rebuilding.")
        return current

    def embed_images(self, paths: list[str]):
        with self._lock:
            return list(self._images.embed(paths, batch_size=4, parallel=None))

    def embed_text(self, text: str):
        with self._lock:
            if len(self._query_tokenizer.encode(text).ids) > 77:
                raise VideoIndexError("The English query exceeds CLIP's 77-token limit; shorten it before searching.")
            return next(self._texts.embed([text], batch_size=1, parallel=None))


@lru_cache(maxsize=1)
def _default_embedder() -> FastEmbedCLIP:
    return FastEmbedCLIP()


def release_models() -> None:
    """Release default CLIP sessions after retrieval, before loading a VLM.

    Call only when no build/search operation is active, under the caller's
    shared inference lock. Injected embedders remain owned by their caller.
    """
    _default_embedder.cache_clear()
    gc.collect()


def _sampling(frame_count: int, fps: float, interval_seconds: float, max_frames: int) -> tuple[list[int], dict]:
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) or not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise VideoIndexError("interval_seconds must be a finite number greater than zero.")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 2 <= max_frames <= MAX_FRAMES:
        raise VideoIndexError("max_frames must be an integer between 2 and 900.")
    step = max(1, round(min(float(interval_seconds), frame_count / fps) * fps))
    indices = list(range(0, frame_count, step))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    adjusted = len(indices) > max_frames
    if adjusted:
        indices = np.rint(np.linspace(0, frame_count - 1, max_frames)).astype(int).tolist()
    gaps = [(b - a) / fps for a, b in zip(indices, indices[1:])]
    return indices, {"strategy": "uniform_full_timeline_with_first_and_last", "requested_interval_seconds": float(interval_seconds),
                     "max_frames": max_frames, "frame_count": len(indices), "budget_adjusted": adjusted,
                     "effective_interval_seconds": (indices[-1] - indices[0]) / fps / max(1, len(indices) - 1),
                     "maximum_gap_seconds": max(gaps, default=0.0), "minimum_gap_seconds": min(gaps, default=0.0),
                     "first_time_seconds": indices[0] / fps, "last_time_seconds": indices[-1] / fps,
                     "coverage_note": "Sparse samples across the whole preview, not continuous event detection."}


def _unit_vectors(values, rows: int, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (rows, dimension) or not np.isfinite(array).all():
        raise VideoIndexError("The embedder returned invalid vector dimensions or non-finite values.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise VideoIndexError("The embedder returned a zero or invalid vector.")
    return array / norms


def _inside(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise VideoIndexError("Index files must have relative paths within the index directory.")
    result = (directory / relative).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise VideoIndexError("An index path escapes its directory.")
    return result


def _load_index(directory: Path, expected_identity: dict | None = None) -> tuple[dict, np.ndarray]:
    """Verify metadata, preview, every evidence image, and vector row alignment."""
    try:
        saved = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        identity = saved["identity"]
        if identity["index_version"] != INDEX_VERSION or saved["identity_sha256"] != _json_hash(identity):
            raise VideoIndexError("The visual index identity is invalid or obsolete; rebuild it.")
        if expected_identity is not None and identity != expected_identity:
            raise VideoIndexError("The video, sampling settings, or embedder changed; rebuild the visual index.")
        source = Path(saved["preview_path"])
        if not source.is_file() or _sha256(source) != identity["preview_sha256"]:
            raise VideoIndexError("The indexed preview is missing or changed; rebuild the visual index.")
        vectors_path = _inside(directory, saved["vectors_file"])
        if _sha256(vectors_path) != saved["vectors_sha256"]:
            raise VideoIndexError("The visual index vectors are corrupted; rebuild the index.")
        frames = saved["frames"]
        if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
            raise VideoIndexError("The index contains an invalid number of evidence frames.")
        expected_indices, expected_sampling = _sampling(identity["video"]["frame_count"], identity["video"]["fps"],
                                                        identity["interval_seconds"], identity["max_frames"])
        if saved["sampling"] != expected_sampling or [f["frame_index"] for f in frames] != expected_indices:
            raise VideoIndexError("The index sample positions do not match its sampling identity.")
        for ordinal, frame in enumerate(frames):
            if frame["id"] != f"I{ordinal + 1:06d}" or frame["time_seconds"] != round(frame["frame_index"] / identity["video"]["fps"], 6):
                raise VideoIndexError("An evidence timestamp or identifier is invalid; rebuild the index.")
            if _sha256(_inside(directory, frame["path"])) != frame["sha256"]:
                raise VideoIndexError("An indexed evidence image changed; rebuild the index.")
        with np.load(vectors_path, allow_pickle=False) as data:
            vectors = data["vectors"].copy()
            if str(data["identity_sha256"].item()) != saved["identity_sha256"] or not np.array_equal(data["frame_indices"], expected_indices):
                raise VideoIndexError("The vector rows belong to a different visual index.")
            if not np.array_equal(data["image_sha256"], [f["sha256"] for f in frames]):
                raise VideoIndexError("The vector rows do not match the evidence image checksums.")
        if vectors.dtype != np.float32 or vectors.shape != (len(frames), identity["embedder"]["dimension"]) or not np.isfinite(vectors).all():
            raise VideoIndexError("The cached vector matrix is invalid.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise VideoIndexError("The cached vectors are not normalized.")
        return saved, vectors
    except VideoIndexError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        raise VideoIndexError("The visual index is missing or damaged; build it again.") from exc


def build_index(preview_path: str | Path, output_dir: str | Path, interval_seconds: float = 2.0,
                max_frames: int = 900, progress_callback: Callable[[str, str], None] | None = None,
                *, embedder: Embedder | None = None) -> dict:
    """Build or verify a visual index; return a JSON-serializable manifest.

    Optional injected embedders implement identity(), embed_images(paths), and
    embed_text(text). A cache hit validates video/model/image identities and the
    NPZ checksum without recomputing embeddings. Callbacks receive (stage, text).
    """
    started = time.perf_counter()
    _sampling(1, 12, interval_seconds, max_frames)  # Validate before loading models.
    source = Path(preview_path).expanduser().resolve()
    directory = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise VideoIndexError("The normalized video preview does not exist.")
    directory.mkdir(parents=True, exist_ok=True)

    def progress(stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(stage, message)

    from filelock import FileLock, Timeout
    try:
        with FileLock(str(directory / ".index.lock"), timeout=0):
            progress("probe", "Checking the preview and planning samples across its complete timeline.")
            metadata = _preview_metadata(source)
            indices, sampling = _sampling(metadata["frame_count"], metadata["fps"], interval_seconds, max_frames)
            source_hash = _sha256(source)
            progress("model", "Loading the pinned CPU CLIP encoders; the first use downloads public weights.")
            engine = embedder if embedder is not None else _default_embedder()
            model_identity = engine.identity()
            dimension = model_identity.get("dimension")
            if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 4096:
                raise VideoIndexError("Embedder identity must declare a valid dimension.")
            identity = {"index_version": INDEX_VERSION, "code_sha256": _sha256(Path(__file__)),
                        "preview_sha256": source_hash, "video": metadata, "interval_seconds": float(interval_seconds),
                        "max_frames": max_frames, "embedder": model_identity,
                        "decoder": {"opencv": version("opencv-python-headless"), "image_format": "lossless_PNG"}}
            try:
                saved, _ = _load_index(directory, identity)
                progress("complete", "Reused the visual index after checking video, model, image and vector checksums.")
                return {**saved, "cache_hit": True, "elapsed_seconds": round(time.perf_counter() - started, 6)}
            except VideoIndexError:
                pass
            generation = directory / "generations" / uuid.uuid4().hex
            generation.mkdir(parents=True)
            frames = []
            rows = []
            import cv2
            capture = _open_capture(source)
            embedding_seconds = 0.0
            try:
                for ordinal, requested in enumerate(indices, 1):
                    actual, pixels = _read_frame(capture, requested)
                    frame_id = f"I{ordinal:06d}"
                    path = generation / f"{frame_id}.png"
                    if not cv2.imwrite(str(path), pixels):
                        raise VideoIndexError("Could not save a visual search evidence frame.")
                    frames.append({"id": frame_id, "path": path.relative_to(directory).as_posix(),
                                   "frame_index": actual, "time_seconds": round(actual / metadata["fps"], 6),
                                   "sha256": _sha256(path)})
                    if ordinal % 4 == 0 or ordinal == len(indices):
                        pending = frames[len(rows):]
                        progress("embed", f"Encoding raw frames {len(rows) + 1}-{ordinal} of {len(indices)} on CPU.")
                        batch_started = time.perf_counter()
                        batch = list(engine.embed_images([str(_inside(directory, f["path"])) for f in pending]))
                        normalized = _unit_vectors(batch, len(pending), dimension)
                        embedding_seconds += time.perf_counter() - batch_started
                        rows.extend(normalized)
            finally:
                capture.release()
            if _sha256(source) != source_hash or engine.identity() != model_identity:
                raise VideoIndexError("The preview or embedder changed while indexing; retry with stable inputs.")
            vectors_path = generation / "vectors.npz"
            with vectors_path.open("wb") as stream:
                np.savez_compressed(stream, vectors=np.asarray(rows, dtype=np.float32),
                                    frame_indices=np.asarray(indices, dtype=np.int64),
                                    image_sha256=np.asarray([f["sha256"] for f in frames]),
                                    identity_sha256=np.asarray(_json_hash(identity)))
            saved = {"identity": identity, "identity_sha256": _json_hash(identity), "preview_path": str(source),
                     "sampling": sampling, "frames": frames,
                     "vectors_file": vectors_path.relative_to(directory).as_posix(), "vectors_sha256": _sha256(vectors_path),
                     "created_at": datetime.now(timezone.utc).isoformat(),
                     "embedding_seconds": round(embedding_seconds, 6),
                     "build_seconds": round(time.perf_counter() - started, 6),
                     "query_language": "English; translate Chinese in the caller and retain the actual English query.",
                     "score_semantics": "Cosine similarity for ranking sampled pixels; not a calibrated probability."}
            _atomic_json(directory / "index.json", saved)
            progress("complete", f"Indexed {len(frames)} raw video frames; maximum sampling gap is {sampling['maximum_gap_seconds']:.3f}s.")
            return {**saved, "cache_hit": False, "elapsed_seconds": round(time.perf_counter() - started, 6)}
    except Timeout as exc:
        raise VideoIndexError("This visual index is busy; wait for its current operation to finish.") from exc


def search_index(index_dir: str | Path, query_text: str, top_k: int = 3,
                 min_separation_seconds: float = 4.0, *, embedder: Embedder | None = None) -> list[dict]:
    """Rank sampled pixels by cosine similarity, then greedily deduplicate time.

    Ties are resolved by earlier frame index. Separation is inclusive: results
    exactly min_separation_seconds apart are allowed. May return fewer than k.
    """
    if not isinstance(query_text, str) or not query_text.strip() or len(query_text) > 512:
        raise VideoIndexError("Enter a nonempty English visual query of at most 512 characters.")
    if re.search(r"[\u3400-\u9fff\uf900-\ufaff]", query_text):
        raise VideoIndexError("CLIP text search requires English; translate the Chinese query before searching.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_FRAMES:
        raise VideoIndexError("top_k must be an integer between 1 and 900.")
    if isinstance(min_separation_seconds, bool) or not isinstance(min_separation_seconds, (int, float)) or not math.isfinite(min_separation_seconds) or min_separation_seconds < 0:
        raise VideoIndexError("min_separation_seconds must be finite and nonnegative.")
    directory = Path(index_dir).expanduser().resolve()
    if not directory.is_dir():
        raise VideoIndexError("The visual index does not exist; build it first.")
    from filelock import FileLock, Timeout
    try:
        with FileLock(str(directory / ".index.lock"), timeout=0):
            saved, vectors = _load_index(directory)
            engine = embedder if embedder is not None else _default_embedder()
            current_identity = engine.identity()
            if current_identity != saved["identity"]["embedder"] or saved["identity"]["code_sha256"] != _sha256(Path(__file__)):
                raise VideoIndexError("The embedder or index implementation changed; rebuild the visual index.")
            query = _unit_vectors([engine.embed_text(query_text.strip())], 1, vectors.shape[1])[0]
            scores = np.clip(vectors @ query, -1.0, 1.0)
            order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), saved["frames"][i]["frame_index"]))
            selected = []
            for i in order:
                frame = saved["frames"][i]
                if any(abs(frame["time_seconds"] - prior["time_seconds"]) < min_separation_seconds - 1e-6 for prior in selected):
                    continue
                selected.append({**frame, "path": str(_inside(directory, frame["path"])), "score": float(scores[i])})
                if len(selected) == top_k:
                    break
            return selected
    except Timeout as exc:
        raise VideoIndexError("This visual index is busy; wait for its current operation to finish.") from exc
