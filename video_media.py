"""Local, bounded video preparation built on FFmpeg and PySceneDetect.

Evidence times refer to decoded frames in the normalized preview, whose timeline
starts at zero. Scene detection is a sampling aid, not semantic video inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Callable
import uuid


MAX_SOURCE_BYTES = 1024 ** 3
MAX_DURATION_SECONDS = 30 * 60
NORMALIZED_FPS = 12
MAX_WIDTH = 960
WINDOW_SECONDS = 15
PIPELINE_VERSION = 1
_NORMALIZATION = {
    "version": PIPELINE_VERSION, "fps": NORMALIZED_FPS, "max_width": MAX_WIDTH,
    "video_codec": "libx264", "crf": 25, "audio": "removed", "timeline": "zero_based",
}


class VideoPreparationError(ValueError):
    """An input or media failure suitable for displaying in the job UI."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ffmpeg_run(arguments: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            arguments, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoPreparationError("Video processing timed out; try a shorter or smaller video.") from exc
    except OSError as exc:
        raise VideoPreparationError("Could not start the bundled FFmpeg executable.") from exc


def _parse_source_probe(stderr: str) -> dict:
    duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not duration or not re.search(r"Stream\s+.*Video:", stderr):
        raise VideoPreparationError("Cannot read a finite video duration; the file may be unsupported, incomplete, or audio-only.")
    hours, minutes, seconds = map(float, duration.groups())
    seconds += 3600 * hours + 60 * minutes
    if not 0 < seconds <= MAX_DURATION_SECONDS:
        raise VideoPreparationError("Video must be longer than zero and no longer than 30 minutes.")
    start = re.search(r"start:\s*(-?\d+(?:\.\d+)?)", stderr)
    return {"source_duration_seconds": seconds, "source_start_seconds": float(start.group(1)) if start else 0.0}


def _probe_source(executable: str, path: Path) -> dict:
    # FFmpeg intentionally returns nonzero for a probe with no output file.
    result = _ffmpeg_run([executable, "-nostdin", "-hide_banner", "-threads", "2", "-i", str(path)], 30)
    return _parse_source_probe(result.stderr)


def _open_capture(path: Path):
    import cv2
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not capture.isOpened():
        capture.release()
        raise VideoPreparationError("The normalized preview could not be opened for decoding.")
    return capture


def _read_frame(capture, requested: int):
    import cv2
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, requested):
        raise VideoPreparationError("The preview decoder cannot seek to an evidence frame.")
    success, frame = capture.read()
    position = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if not success or frame is None or not math.isfinite(position) or position < 1:
        raise VideoPreparationError("Could not decode an evidence frame; the preview may be damaged.")
    actual = round(position) - 1
    if abs(position - round(position)) > 0.01 or actual != requested:
        raise VideoPreparationError("The decoder returned an unexpected frame position; refusing an inaccurate evidence timestamp.")
    return actual, frame


def _preview_metadata(path: Path) -> dict:
    import cv2
    capture = _open_capture(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or abs(fps - NORMALIZED_FPS) > 0.001:
            raise VideoPreparationError("The preview does not have the required fixed 12 fps timebase.")
        if not math.isfinite(raw_count) or raw_count < 1 or width < 2 or height < 2 or width > MAX_WIDTH:
            raise VideoPreparationError("The preview has invalid dimensions or contains no video frames.")
        count = round(raw_count)
        if count / fps > MAX_DURATION_SECONDS + 1 / fps:
            raise VideoPreparationError("The decoded preview exceeds the 30 minute limit.")
        _read_frame(capture, 0)
        _read_frame(capture, count - 1)
        return {"width": width, "height": height, "fps": fps, "frame_count": count,
                "duration_seconds": count / fps, "preview_duration_seconds": count / fps}
    finally:
        capture.release()


def _normalize(executable: str, source: Path, preview: Path, duration: float) -> dict:
    temporary = preview.with_name(f".preview.{uuid.uuid4().hex}.mp4")
    # Keep the aspect ratio, avoid enlarging, and produce even H.264 dimensions.
    filters = "setpts=PTS-STARTPTS,scale=w='min(960,trunc(iw/2)*2)':h=-2,setsar=1,fps=12:start_time=0"
    command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-threads", "2", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", filters, "-c:v", "libx264", "-threads", "2", "-filter_threads", "2",
               "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
               "-movflags", "+faststart", str(temporary)]
    try:
        result = _ffmpeg_run(command, max(120, min(3600, duration * 6)))
        if result.returncode != 0 or not temporary.is_file():
            detail = result.stderr.strip().splitlines()[-1:] or ["No playable output was produced."]
            raise VideoPreparationError(f"FFmpeg could not normalize this video: {detail[0][:350]}")
        metadata = _preview_metadata(temporary)
        os.replace(temporary, preview)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def _cached_preview(preview: Path, manifest: Path, source_hash: str) -> dict | None:
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("source_sha256") != source_hash or saved.get("normalization") != _NORMALIZATION:
            return None
        if not preview.is_file() or _sha256(preview) != saved.get("preview_sha256"):
            return None
        return _preview_metadata(preview)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _detect_scenes(preview: Path, frame_count: int) -> list[tuple[int, int]]:
    from scenedetect import SceneManager
    from scenedetect.backends.opencv import VideoCaptureAdapter
    from scenedetect.detectors import AdaptiveDetector
    capture = _open_capture(preview)
    try:
        video = VideoCaptureAdapter(capture, frame_rate=NORMALIZED_FPS, max_read_attempts=0)
        manager = SceneManager()
        manager.auto_downscale = True
        manager.add_detector(AdaptiveDetector(min_scene_len=NORMALIZED_FPS))
        processed = manager.detect_scenes(video=video, show_progress=False)
        if processed != frame_count:
            raise VideoPreparationError("The preview could not be decoded completely; refusing an incomplete scene timeline.")
        return [(start.get_frames(), end.get_frames()) for start, end in manager.get_scene_list(start_in_scene=True)]
    finally:
        capture.release()


def _candidate_windows(scenes: list[tuple[int, int]], frame_count: int,
                       fps: int = NORMALIZED_FPS) -> list[tuple[int, int]]:
    """Cover every normalized frame, splitting long shots into bounded windows."""
    boundaries = {0, frame_count}
    for start, end in scenes:
        boundaries.update((max(0, min(frame_count, start)), max(0, min(frame_count, end))))
    ordered = sorted(boundaries)
    maximum = WINDOW_SECONDS * fps
    return [(start, min(start + maximum, end))
            for first, end in zip(ordered, ordered[1:])
            for start in range(first, end, maximum)]


def _select_windows(windows: list[tuple[int, int]], budget: int) -> list[int]:
    """Keep endpoints and sample the interior by elapsed time, not cut density."""
    if len(windows) <= budget:
        return list(range(len(windows)))
    selected = {0, len(windows) - 1}
    duration = windows[-1][1] - windows[0][0]
    for ordinal in range(1, budget - 1):
        target = windows[0][0] + duration * ordinal / (budget - 1)
        selected.add(min((index for index in range(len(windows)) if index not in selected),
                         key=lambda index: (abs((windows[index][0] + windows[index][1]) / 2 - target), index)))
    return sorted(selected)


def _evidence_indices(start: int, end: int) -> list[int]:
    return sorted({round(start + (end - start - 1) * fraction) for fraction in (0.2, 0.8)})


def _extract_frames(preview: Path, folder: Path, selected: list[tuple[int, int]]) -> list[dict]:
    import cv2
    folder.mkdir(parents=True, exist_ok=True)
    capture = _open_capture(preview)
    segments = []
    try:
        for ordinal, (start, end) in enumerate(selected, 1):
            segment_id = f"S{ordinal:04d}"
            frames = []
            for frame_ordinal, requested in enumerate(_evidence_indices(start, end), 1):
                actual, pixels = _read_frame(capture, requested)
                frame_id = f"{segment_id}-F{frame_ordinal:02d}"
                path = folder / f"{frame_id}.png"
                temporary = folder / f".{frame_id}.{uuid.uuid4().hex}.png"
                try:
                    if not cv2.imwrite(str(temporary), pixels):
                        raise VideoPreparationError("Could not save an evidence screenshot.")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                frames.append({"id": frame_id, "time_seconds": round(actual / NORMALIZED_FPS, 6),
                               "frame_index": actual, "path": str(path), "sha256": _sha256(path)})
            segments.append({"id": segment_id, "start_seconds": round(start / NORMALIZED_FPS, 6),
                             "end_seconds": round(end / NORMALIZED_FPS, 6), "frames": frames})
    finally:
        capture.release()
    return segments


def prepare_video(source_path: str | Path, output_dir: str | Path, max_segments: int = 4,
                  frames_per_segment: int = 2,
                  progress_callback: Callable[[str, str], None] | None = None) -> dict:
    """Prepare a local source and write a reproducible, evidence-linked plan.json.

    Re-running verifies a source-keyed preview checksum and decoding before reuse.
    Scene cuts and evidence are regenerated deterministically from that preview.
    A one-frame window produces one screenshot instead of duplicate evidence.
    """
    if isinstance(max_segments, bool) or not isinstance(max_segments, int) or not 2 <= max_segments <= 12:
        raise VideoPreparationError("max_segments must be an integer between 2 and 12.")
    if isinstance(frames_per_segment, bool) or not isinstance(frames_per_segment, int) or frames_per_segment != 2:
        raise VideoPreparationError("This pipeline requires exactly two evidence frames per segment.")
    source = Path(source_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise VideoPreparationError("Video source does not exist or is not a regular file.")
    source_stat = source.stat()
    if not 0 < source_stat.st_size <= MAX_SOURCE_BYTES:
        raise VideoPreparationError("Video source must be nonempty and no larger than 1 GiB.")
    preview = output / "preview.mp4"
    if source == preview:
        raise VideoPreparationError("The source cannot also be the generated preview path.")

    def progress(stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(stage, message)

    try:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise VideoPreparationError("Video dependencies are unavailable; install requirements-video.txt.") from exc
    progress("probe", "Validating the local video and computing its SHA256.")
    probe = _probe_source(executable, source)
    source_hash = _sha256(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "preview-info.json"
    metadata = _cached_preview(preview, manifest, source_hash)
    if metadata is None:
        progress("normalize", "Creating a 12 fps H.264 preview, up to 960 pixels wide; audio is removed.")
        metadata = _normalize(executable, source, preview, probe["source_duration_seconds"])
    else:
        progress("normalize", "Reusing the preview after checking its checksum and decoded endpoints.")
    current_stat = source.stat()
    if (current_stat.st_size, current_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise VideoPreparationError("The source video changed during preparation; retry with a stable local file.")
    preview_hash = _sha256(preview)
    progress("scenes", "Detecting visual cuts and splitting long scenes into windows of at most 15 seconds.")
    scenes = _detect_scenes(preview, metadata["frame_count"])
    # Record a reusable preview only after the complete scene scan has decoded it.
    _atomic_json(manifest, {"source_sha256": source_hash, "preview_sha256": preview_hash,
                            "normalization": _NORMALIZATION})
    windows = _candidate_windows(scenes, metadata["frame_count"])
    chosen = _select_windows(windows, max_segments)
    progress("frames", f"Extracting evidence from {len(chosen)} of {len(windows)} candidate windows.")
    segments = _extract_frames(preview, output / "frames", [windows[index] for index in chosen])
    video = {"source_name": source.name, "source_sha256": source_hash,
             "source_bytes": source_stat.st_size, **probe, **metadata,
             "preview_path": str(preview), "preview_sha256": preview_hash,
             "normalized_fps": NORMALIZED_FPS,
             "audio_policy": "removed; visual-only; audio is not analyzed",
             "timebase": "zero-based normalized preview; actual decoded frame index / 12"}
    plan = {"pipeline": {**_NORMALIZATION, "window_seconds": WINDOW_SECONDS,
                          "scene_detector": "PySceneDetect AdaptiveDetector", "max_segments": max_segments,
                          "frames_per_segment": frames_per_segment},
            "video": video, "segments": segments,
            "selection": {"candidate_count": len(windows), "selected_count": len(chosen),
                          "unselected_count": len(windows) - len(chosen),
                          "strategy": "uniform_time_with_first_and_last; screenshot_positions=20%,80%",
                          "all_windows": [{"start_seconds": round(start / NORMALIZED_FPS, 6),
                                           "end_seconds": round(end / NORMALIZED_FPS, 6),
                                           "selected": index in chosen}
                                          for index, (start, end) in enumerate(windows)]}}
    _atomic_json(output / "plan.json", plan)
    progress("complete", "Video plan and timestamped evidence screenshots are ready.")
    return plan
