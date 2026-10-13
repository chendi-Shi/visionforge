"""Timebase, budget, and cache safety checks; no models or FFmpeg processes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import video_media as media


class WindowTests(unittest.TestCase):
    def test_long_shot_is_completely_covered_without_large_windows(self):
        windows = media._candidate_windows([(0, 421)], 421)
        self.assertEqual(windows, [(0, 180), (180, 360), (360, 421)])
        self.assertEqual(sum(end - start for start, end in windows), 421)
        self.assertTrue(all(end - start <= 15 * 12 for start, end in windows))

    def test_missing_scene_endpoints_do_not_drop_source_frames(self):
        self.assertEqual(media._candidate_windows([(12, 24)], 36), [(0, 12), (12, 24), (24, 36)])
        self.assertEqual(media._candidate_windows([], 12), [(0, 12)])

    def test_budget_spreads_over_time_instead_of_cut_count(self):
        # Ten cuts in the first second, then three long windows.
        windows = [(index, index + 1) for index in range(10)] + [(10, 190), (190, 370), (370, 550)]
        selected = media._select_windows(windows, 3)
        self.assertEqual(selected, [0, 11, 12])
        self.assertEqual(media._select_windows(windows, 2), [0, 12])
        self.assertEqual(media._select_windows(windows[:2], 4), [0, 1])

    def test_evidence_is_inside_window_and_duplicate_frames_are_removed(self):
        self.assertEqual(media._evidence_indices(0, 180), [36, 143])
        self.assertEqual(media._evidence_indices(60, 61), [60])
        for start, end in ((0, 2), (1, 4), (180, 421)):
            selected = media._evidence_indices(start, end)
            self.assertEqual(selected, sorted(set(selected)))
            self.assertTrue(all(start <= index < end for index in selected))


class ValidationTests(unittest.TestCase):
    def test_probe_uses_duration_not_unreliable_source_fps(self):
        probe = media._parse_source_probe(
            "Duration: 00:00:35.03, start: 0.000000, bitrate: 344 kb/s\n"
            "Stream #0:0: Video: vp9, yuv420p, 640x360, 1k fps, 30 tbr, 1k tbn"
        )
        self.assertEqual(probe["source_duration_seconds"], 35.03)
        self.assertEqual(probe["source_start_seconds"], 0)

    def test_probe_rejects_audio_only_unknown_and_overlong_sources(self):
        for value in ("Duration: N/A\nStream #0:0: Video: vp9",
                      "Duration: 00:00:01.00\nStream #0:0: Audio: mp3",
                      "Duration: 00:30:00.01\nStream #0:0: Video: vp9"):
            with self.subTest(value=value), self.assertRaises(media.VideoPreparationError):
                media._parse_source_probe(value)

    def test_invalid_budget_is_rejected_before_any_media_processing(self):
        for budget in (True, 1, 13, 2.5):
            with self.subTest(budget=budget), self.assertRaises(media.VideoPreparationError):
                media.prepare_video("missing.mp4", "unused", max_segments=budget)
        with self.assertRaises(media.VideoPreparationError):
            media.prepare_video("missing.mp4", "unused", frames_per_segment=3)

    def test_size_limit_is_checked_before_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.webm"
            source.write_bytes(b"12345")
            with patch.object(media, "MAX_SOURCE_BYTES", 4), patch.object(media, "_probe_source") as probe:
                with self.assertRaisesRegex(media.VideoPreparationError, "1 GiB"):
                    media.prepare_video(source, Path(temporary) / "output")
                probe.assert_not_called()

    def test_decoder_position_must_match_requested_evidence(self):
        capture = Mock()
        capture.set.return_value = True
        pixels = object()
        capture.read.return_value = (True, pixels)
        capture.get.return_value = 61.0
        self.assertEqual(media._read_frame(capture, 60), (60, pixels))
        capture.get.return_value = 60.0
        with self.assertRaisesRegex(media.VideoPreparationError, "inaccurate evidence timestamp"):
            media._read_frame(capture, 60)

    def test_cached_preview_requires_matching_source_hash_content_and_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            preview = Path(temporary) / "preview.mp4"
            manifest = Path(temporary) / "preview-info.json"
            preview.write_bytes(b"preview")
            manifest.write_text(json.dumps({"source_sha256": "source", "preview_sha256": media._sha256(preview),
                                            "normalization": media._NORMALIZATION}), encoding="utf-8")
            with patch.object(media, "_preview_metadata", return_value={"frame_count": 12}) as decode:
                self.assertEqual(media._cached_preview(preview, manifest, "source"), {"frame_count": 12})
                decode.assert_called_once()
                self.assertIsNone(media._cached_preview(preview, manifest, "different-source"))
                preview.write_bytes(b"corrupted")
                self.assertIsNone(media._cached_preview(preview, manifest, "source"))
                self.assertEqual(decode.call_count, 1)
            preview.write_bytes(b"preview")
            with patch.object(media, "_preview_metadata", side_effect=media.VideoPreparationError("bad decoder")):
                self.assertIsNone(media._cached_preview(preview, manifest, "source"))


if __name__ == "__main__":
    unittest.main()
