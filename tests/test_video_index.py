"""Visual-index identity, corruption, and ranking tests without model downloads."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

import video_index as index


class FakeEmbedder:
    def __init__(self, revision="test-v1", vector=(1, 0, 0)):
        self.revision = revision
        self.vector = vector
        self.image_calls = 0
        self.text_calls = 0

    def identity(self):
        return {"name": "fake-paired-encoder", "revision": self.revision, "dimension": 3}

    def embed_images(self, paths):
        self.image_calls += 1
        return [np.asarray(self.vector, dtype=np.float32) for path in paths]

    def embed_text(self, text):
        self.text_calls += 1
        return np.asarray([7, 0, 0], dtype=np.float32)


class SamplingTests(unittest.TestCase):
    def test_release_models_drops_cached_encoder_and_next_use_reloads(self):
        index.release_models()
        self.addCleanup(index.release_models)
        with patch.object(index, "FastEmbedCLIP", side_effect=FakeEmbedder) as factory:
            first = weakref.ref(index._default_embedder())
            self.assertIsNotNone(first())
            self.assertIs(first(), index._default_embedder())
            index.release_models()
            self.assertIsNone(first())
            self.assertEqual(index._default_embedder.cache_info().currsize, 0)
            index._default_embedder()
            self.assertEqual(factory.call_count, 2)

    def test_clip_checks_full_query_before_silently_truncating_target_words(self):
        engine = object.__new__(index.FastEmbedCLIP)
        engine._lock = threading.Lock()
        engine._query_tokenizer = Mock()
        engine._texts = Mock()
        engine._query_tokenizer.encode.return_value.ids = list(range(78))
        with self.assertRaisesRegex(index.VideoIndexError, "77-token"):
            engine.embed_text("a " * 80 + "red bus")
        engine._texts.embed.assert_not_called()
        engine._query_tokenizer.encode.return_value.ids = list(range(77))
        engine._texts.embed.return_value = iter([np.ones(512)])
        self.assertEqual(engine.embed_text("a red bus").shape, (512,))

    def test_uniform_sampling_keeps_both_ends_and_records_real_gaps(self):
        frames, info = index._sampling(420, 12, 2.0, 900)
        self.assertEqual(frames, list(range(0, 420, 24)) + [419])
        self.assertEqual(len(frames), 19)
        self.assertFalse(info["budget_adjusted"])
        self.assertEqual(info["maximum_gap_seconds"], 2)
        self.assertAlmostEqual(info["last_time_seconds"], 419 / 12)

    def test_budget_uses_complete_timeline_instead_of_truncating_tail(self):
        frames, info = index._sampling(21600, 12, 0.001, 900)
        self.assertEqual((len(frames), frames[0], frames[-1]), (900, 0, 21599))
        self.assertEqual(frames, sorted(set(frames)))
        self.assertTrue(info["budget_adjusted"])
        self.assertGreater(info["effective_interval_seconds"], 2)
        self.assertLessEqual(info["maximum_gap_seconds"], info["effective_interval_seconds"] + 1 / 12)

    def test_single_frame_video_and_extreme_finite_interval(self):
        self.assertEqual(index._sampling(1, 12, 2, 2)[0], [0])
        self.assertEqual(index._sampling(420, 12, 1e300, 900)[0], [0, 419])

    def test_invalid_options_fail_before_any_download(self):
        for value in (True, 0, -1, float("nan"), float("inf"), "2"):
            with self.subTest(value=value), self.assertRaises(index.VideoIndexError):
                index.build_index("missing", "unused", interval_seconds=value)
        for budget in (True, 1, 901, 2.1):
            with self.subTest(budget=budget), self.assertRaises(index.VideoIndexError):
                index.build_index("missing", "unused", max_frames=budget)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.preview = self.root / "preview.mp4"
        self.preview.write_bytes(b"stable-preview-content")
        self.output = self.root / "index"
        self.engine = FakeEmbedder(vector=(3, 0, 0))
        self.metadata = {"width": 8, "height": 8, "fps": 12.0, "frame_count": 120,
                         "duration_seconds": 10.0, "preview_duration_seconds": 10.0}
        for patcher in (
            patch.object(index, "_preview_metadata", return_value=self.metadata),
            patch.object(index, "_open_capture", return_value=Mock()),
            patch.object(index, "_read_frame", side_effect=lambda cap, n: (n, np.full((8, 8, 3), n % 255, dtype=np.uint8))),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, **kwargs):
        return index.build_index(self.preview, self.output, embedder=self.engine, **kwargs)

    def test_cache_hit_verifies_identity_and_avoids_reembedding(self):
        first = self.build()
        self.assertFalse(first["cache_hit"])
        self.assertEqual(first["sampling"]["frame_count"], 6)
        json.dumps(first, allow_nan=False)
        calls = self.engine.image_calls
        second = self.build()
        self.assertTrue(second["cache_hit"])
        self.assertEqual(self.engine.image_calls, calls)
        with np.load(self.output / first["vectors_file"], allow_pickle=False) as data:
            self.assertEqual(data["vectors"].dtype, np.float32)
            np.testing.assert_allclose(np.linalg.norm(data["vectors"], axis=1), 1)

    def test_different_model_identity_requires_rebuild_for_search(self):
        original = self.build()
        self.engine.revision = "changed-model-files"
        with self.assertRaisesRegex(index.VideoIndexError, "embedder"):
            index.search_index(self.output, "a red bus", embedder=self.engine)
        changed = self.build()
        self.assertFalse(changed["cache_hit"])
        self.assertNotEqual(original["identity_sha256"], changed["identity_sha256"])

    def test_preview_and_sampling_changes_invalidate_cache(self):
        original = self.build()
        self.preview.write_bytes(b"replacement-preview-content")
        with self.assertRaisesRegex(index.VideoIndexError, "preview"):
            index.search_index(self.output, "a red bus", embedder=self.engine)
        changed = self.build()
        self.assertFalse(changed["cache_hit"])
        self.assertNotEqual(original["identity_sha256"], changed["identity_sha256"])
        self.assertFalse(self.build(interval_seconds=3)["cache_hit"])

    def test_modified_evidence_refuses_search_and_rebuild_repairs_it(self):
        saved = self.build()
        (self.output / saved["frames"][0]["path"]).write_bytes(b"not-the-indexed-image")
        with self.assertRaisesRegex(index.VideoIndexError, "evidence image"):
            index.search_index(self.output, "a red bus", embedder=self.engine)
        self.assertFalse(self.build()["cache_hit"])
        self.assertEqual(len(index.search_index(self.output, "a red bus", embedder=self.engine)), 3)

    def test_npz_corruption_and_row_mismatch_are_detected(self):
        saved = self.build()
        path = self.output / saved["vectors_file"]
        path.write_bytes(b"truncated")
        with self.assertRaisesRegex(index.VideoIndexError, "vectors"):
            index.search_index(self.output, "a red bus", embedder=self.engine)
        saved = self.build()
        path = self.output / saved["vectors_file"]
        with np.load(path, allow_pickle=False) as data:
            changed = {key: data[key].copy() for key in data.files}
        changed["image_sha256"][0] = "0" * 64
        np.savez_compressed(path, **changed)
        saved["vectors_sha256"] = index._sha256(path)
        (self.output / "index.json").write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(index.VideoIndexError, "vector rows"):
            index.search_index(self.output, "a red bus", embedder=self.engine)

    def test_forged_frame_timestamp_or_external_path_is_not_used(self):
        saved = self.build()
        changed = copy.deepcopy(saved)
        changed["frames"][0]["time_seconds"] = 99
        (self.output / "index.json").write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(index.VideoIndexError, "timestamp"):
            index.search_index(self.output, "a red bus", embedder=self.engine)
        changed = copy.deepcopy(saved)
        changed["frames"][0]["path"] = "../outside.png"
        (self.output / "index.json").write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(index.VideoIndexError, "escapes"):
            index.search_index(self.output, "a red bus", embedder=self.engine)

    def test_time_deduplication_is_inclusive_stable_and_can_return_fewer(self):
        self.build()
        results = index.search_index(self.output, "a red bus", top_k=3, min_separation_seconds=4, embedder=self.engine)
        self.assertEqual([r["time_seconds"] for r in results], [0, 4, 8])
        self.assertTrue(all(abs(r["score"] - 1) < 1e-6 for r in results))
        self.assertTrue(all(Path(r["path"]).is_absolute() and Path(r["path"]).exists() for r in results))
        all_results = index.search_index(self.output, "a red bus", top_k=900, min_separation_seconds=0, embedder=self.engine)
        self.assertEqual(len(all_results), 6)
        far_apart = index.search_index(self.output, "a red bus", top_k=3, min_separation_seconds=100, embedder=self.engine)
        self.assertEqual(len(far_apart), 1)

    def test_cosine_ranking_uses_vectors_not_input_order_or_vector_length(self):
        vectors = iter([[0, 4, 0], [-20, 0, 0], [3, 4, 0], [100, 0, 0], [1, 2, 0], [0, 1, 0]])
        self.engine.embed_images = lambda paths: [next(vectors) for _ in paths]
        self.build()
        results = index.search_index(self.output, "a bus", top_k=3, min_separation_seconds=0, embedder=self.engine)
        self.assertEqual([r["frame_index"] for r in results], [72, 48, 96])
        self.assertAlmostEqual(results[1]["score"], .6, places=6)

    def test_bad_embeddings_do_not_publish_a_successful_index(self):
        for vector in ([0, 0, 0], [float("nan"), 0, 0], [1, 2]):
            self.engine.vector = vector
            with self.subTest(vector=vector), self.assertRaises(index.VideoIndexError):
                self.build()
            self.assertFalse((self.output / "index.json").exists())

    def test_failed_rebuild_preserves_previous_complete_manifest(self):
        original = self.build()
        before = (self.output / "index.json").read_bytes()
        self.engine.revision = "new-model"
        self.engine.vector = [0, 0, 0]
        with self.assertRaises(index.VideoIndexError):
            self.build()
        self.assertEqual((self.output / "index.json").read_bytes(), before)
        previous = FakeEmbedder()
        self.assertEqual(len(index.search_index(self.output, "a bus", embedder=previous)), 3)
        self.assertTrue((self.output / original["vectors_file"]).is_file())

    def test_invalid_queries_and_options_fail_before_query_embedding(self):
        self.build()
        for query in ("", "  ", None, "a" * 513, "公交车"):
            with self.subTest(query=query), self.assertRaises(index.VideoIndexError):
                index.search_index(self.output, query, embedder=self.engine)
        for options in ({"top_k": 0}, {"top_k": True}, {"top_k": 901},
                        {"min_separation_seconds": -1}, {"min_separation_seconds": float("nan")}):
            with self.subTest(options=options), self.assertRaises(index.VideoIndexError):
                index.search_index(self.output, "a bus", embedder=self.engine, **options)
        self.assertEqual(self.engine.text_calls, 0)


if __name__ == "__main__":
    unittest.main()
