import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import h5py
import numpy as np

from robotbench import temporal_data as td


def seal(plan):
    plan = copy.deepcopy(plan)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = hashlib.sha256(td._canonical(plan)).hexdigest()
    return plan


def sample(name="fixture_episode_0", task="block_assembly", split="development", verdict="fail"):
    annotation = {"sample": name, "task_id": task, "robot": "so101", "verdict": verdict}
    if verdict == "fail":
        annotation.update(onset=0.1, type=["wrong_object"], reason="PRIVATE GROUND TRUTH")
    return {"sample_id": "ft-" + hashlib.sha256(name.encode()).hexdigest()[:20], "source_sample": name,
            "task_id": task, "robot": "so101", "split": split, "bytes": 10, "sha256": "a" * 64,
            "relative_hdf5": "failtime_long/episodes/" + name + ".hdf5", "annotation": annotation}


def plan(*samples):
    return seal({"format": 1, "dataset": td.DATASET, "revision": td.REVISION, "samples": list(samples)})


class TemporalPlanTests(unittest.TestCase):
    def test_selection_is_order_invariant_excludes_dev_task_and_ignores_difficulty(self):
        rows = [{"sample": f"{task}_{label}_{number}", "task_id": task, "verdict": label,
                 "onset": number, "reason": "arbitrary difficulty"}
                for task in ["dev", "one", "two", "three"] for label in ["fail", "success"] for number in range(2)]
        kwargs = {"seed": "fixture", "development": ["dev_fail_0"], "task_count": 2}
        chosen = td.select_test_samples(rows, **kwargs)
        changed = [dict(row, onset=999, reason="changed difficulty") for row in reversed(rows)]
        self.assertEqual(td.select_test_samples(changed, **kwargs), chosen)
        self.assertEqual(len(chosen), 4)
        self.assertFalse(any(name.startswith("dev_") for name in chosen))
        self.assertEqual(sum("_fail_" in name for name in chosen), 2)

    def test_plan_detects_tampering_duplicates_and_dev_task_overlap(self):
        good = plan(sample(), sample("other_episode_1", "cube_stack", "test", "success"))
        td.validate_plan(good)
        bad = copy.deepcopy(good)
        bad["samples"][0]["annotation"]["onset"] = 5
        with self.assertRaisesRegex(ValueError, "digest"):
            td.validate_plan(bad)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            td.validate_plan(plan(sample(), sample()))
        with self.assertRaisesRegex(ValueError, "leaks"):
            td.validate_plan(plan(sample(), sample("other_episode_1", "block_assembly", "test")))

    def test_no_success_onset_or_invalid_failure_time(self):
        for value in [True, float("nan"), -1, "0.1"]:
            row = sample()
            row["annotation"]["onset"] = value
            if isinstance(value, float) and np.isnan(value):
                with self.assertRaises(ValueError):
                    plan(row)
            else:
                with self.assertRaises(ValueError):
                    td.validate_plan(plan(row))
        row = sample(verdict="success")
        row["annotation"]["onset"] = 0
        with self.assertRaisesRegex(ValueError, "invented"):
            td.validate_plan(plan(row))

    def test_path_traversal_rejected(self):
        for name in ["../escape", "C:/private", "folder\\file", "/absolute"]:
            with self.assertRaises(ValueError):
                td._relative(Path.cwd(), name)
        row = sample()
        row["relative_hdf5"] = "../private.hdf5"
        with self.assertRaises(ValueError):
            td.validate_plan(plan(row))

    def test_download_integrity_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"known content"
            expected = hashlib.sha256(content).hexdigest()
            path = root / "source.hdf5"
            with mock.patch.object(td.urllib.request, "urlopen", return_value=io.BytesIO(content)) as request:
                td._download_verified("https://example.test/file", path, len(content), expected)
                td._download_verified("https://example.test/file", path, len(content), expected)
                self.assertEqual(request.call_count, 1)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Existing"):
                td._download_verified("https://example.test/file", path, len(content), expected)

    def test_bad_download_is_never_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.hdf5"
            with mock.patch.object(td.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(ValueError):
                    td._download_verified("https://example.test/file", path, 3, "0" * 64)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_download_resumes_only_the_exact_missing_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.hdf5"
            content = b"123456789"
            expected = hashlib.sha256(content).hexdigest()
            partial = path.with_name(path.name + "." + expected[:12] + ".part")
            partial.write_bytes(content[:3])
            response = io.BytesIO(content[3:])
            response.status = 206
            response.headers = {"Content-Range": "bytes 3-8/9"}
            with mock.patch.object(td.urllib.request, "urlopen", return_value=response) as request:
                td._download_verified("https://example.test/file", path, len(content), expected)
                self.assertEqual(request.call_args.args[0].headers["Range"], "bytes=3-8")
            self.assertEqual(path.read_bytes(), content)
            self.assertFalse(partial.exists())

    def test_wrong_range_is_not_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.hdf5"
            expected = hashlib.sha256(b"123456789").hexdigest()
            partial = path.with_name(path.name + "." + expected[:12] + ".part")
            partial.write_bytes(b"123")
            response = io.BytesIO(b"123456789")
            response.status = 206
            response.headers = {"Content-Range": "bytes 0-8/9"}
            with mock.patch.object(td.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "different byte range"):
                    td._download_verified("https://example.test/file", path, 9, expected)
            self.assertEqual(partial.read_bytes(), b"123")

    def test_author_metadata_is_independently_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            row = sample()
            (root / "source/test.json").write_text(json.dumps([row["annotation"]]))
            with mock.patch.object(td, "_download_verified"):
                td._verify_author_annotations(plan(row), root, root / "plan.json")
                row["annotation"]["onset"] = 99
                with self.assertRaisesRegex(ValueError, "author annotation"):
                    td._verify_author_annotations(plan(row), root, root / "plan.json")


class TemporalMediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.hdf5"
        self.task = self.root / "task.xml"
        self.task.write_text('<task><description>Place red block.\\nThen release.</description><substages><substage>grasp</substage><substage>place</substage></substages></task>')
        video = self.root / "fixture.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
        self.assertTrue(writer.isOpened())
        for i in range(9):
            writer.write(np.full((32, 32, 3), i * 20, dtype=np.uint8))
        writer.release()
        raw = np.frombuffer(video.read_bytes(), dtype=np.uint8)
        with h5py.File(self.source, "w") as data:
            data.attrs["fps"] = 30
            data.attrs["task_description"] = "WRONG OLD TASK; DO NOT USE"
            data.create_dataset("timestamp", data=np.arange(9, dtype=np.float64) / 30)
            data.create_dataset("frame_index", data=np.arange(9))
            state = data.create_dataset("observation.state_eef", data=np.zeros((9, 16)))
            state.attrs["frame"] = "separate arm base frames"
            state.attrs["reason"] = "PRIVATE GROUND TRUTH"
            data.create_dataset("observation.state", data=np.zeros((9, 12)))
            data.create_dataset("action", data=np.zeros((9, 12)))
            for camera in ["head", "left_wrist", "right_wrist"]:
                group = data.create_group("observation.images." + camera)
                group.create_dataset("timestamp", data=np.arange(9) / 30)
                group.create_dataset("mp4_bytes", data=raw)
        self.sample = sample()
        self.sample["bytes"] = self.source.stat().st_size
        self.sample["sha256"] = td.sha256_file(self.source)

    def tearDown(self):
        self.tmp.cleanup()

    def extract(self):
        return td._extract_sample(self.source, self.root, self.sample, self.task)

    def write_input(self, record):
        (self.root / "inputs.jsonl").write_text(json.dumps(record) + "\n")
        (self.root / "manifest.json").write_text(json.dumps({"inputs_sha256": td.sha256_file(self.root / "inputs.jsonl")}))

    def test_real_video_extract_input_allowlist_and_resume(self):
        record = self.extract()
        self.assertIn("Place red block.\nThen release.", record["task_description"])
        self.assertNotIn("reason", record["eef_metadata"])
        self.assertEqual(record["head_view"], "head")
        self.assertEqual(len(record["videos"]), 3)
        self.assertNotIn("PRIVATE GROUND TRUTH", json.dumps(record))
        with mock.patch.object(td, "_video_metadata", side_effect=AssertionError("must reuse complete cache")):
            self.assertEqual(self.extract(), record)
        self.write_input(record)
        # No labels.jsonl or plan.json exists: prediction loading must still work.
        loaded = td.load_inputs(self.root)[0]
        self.assertEqual(set(loaded), td.INPUT_FIELDS)
        self.assertEqual(len(loaded["timestamp"]), 9)
        self.assertEqual(len(loaded["state_eef"]), 9)
        self.assertTrue(Path(loaded["videos"]["head"]).is_absolute())

    def test_prepare_publishes_separate_labels_and_plan_bound_manifest(self):
        destination = self.root / "dataset"
        source_copy = self.root / (self.sample["source_sample"] + ".hdf5")
        source_copy.write_bytes(self.source.read_bytes())
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan(self.sample)))
        task_path = destination / "source/tasks/block_assembly.xml"
        task_path.parent.mkdir(parents=True)
        task_path.write_bytes(self.task.read_bytes())
        raw = task_path.read_bytes()
        expected = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        with mock.patch.object(td, "_verify_author_annotations"), mock.patch.dict(td.TASK_GIT_BLOBS, {"block_assembly": expected}):
            manifest_path = td.prepare_dataset(plan_path, destination)
            td.prepare_dataset(plan_path, destination)
        manifest = json.loads(manifest_path.read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["samples_prepared"], 1)
        inputs = td.load_inputs(destination)
        labels = td.load_labels(destination)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(labels[0]["onset_seconds"], 0.1)
        self.assertEqual(labels[0]["label"], "failure")
        self.assertNotIn("onset", json.dumps(inputs))
        self.assertNotIn("PRIVATE GROUND TRUTH", json.dumps(inputs))

    def test_video_and_state_cache_tampering_rejected(self):
        record = self.extract()
        self.write_input(record)
        (self.root / record["videos"]["head"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "video changed"):
            td.load_inputs(self.root)
        with self.assertRaisesRegex(ValueError, "cache was modified"):
            self.extract()

    def test_label_injected_into_inputs_rejected(self):
        record = self.extract()
        record["onset_seconds"] = 0.1
        self.write_input(record)
        with self.assertRaisesRegex(ValueError, "label-bearing"):
            td.load_inputs(self.root)

    def test_task_text_edit_breaks_input_identity(self):
        record = self.extract()
        self.write_input(record)
        record["task_description"] = "Changed after preparing"
        (self.root / "inputs.jsonl").write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            td.load_inputs(self.root)

    def test_camera_clock_mismatch_rejected(self):
        with h5py.File(self.source, "r+") as data:
            data["observation.images.head/timestamp"][2] += 0.2
        with self.assertRaisesRegex(ValueError, "Camera timeline"):
            self.extract()

    def test_nonmonotonic_state_time_rejected(self):
        with h5py.File(self.source, "r+") as data:
            data["timestamp"][2] = 0
        with self.assertRaisesRegex(ValueError, "increase"):
            self.extract()

    def test_video_fps_mismatch_rejected(self):
        with h5py.File(self.source, "r+") as data:
            data.attrs["fps"] = 25
            data["timestamp"][:] = np.arange(9) / 25
            for camera in ["head", "left_wrist", "right_wrist"]:
                data["observation.images." + camera + "/timestamp"][:] = np.arange(9) / 25
        with self.assertRaisesRegex(ValueError, "frame count/fps"):
            self.extract()


if __name__ == "__main__":
    unittest.main()
