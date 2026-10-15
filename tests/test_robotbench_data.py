"""Pinned evidence preparation keeps ground truth and source hints out of inference."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from robotbench.data import load_samples, model_input, prepare_dataset, verify_download


class RobotBenchmarkDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = self.root / "metadata.jsonl"
        self.conversation = self.root / "conversation.jsonl"
        self.archive = self.root / "records.tar.gz"
        # Filename numbers run backwards; only the official list defines order.
        sources = [f"data/private/records/task+0/episode7/{stamp}_img_viewpoint_{view}.png"
                   for stamp in (4, 2) for view in range(3)]
        self.row = {"taskvar": "task+0", "episode_id": 7, "task_instruction": "put the cube in the box",
                    "detailed_subtask_name": "move grasped cube to box", "images": sources,
                    "execution_reward": 0, "reward": 0, "planning_reward": 1,
                    "failure_mode": "secret_failure_type", "failure_reason": "secret oracle explanation"}
        self.qa = {"id": 0, "image": sources,
                   "conversations": [{"from": "human", "value": "Compare the transition"},
                                     {"from": "gpt", "value": "<answer> False </answer> <category> wrong object state or placement: final state is wrong </category>"}]}
        self.image_bytes = []
        with tarfile.open(self.archive, "w:gz") as archive:
            for index, source in enumerate(sources):
                buffer = io.BytesIO()
                Image.new("RGB", (4, 4), color=(index * 30, 20, 40)).save(buffer, format="PNG")
                content = buffer.getvalue()
                self.image_bytes.append(content)
                member = tarfile.TarInfo("records/" + source.split("/records/", 1)[1])
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        self.write_sources()

    def write_sources(self):
        self.metadata.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        self.conversation.write_text(json.dumps(self.qa) + "\n", encoding="utf-8")

    def prepare(self):
        return prepare_dataset(self.metadata, self.conversation, self.archive, self.root / "prepared", expected_count=1)

    def test_separate_labels_canonical_category_and_no_source_hints_in_input(self):
        manifest = self.prepare()
        root = self.root / "prepared"
        inputs = json.loads((root / "inputs.jsonl").read_text())
        labels = json.loads((root / "labels.jsonl").read_text())
        self.assertEqual(labels["label"], "failure")
        self.assertEqual(labels["failure_category"], "wrong_state_or_placement")
        self.assertEqual(labels["task_group"], "task")
        self.assertNotIn("label", inputs)
        self.assertNotIn("failure", json.dumps(inputs))
        self.assertNotIn("episode7", json.dumps(inputs))
        self.assertNotIn("secret", json.dumps(inputs))
        self.assertEqual(manifest["samples"], 1)
        self.assertNotIn("accuracy", manifest)
        sample = load_samples(root)[0]
        safe = model_input(sample, root, view="all")
        self.assertEqual(set(safe), {"task", "subtask", "frames"})
        self.assertEqual([frame["phase"] for frame in safe["frames"]], ["before"] * 3 + ["after"] * 3)
        self.assertEqual(Path(safe["frames"][0]["path"]).read_bytes(), self.image_bytes[0])
        self.assertEqual(Path(safe["frames"][3]["path"]).read_bytes(), self.image_bytes[3])
        for frame in safe["frames"]:
            self.assertEqual(Path(frame["path"]).stem, frame["sha256"])

    def test_reversed_ablation_swaps_pixels_without_leaking_original_phase(self):
        self.prepare()
        root = self.root / "prepared"
        sample = load_samples(root)[0]
        before = deepcopy(sample)
        ordered = model_input(sample, root, "ordered", "0")
        reversed_input = model_input(sample, root, "reversed", "front")
        after = model_input(sample, root, "after_only", "0")
        self.assertEqual(sample, before)
        self.assertEqual([f["phase"] for f in reversed_input["frames"]], ["before", "after"])
        self.assertEqual([f["frame_id"] for f in reversed_input["frames"]], ["before_0", "after_0"])
        self.assertEqual(reversed_input["frames"][0]["sha256"], ordered["frames"][1]["sha256"])
        self.assertEqual(reversed_input["frames"][1]["sha256"], ordered["frames"][0]["sha256"])
        self.assertEqual(after["frames"], [ordered["frames"][1]])

    def test_disagreement_between_official_metadata_and_answers_is_rejected(self):
        self.qa["conversations"][1]["value"] = "<answer> True </answer> <category> success </category>"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.prepare()
        self.qa["image"] = list(reversed(self.row["images"]))
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "matching ordered"):
            self.prepare()

    def test_named_left_right_wrist_views_keep_positional_slots(self):
        names = ("left", "right", "wrist")
        sources = [source.replace(f"_viewpoint_{index % 3}.png", f"_viewpoint_{names[index % 3]}.png")
                   for index, source in enumerate(self.row["images"])]
        self.row["images"] = self.qa["image"] = sources
        self.write_sources()
        with tarfile.open(self.archive, "w:gz") as archive:
            for index, source in enumerate(sources):
                member = tarfile.TarInfo("records/" + source.split("/records/", 1)[1])
                member.size = len(self.image_bytes[index])
                archive.addfile(member, io.BytesIO(self.image_bytes[index]))
        self.prepare()
        safe = model_input(load_samples(self.root / "prepared")[0], self.root / "prepared", view="all")
        self.assertEqual([frame["view"] for frame in safe["frames"]], ["0", "1", "2"] * 2)
        self.assertEqual(Path(safe["frames"][2]["path"]).read_bytes(), self.image_bytes[2])

    def test_corrupted_labels_or_pixels_cannot_be_silently_loaded(self):
        self.prepare()
        root = self.root / "prepared"
        sample = load_samples(root)[0]
        (root / sample["frames"][0]["path"]).write_bytes(b"changed pixels")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            model_input(sample, root)
        (root / "labels.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "labels changed"):
            load_samples(root)

    def test_archive_path_traversal_and_symlink_are_rejected_without_extraction(self):
        for unsafe in ("records/../../escape.png", "/records/escape.png"):
            with self.subTest(name=unsafe):
                with tarfile.open(self.archive, "w:gz") as archive:
                    member = tarfile.TarInfo(unsafe)
                    member.size = len(self.image_bytes[0])
                    archive.addfile(member, io.BytesIO(self.image_bytes[0]))
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    self.prepare()
                self.assertFalse((self.root / "escape.png").exists())
        with tarfile.open(self.archive, "w:gz") as archive:
            member = tarfile.TarInfo("records/" + self.row["images"][0].split("/records/", 1)[1])
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside.png"
            archive.addfile(member)
        with self.assertRaisesRegex(ValueError, "linked"):
            self.prepare()

    def test_pinned_download_verifies_size_sha_and_git_blob(self):
        path = self.root / "download"
        content = b"fixed official bytes"
        path.write_bytes(content)
        spec = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "git_sha1": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()}
        verify_download(path, spec)
        for mutation in ({"size": 0}, {"sha256": "0" * 64}, {"git_sha1": "0" * 40}):
            with self.subTest(spec=mutation), self.assertRaises(ValueError):
                verify_download(path, {**spec, **mutation})


if __name__ == "__main__":
    unittest.main()
