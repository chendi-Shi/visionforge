"""Independent release verification and re-signed tampering regressions."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import export_vlm_diagnostic as release

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts" / "robot-vlm-diagnostic"


@unittest.skipUnless((BUNDLE / "manifest.json").is_file(), "Published diagnostic fixture is unavailable")
class DiagnosticReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vdr-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bundle"
        shutil.copytree(BUNDLE, self.root)

    def resign(self, changed, *, remove=False):
        path = self.root / "manifest.json"
        manifest = release.read(path)
        if remove:
            del manifest["files"][changed]
        else:
            manifest["files"][changed] = release.sha(self.root / changed)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        (self.root / "manifest.sha256").write_text(release.sha(path), encoding="utf-8")

    def test_bundle_recomputes_with_no_site_packages_or_repo_imports(self):
        result = subprocess.run([sys.executable, "-I", "-S", str(self.root / "verify.py"), "verify", str(self.root)],
                                capture_output=True, text=True, timeout=40, cwd=self.temp.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertTrue(parsed["verified"])
        self.assertEqual(parsed["conditions"]["no_prefill"]["error_counts"], {"truncated": 4})

    def test_missing_runtime_is_rejected_even_after_resigning_inventory(self):
        name = "runtime/core.py"
        (self.root / name).unlink()
        self.resign(name, remove=True)
        with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
            release.verify(self.root)

    def test_recomputed_report_rejects_re_signed_better_score(self):
        name = "report.json"
        value = release.read(self.root / name)
        value["conditions"]["baseline_short"]["correct_of_fixed_four"] = 4
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        self.resign(name)
        with self.assertRaisesRegex(ValueError, "independent raw-response"):
            release.verify(self.root)

    def test_re_signed_prediction_must_match_raw_model_answer(self):
        name = "records/ur5-a17751fac2eb11b5cc8d.baseline_short.json"
        value = release.read(self.root / name)
        value["prediction"] = "failure"
        value["record_sha256"] = release.digest({k: v for k, v in value.items() if k != "record_sha256"})
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        self.resign(name)
        with self.assertRaisesRegex(ValueError, "disagree with raw"):
            release.verify(self.root)

    def test_author_labels_are_bound_separately_from_saved_score(self):
        name = "source-labels.jsonl"
        text = (self.root / name).read_text(encoding="utf-8").replace('"label":"failure"', '"label":"success"', 1)
        (self.root / name).write_text(text, encoding="utf-8")
        self.resign(name)
        with self.assertRaisesRegex(ValueError, "Source identity"):
            release.verify(self.root)

    def test_no_omitted_prediction_can_improve_denominator(self):
        name = "records/ur5-a17751fac2eb11b5cc8d.no_prefill.json"
        (self.root / name).unlink()
        self.resign(name, remove=True)
        with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
            release.verify(self.root)

    def test_private_paths_are_rejected_even_with_valid_checksums(self):
        name = "README.txt"
        (self.root / name).write_text("A private file is C:\\Users\\Example\\private.txt", encoding="utf-8")
        self.resign(name)
        with self.assertRaisesRegex(ValueError, "Private absolute path"):
            release.verify(self.root)

    def test_failed_staged_export_does_not_block_retry(self):
        destination = Path(self.temp.name) / "retry"
        arguments = (REPO / "data/robotbench", REPO / "data/vlm-diagnostic",
                     REPO / "data/robot-run/vlm-plan.json", destination)
        with patch.object(release, "verify", side_effect=ValueError("deliberate failure")):
            with self.assertRaisesRegex(ValueError, "deliberate failure"):
                release.export(*arguments)
        self.assertFalse(destination.exists())
        self.assertFalse(list(destination.parent.glob(".vd-*")))
        self.assertTrue(release.export(*arguments)["verified"])

    def test_existing_verified_bundle_does_not_hide_changed_raw_run(self):
        run = Path(self.temp.name) / "run"
        shutil.copytree(REPO / "data/vlm-diagnostic", run)
        arguments = (REPO / "data/robotbench", run, REPO / "data/robot-run/vlm-plan.json", self.root)
        self.assertTrue(release.export(*arguments)["verified"])
        (run / "report.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "current raw run bytes"):
            release.export(*arguments)


if __name__ == "__main__":
    unittest.main()
