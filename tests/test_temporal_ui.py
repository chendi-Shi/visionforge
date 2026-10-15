"""Temporary fixtures exercise media identity and switching; no model calls."""
import json
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import gradio as gr

from robotbench import temporal_ui as ui
from robotbench.temporal_plot import render_motion_plot


class TemporalUITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "predictions").mkdir()
        samples = [{"sample_id": sid, "task_id": sid, "task_description": "Synthetic test fixture",
                    "label": "failure" if sid == "a" else "success", "onset_seconds": 3.25 if sid == "a" else None,
                    "video_file": "videos/" + sid + ".mp4"} for sid in ("a", "b")]
        self.save("samples.json", samples)
        for sid in ("a", "b"):
            for method, times in (("uniform", [0.0, 5.0]), ("state_event", [0.0, 3.0])):
                self.save(f"predictions/{sid}--{method}.json", {
                    "sample_id": sid, "method": method, "selection": {"timestamps": times},
                    "frame_id": 1, "onset_seconds": times[1], "prediction": "failure", "status": "ok",
                    "reason": "Synthetic fixture", "panel_file": f"panels/{sid}--{method}.jpg"})

    def save(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def test_sample_and_condition_resolve_same_evidence(self):
        for sid in ("a", "b"):
            for method, second in (("uniform", 5.0), ("state_event", 3.0)):
                result = ui.detail(self.root, sid, method)
                self.assertEqual(Path(result[1]).name, sid + ".mp4")
                self.assertEqual(Path(result[2]).name, f"{sid}--{method}.jpg")
                binding = ui.view_binding(result, sid, method)
                self.assertNotIn(str(self.root), json.dumps(binding))
                request = ui.seek_request(self.root, sid, method, "1", binding)
                self.assertEqual(request, {**binding, "seconds": second})

    def test_stale_selection_and_mislabeled_records_fail_closed(self):
        binding = ui.view_binding(ui.detail(self.root, "a", "uniform"), "a", "uniform")
        for sid, method in (("b", "uniform"), ("a", "state_event")):
            with self.assertRaises(gr.Error):
                ui.seek_request(self.root, sid, method, "1", binding)
        self.save("predictions/a--uniform.json", {"sample_id": "b", "method": "uniform"})
        with self.assertRaises(ValueError):
            ui.selected_time(self.root, "a", "uniform", "1")
        self.assertIsNone(ui.detail(self.root, "a", "uniform")[1])

    def test_author_onset_and_success_absence(self):
        for sid, expected in (("a", 3.25), ("b", -1)):
            binding = ui.view_binding(ui.detail(self.root, sid, "uniform"), sid, "uniform")
            result = ui.seek_request(self.root, sid, "uniform", None, binding, truth=True)
            self.assertEqual(result["seconds"], expected)

    def test_missing_export_builds_two_independent_model_views(self):
        missing = self.root / "not-exported"
        self.assertEqual(ui.overview(missing)[1:], ([], []))
        self.assertIsNone(ui.downloads(missing))
        self.assertIn("尚未导出", ui.trajectory_plot(missing, "a", "uniform"))
        with gr.Blocks() as demo:
            ui.build_temporal_tab(missing, video_id="temporal-video")
            ui.build_temporal_tab(missing, video_id="temporal-video-minicpm")
        ids = [c["props"].get("elem_id") for c in demo.config["components"] if c["props"].get("elem_id")]
        self.assertEqual(len(ids), len(set(ids)))
        scripts = [d["js"] for d in demo.config["dependencies"] if d.get("js")]
        self.assertTrue(any('id="temporal-video"' in s for s in scripts))
        self.assertTrue(any('id="temporal-video-minicpm"' in s for s in scripts))
        self.save("evaluation.json", {})
        self.assertEqual(ui.downloads(self.root), [str(self.root / "evaluation.json")])

    @unittest.skipUnless(shutil.which("node"), "Node is needed only for the browser seek unit test")
    def test_browser_waits_for_matching_media_and_cancels_per_model(self):
        script = """
const assert=require('node:assert/strict');
global.window={location:{href:'http://127.0.0.1/'}};
const timers=[];global.setTimeout=(fn)=>timers.push(fn);
const videos={q:{currentSrc:'http://localhost/cache/old/a.mp4',readyState:1,duration:100,currentTime:0,play:()=>Promise.resolve(),pause:()=>{}},
 m:{currentSrc:'http://localhost/cache/another/a.mp4',readyState:1,duration:100,currentTime:0,play:()=>Promise.resolve(),pause:()=>{}}};
global.document={getElementById:(id)=>({querySelector:()=>videos[id]})};
const drain=()=>{while(timers.length)timers.shift()();};
"""
        script += "const q=" + ui.SEEK_JS.replace("__VIDEO_ID__", '"q"') + ";\n"
        script += "const m=" + ui.SEEK_JS.replace("__VIDEO_ID__", '"m"') + ";\n"
        script += "const cancel=" + ui.CANCEL_SEEK_JS.replace("__VIDEO_ID__", '"q"') + ";\n"
        script += """
const bind=(id)=>({sample_id:id,method:'uniform',video_name:id+'.mp4'});
q({...bind('b'),seconds:12},'b','uniform',bind('b'));
assert.equal(videos.q.currentTime,0);assert.equal(timers.length,1);
m({...bind('a'),seconds:7},'a','uniform',bind('a'));
assert.equal(videos.m.currentTime,7);
videos.q.currentSrc='http://localhost/new%20cache/hash/b.mp4';drain();
assert.equal(videos.q.currentTime,12);assert.equal(videos.m.currentTime,7);
q({...bind('a'),seconds:31},'a','uniform',bind('a'));cancel();
videos.q.currentSrc='http://localhost/cache/a.mp4';drain();
assert.equal(videos.q.currentTime,12);assert.equal(videos.m.currentTime,7);
q({...bind('a'),seconds:42},'b','uniform',bind('b'));drain();
assert.equal(videos.q.currentTime,12);
"""
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TemporalPlotTest(unittest.TestCase):
    def setUp(self):
        identity = {"sample_id": "test<script>", "task_id": "task", "source_sha256": "source"}
        self.episode = {**identity, "timestamp": [i * .02 for i in range(11)] + [10.0]}
        self.sample = {**identity, "label": "failure", "onset_seconds": .15}
        self.record = {**identity, "method": "state_event", "status": "ok", "prediction": "failure", "frame_id": 10,
                       "onset_seconds": .2, "selection": {"method": "state_event", "indices": list(range(12)),
                       "timestamps": self.episode["timestamp"].copy(), "scores": [1.0] * 12, "event_indices": [10]}}

    def test_measured_irregular_time_markers_and_escaping(self):
        result = render_motion_plot(self.episode, self.sample, self.record)
        self.assertIn('cx="78.24"', result)  # 0.2 measured seconds / 10; never index / fps.
        self.assertEqual(result.count("<circle "), 12)
        self.assertIn("作者 0.150s", result)
        self.assertIn("模型 0.200s", result)
        self.assertIn("事件 1 帧", result)
        self.assertIn("test&lt;script&gt;", result)
        self.assertNotIn("<script>", result)
        self.assertIn("不是失败概率", result)

    def test_misalignment_and_nonfinite_values_are_rejected(self):
        for field, value in (("scores", [1.0]), ("scores", [float("nan")] * 12),
                             ("timestamps", list(range(12))), ("event_indices", [13])):
            record = copy.deepcopy(self.record)
            record["selection"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_motion_plot(self.episode, self.sample, record)
        episode = copy.deepcopy(self.episode)
        episode["source_sha256"] = "different"
        with self.assertRaises(ValueError):
            render_motion_plot(episode, self.sample, self.record)

    def test_stationary_success_has_no_invented_onset(self):
        sample = {**self.sample, "label": "success", "onset_seconds": None}
        record = copy.deepcopy(self.record)
        record.update(prediction="success", onset_seconds=None, frame_id=-1)
        record["selection"]["scores"] = [0.0] * 12
        result = render_motion_plot(self.episode, sample, record)
        self.assertIn("成功轨迹，无失败起点", result)
        self.assertIn("无有效失败定位", result)
        self.assertNotIn("nan", result)
        self.assertNotIn("作者 0.", result)


if __name__ == "__main__":
    unittest.main()
