"""Exercise real Gradio session preprocessing without workers or models."""
from copy import deepcopy
import asyncio
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from gradio.state_holder import SessionState
import app


class Library:
    def __init__(self):
        self.runs = [{"id": "saved-run", "target": "find object"}]
        self.read_ids = []
        self.created_scopes = []

    def list_searches(self):
        return deepcopy(self.runs)

    def list_videos(self, include_archived=True):
        return [{"id": "video-1", "name": "sample", "title": "sample", "status": "ready", "ready_for_search": True,
                 "tags": [], "video": {"duration_seconds": 1.}}]

    def get_search(self, identifier):
        self.read_ids.append(identifier)
        assert identifier in {row["id"] for row in self.runs}
        return {"status": "completed", "children": [], "progress": {"videos_completed": 1, "videos_total": 1,
                "frames_verified": 0, "videos_failed": 0}}

    def create_search(self, job_ids, target, **kwargs):
        self.created_scopes.append(list(job_ids))
        self.runs.append({"id": "created-run", "target": target})
        return "created-run"


class Searches:
    def __init__(self):
        self.read_ids = []

    def start(self):
        pass

    def list_searches(self):
        return [{"id": "saved-evidence", "job_id": "video-1", "job_name": "sample", "target": "find object", "status": "completed"}]

    def get_search(self, identifier):
        self.read_ids.append(identifier)
        assert identifier == "saved-evidence"
        return {**self.list_searches()[0], "updated_at": "unchanged", "results": [], "frames": [], "video": {"preview_path": None}}


class AppSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library, self.searches = Library(), Searches()
        self.engine = SimpleNamespace(root=self.root, start=lambda: None)
        self.reviews = SimpleNamespace(apply=lambda value: value)
        self.patches = [patch.object(app, "build_benchmark_tab", lambda *_: None),
                        patch.object(app, "build_temporal_tab", lambda *_, **__: None),
                        patch.object(app, "list_vision_models", lambda: ["fixture-model"]),
                        patch.object(app, "TEMPORAL_ARTIFACT_ROOT", self.root / "qwen"),
                        patch.object(app, "TEMPORAL_MINICPM_ROOT", self.root / "mini")]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.demo = app.build_app(self.engine, self.searches, self.library, self.reviews)

    def functions(self, event, name=None):
        return [fn for fn in self.demo.fns.values() if any(target[1] == event for target in fn.targets)
                and (name is None or fn.fn.__name__ == name)]

    async def call(self, fn, inputs, state):
        return await self.demo.process_api(fn, inputs, state=state)

    async def test_lost_session_with_stale_browser_ids_cannot_fail_timer_choice_validation(self):
        # The original failure: a prior session updated choices; a new session
        # has only constructor choices, while an existing browser keeps its ID.
        dynamic = [component for component in self.demo.blocks.values()
                   if isinstance(component, (gr.Dropdown, gr.CheckboxGroup))
                   and component.label in {"已保存的跨视频任务", "视频检索记录（含历史单视频任务）", "本次搜索范围 · 勾选已准备的视频"}]
        for component in dynamic:
            component.choices = []
        picker = next(c for c in dynamic if c.label == "已保存的跨视频任务")
        with self.assertRaises(gr.Error):
            picker.preprocess("stale-browser-run")
        fresh = SessionState(self.demo)
        ticks = self.functions("tick")
        self.assertEqual(len(ticks), 3)
        for fn in ticks:
            self.assertFalse(any(isinstance(c, (gr.Dropdown, gr.CheckboxGroup)) for c in fn.inputs))
            # Gradio must read State defaults, never the browser's stale value.
            values = ["stale-browser-id" if c.stateful else c.value for c in fn.inputs]
            await self.call(fn, values, fresh)
        self.assertEqual(self.library.read_ids, [])
        self.assertEqual(self.searches.read_ids, [])

    async def test_load_atomically_restores_widget_choices_and_canonical_selection(self):
        state = SessionState(self.demo)
        for name, identifier in (("run_options", "saved-run"), ("evidence_options", "saved-evidence")):
            loader = self.functions("load", name)[0]
            self.assertIsInstance(loader.outputs[0], gr.Dropdown)
            self.assertIsInstance(loader.outputs[1], gr.State)
            result = await self.call(loader, [], state)
            self.assertEqual(state[loader.outputs[1]._id], identifier)
            self.assertIn(loader.outputs[1]._id, result["changed_state_ids"])
            self.assertEqual(state.blocks_config.blocks[loader.outputs[0]._id].preprocess(identifier), identifier)
        for name in ("run_view", "evidence_view"):
            tick = self.functions("tick", name)[0]
            await self.call(tick, [None if c.stateful else c.value for c in tick.inputs], state)
        self.assertEqual(self.library.read_ids, ["saved-run"])
        self.assertEqual(self.searches.read_ids, ["saved-evidence"])

    async def test_user_selection_is_validated_and_changes_only_its_session_state(self):
        self.library.runs.append({"id": "new-run", "target": "other"})
        state = SessionState(self.demo)
        loader = self.functions("load", "run_options")[0]
        await self.call(loader, [], state)
        picker, selection = loader.outputs
        choose = next(fn for fn in self.functions("input") if fn.inputs == [picker])
        await self.call(choose, ["new-run"], state)
        self.assertEqual(state[selection._id], "new-run")
        self.assertIsNone(SessionState(self.demo)[selection._id])
        with self.assertRaises(gr.Error):
            await self.call(choose, ["not-a-listed-run"], state)
        self.assertEqual(state[selection._id], "new-run")

    async def test_backend_removed_choice_clears_widget_and_state_together(self):
        state = SessionState(self.demo)
        loader = self.functions("load", "run_options")[0]
        await self.call(loader, [], state)
        picker, selection = loader.outputs
        choose = next(fn for fn in self.functions("input") if fn.inputs == [picker])
        self.library.runs = []
        result = await self.call(choose, ["saved-run"], state)
        self.assertIsNone(state[selection._id])
        self.assertIsNone(result["data"][0]["value"])
        self.assertEqual(state.blocks_config.blocks[picker._id].choices, [])

    async def test_late_library_poll_cannot_overwrite_new_user_selection(self):
        state = SessionState(self.demo)
        tick = self.functions("tick", "poll_library")[0]
        chooser = next(fn for fn in self.functions("input") if isinstance(fn.inputs[0], gr.CheckboxGroup))
        entered, release = threading.Event(), threading.Event()
        original = self.library.list_videos
        blocked = False

        def delayed(*args, **kwargs):
            nonlocal blocked
            if not blocked:
                blocked = True
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test poll was not released")
            return original(*args, **kwargs)

        self.library.list_videos = delayed
        polling = asyncio.create_task(self.call(tick, ["", False, None], state))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            await self.call(chooser, [["video-1"], "", False], state)
            self.assertEqual(state[chooser.outputs[1]._id], ["video-1"])
        finally:
            release.set()
        await polling
        self.assertEqual(state[chooser.outputs[1]._id], ["video-1"])
        self.assertFalse(any(isinstance(c, (gr.CheckboxGroup, gr.State)) and c._id == chooser.outputs[1]._id for c in tick.outputs))

    async def test_immediate_submit_uses_current_checkbox_value_not_pending_state_mirror(self):
        state = SessionState(self.demo)
        await self.call(self.functions("load", "model_options")[0], [], state)
        chooser = next(fn for fn in self.functions("input") if isinstance(fn.inputs[0], gr.CheckboxGroup))
        self.assertEqual(state[chooser.outputs[1]._id], [])
        submit = next(fn for fn in self.demo.fns.values() if fn.api_name == "create_library_search")
        save = next(fn for fn in self.demo.fns.values() if fn.api_name == "save_evidence_review")
        self.assertIsInstance(submit.inputs[0], gr.CheckboxGroup)
        self.assertIsInstance(save.inputs[0], gr.Dropdown)
        await self.call(submit, [["video-1"], "new target", "", "fixture-model", 48, 2, 2, 1, 12, 448, 160], state)
        self.assertEqual(self.library.created_scopes, [["video-1"]])
        self.assertEqual(state[submit.outputs[1]._id], "created-run")

    def test_temporal_default_prefers_existing_minicpm_until_qwen_is_exported(self):
        temporal_tabs = [block for block in self.demo.blocks.values() if isinstance(block, gr.Tabs) and block.selected == "minicpm"]
        self.assertEqual(len(temporal_tabs), 1)
        self.assertEqual({block.id for block in self.demo.blocks.values() if isinstance(block, gr.Tab) and block.id in {"qwen", "minicpm"}}, {"qwen", "minicpm"})
        (self.root / "qwen").mkdir()
        (self.root / "qwen" / "evaluation.json").write_text("{}", encoding="utf-8")
        another = app.build_app(self.engine, self.searches, self.library, self.reviews)
        self.assertEqual(sum(isinstance(block, gr.Tabs) and block.selected == "qwen" for block in another.blocks.values()), 1)


if __name__ == "__main__":
    unittest.main()
