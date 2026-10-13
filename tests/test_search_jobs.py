"""Persistence and evidence integrity with deterministic non-model fixtures."""
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_jobs import SearchEngine, refinement_indices, QueryTranslationError
from video_media import _sha256


class SearchJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.preview = self.root / 'preview.mp4'
        self.preview.write_bytes(b'fixed-preview')
        job = {'name':'test.mp4','video': {'preview_path':str(self.preview),
            'preview_sha256':_sha256(self.preview), 'frame_count':120,'duration_seconds':10}}
        videos = types.SimpleNamespace(root=self.root, get_job=lambda _: job)
        self.engine = SearchEngine(videos)
        self.identity = {'model':'fake:local','digest':'fixed-model','code':'fixed-code'}
        self.calls = []
        self.check = lambda frame: 'ok'
        self.release = threading.Event()
        self.enter = threading.Event()
        self.block = False
        def verify(frame, *a, **kw):
            self.calls.append(frame['id'])
            self.enter.set()
            if self.block:
                self.release.wait(60)
            return {'frame':frame,'status':self.check(frame),'digest':self.identity['digest'],
                    'structured':{'verdict':'visible','reason':'fixture'}}
        index = types.ModuleType('video_index')
        index.release_models = lambda: None
        index.build_index = lambda *a, **kw: {'frame_count':6,'max_gap_seconds':2}
        index.search_index = lambda *a, **kw: [{'id':'hit','frame_index':48,'time_seconds':4,'score':.7}]
        grounding = types.ModuleType('core_grounding')
        grounding.verify_frame = verify
        grounding.derive_intervals = lambda rows, **kw: {'verified_count':len(rows)}
        def extract(preview, folder, indices):
            folder.mkdir(parents=True,exist_ok=True)
            result=[]
            for index in indices:
                path=folder/f'{index}.png'
                path.write_bytes(f'frame-{index}'.encode())
                result.append({'id':f'f{index}','frame_index':index,'time_seconds':index/12,
                               'path':str(path),'sha256':_sha256(path)})
            return result
        self.patches = [patch.dict(sys.modules, {'video_index':index,'core_grounding':grounding}),
                        patch('search_jobs.release_search_model'),
                        patch('search_jobs.model_identity', side_effect=lambda _:dict(self.identity)),
                        patch('search_jobs.extract_evidence',side_effect=extract)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        self.release.set()
        self.engine.stop()
        for item in self.patches[::-1]:
            item.stop()
        self.temp.cleanup()

    def create(self, **kwargs):
        return self.engine.create_search('video','红色物体','red object',model='fake:local',top_k=1,**kwargs)

    def wait(self, sid, statuses):
        # Shared CPU-only hosts can be descheduled for seconds during model
        # work. This bounds hangs without treating scheduling as correctness.
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            result=self.engine.get_search(sid)
            if result['status'] in statuses:
                return result
            time.sleep(.01)
        self.fail(str(self.engine.get_search(sid)))

    def test_pause_resume_reuses_successful_frames(self):
        self.block=True
        sid=self.create()
        self.engine.start()
        self.assertTrue(self.enter.wait(30))
        self.engine.pause(sid)
        self.release.set()
        paused=self.wait(sid,{'paused'})
        self.assertEqual(len(paused['results']),1)
        first=self.calls[0]
        self.engine.resume(sid)
        result=self.wait(sid,{'completed','failed'})
        self.assertEqual(result['status'],'completed',result.get('error'))
        self.assertEqual(self.calls.count(first),1)
        self.assertEqual(len(result['results']),5)
        self.assertEqual(SearchEngine(self.engine.videos).get_search(sid)['results'],result['results'])

    def test_resume_rejects_changed_model(self):
        sid=self.create()
        self.check=lambda _: 'failed'
        self.engine.start()
        result=self.wait(sid,{'partial'})
        prior=len(self.calls)
        self.identity['digest']='different'
        self.engine.resume(sid)
        result=self.wait(sid,{'failed'})
        self.assertIn('改变',result['error'])
        self.assertEqual(len(self.calls),prior)
        self.assertEqual(len(result['results']),5)

    def test_retry_only_failed_frames(self):
        sid=self.create()
        self.check=lambda f:'failed' if f['frame_index']==48 else 'ok'
        self.engine.start()
        self.wait(sid,{'partial'})
        self.check=lambda _:'ok'
        self.engine.resume(sid)
        self.wait(sid,{'completed'})
        self.assertEqual(len(self.calls),6)
        self.assertEqual(self.calls.count('f48'),2)

    def test_tampered_successful_evidence_blocks_resume(self):
        self.block=True
        sid=self.create()
        self.engine.start()
        self.assertTrue(self.enter.wait(30))
        self.engine.pause(sid)
        self.release.set()
        result=self.wait(sid,{'paused'})
        Path(result['frames'][0]['path']).write_bytes(b'tampered')
        self.engine.resume(sid)
        failed=self.wait(sid,{'failed'})
        self.assertIn('证据已改变',failed['error'])
        self.assertEqual(len(self.calls),1)

    def test_one_worker_owner_and_queued_pause(self):
        sid=self.create()
        self.engine.pause(sid)
        self.assertTrue(self.engine.start())
        other=SearchEngine(self.engine.videos)
        self.assertFalse(other.start())
        self.assertEqual(self.engine.get_search(sid)['status'],'paused')
        self.assertEqual(self.calls,[])

    def test_validate_budgets_and_query_language(self):
        for args in ({'max_frames':True},{'step_seconds':float('nan')},{'top_k':0}):
            with self.assertRaises(ValueError):
                self.engine.create_search('video','target','red car',**args)
        with self.assertRaises(ValueError):
            self.engine.create_search('video','target','红色车辆')

    def test_translation_failure_persists_raw_output_without_running_verification(self):
        sid = self.engine.create_search('video', '红色货车什么时候出现', model='fake:local')
        record = {'raw_response':'truncated response', 'done_reason':'length'}
        with patch('search_jobs.translate_query', side_effect=QueryTranslationError('truncated', record)):
            self.engine.start()
            result = self.wait(sid, {'failed'})
        self.assertEqual(result['translation_failure'], record)
        self.assertIsNone(result['translation'])
        self.assertEqual(self.calls, [])

    def test_refinement_spends_budget_across_hits_and_clamps(self):
        values=refinement_indices([{'frame_index':0},{'frame_index':119}],120,2,1,6)
        self.assertEqual(len(values),6)
        self.assertTrue({0,119}.issubset(values))
        self.assertTrue(all(0<=n<120 for n in values))
        self.assertEqual(values,sorted(set(values)))


if __name__=='__main__':
    unittest.main()
