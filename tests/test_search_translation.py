import json
import unittest
from unittest.mock import MagicMock, patch

from search_jobs import QueryTranslationError, translate_query


class TranslationTests(unittest.TestCase):
    def call(self, raw, done='stop'):
        client = MagicMock()
        client.chat.return_value = {'message': {'content': raw}, 'done_reason': done}
        info = {'capabilities': ['thinking'], 'details': {'family': 'qwen3vl'}}
        with patch('core._create_client', return_value=client), patch('core._inspect_model', return_value=('qwen3-vl:2b', 'digest', info)):
            result = translate_query('红色货车什么时候出现？', 'qwen3-vl:2b')
        return result, client.chat.call_args.kwargs

    def test_question_becomes_visible_target_with_auditable_request(self):
        raw = json.dumps({'query': 'a red truck', 'visual_target': '红色货车'}, ensure_ascii=False)
        result, request = self.call(raw)
        self.assertEqual(result['visual_target'], '红色货车')
        self.assertEqual(result['raw_response'], raw)
        self.assertEqual(result['options'], request['options'])
        self.assertGreaterEqual(request['options']['num_predict'], 256)
        self.assertFalse(request['think'])

    def test_truncated_response_is_preserved_but_never_used_as_query(self):
        raw = '{"query":"a red truck", "visual_target":"红'
        with self.assertRaises(QueryTranslationError) as context:
            self.call(raw, 'length')
        self.assertEqual(context.exception.record['raw_response'], raw)
        self.assertEqual(context.exception.record['done_reason'], 'length')
        self.assertNotIn('query', context.exception.record)

    def test_whitespace_target_and_invalid_json_are_rejected(self):
        for raw in ('not json', '{"query":"a red truck","visual_target":"   "}'):
            with self.subTest(raw=raw), self.assertRaises(QueryTranslationError) as context:
                self.call(raw)
            self.assertEqual(context.exception.record['raw_response'], raw)


if __name__ == '__main__':
    unittest.main()
