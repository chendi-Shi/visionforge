"""Offline unit checks for validation, data contracts and local-only routing."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import core


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private-image.png"
        Image.new("RGB", (1200, 600), "white").save(self.path)
        self.client = MagicMock()
        self.client.list.return_value = {
            "models": [{"model": core.DEFAULT_MODEL, "digest": "sha256:test"}]
        }
        self.client.show.return_value = {"capabilities": ["completion", "vision"]}
        self.client.chat.return_value = {
            "message": {"content": "图片中是白色背景。"},
            "done_reason": "stop", "eval_count": 12, "total_duration": 123456,
        }
        self.factory = patch.object(core, "_create_client", return_value=self.client)
        self.factory.start()
        self.addCleanup(self.factory.stop)

    def test_qa_resizes_without_leaking_file_path_and_preserves_metrics(self):
        result = core.analyze_image(self.path, "图片是什么？")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["image"]["original_size"], [1200, 600])
        self.assertEqual(result["image"]["processed_size"], [768, 384])
        self.assertEqual(result["metrics"]["eval_count"], 12)
        self.assertEqual(result["digest"], "sha256:test")
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(str(self.path), serialized)
        self.assertNotIn(self.path.name, serialized)
        kwargs = self.client.chat.call_args.kwargs
        self.assertFalse(kwargs["think"])
        self.assertEqual(kwargs["keep_alive"], "2m")
        self.assertEqual(kwargs["options"]["num_thread"], 4)
        self.assertEqual(kwargs["options"]["num_ctx"], 4096)
        with Image.open(io.BytesIO(kwargs["messages"][1]["images"][0])) as sent:
            self.assertEqual(sent.size, (768, 384))
            self.assertEqual(sent.mode, "RGB")

    def test_small_images_are_not_upscaled_and_exif_is_applied(self):
        small = Path(self.temp.name) / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (80, 40), "red").save(small, exif=exif)
        result = core.analyze_image(small, "颜色？")
        self.assertEqual(result["image"]["original_size"], [80, 40])
        self.assertEqual(result["image"]["processed_size"], [40, 80])

    def test_structured_validates_and_rejects_invalid_types(self):
        data = {"summary": "白色背景", "visible_text": [], "objects": [], "uncertainties": []}
        self.client.chat.return_value["message"]["content"] = json.dumps(data)
        result = core.analyze_image(self.path, "提取内容", mode="extract")
        self.assertEqual(result["structured"], data)
        self.assertEqual(result["status"], "ok")
        self.assertIn("properties", self.client.chat.call_args.kwargs["format"])
        for bad_output in ('{"summary":"missing fields"}', "not json", json.dumps({**data, "objects": [123]})):
            with self.subTest(output=bad_output):
                self.client.chat.return_value["message"]["content"] = bad_output
                result = core.analyze_image(self.path, "提取内容", mode="extract")
                self.assertEqual(result["status"], "invalid_output")
                self.assertIsNone(result["structured"])
                self.assertEqual(result["raw_response"], bad_output)

    def test_transparent_png_is_composited_on_white_before_inference(self):
        transparent = Path(self.temp.name) / "transparent.png"
        picture = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
        picture.putpixel((1, 0), (0, 0, 0, 255))
        picture.save(transparent)
        core.analyze_image(transparent, "内容？")
        sent_bytes = self.client.chat.call_args.kwargs["messages"][1]["images"][0]
        with Image.open(io.BytesIO(sent_bytes)) as sent:
            self.assertEqual(sent.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(sent.getpixel((1, 0)), (0, 0, 0))

    def test_truncation_is_never_counted_as_success(self):
        self.client.chat.return_value["done_reason"] = "length"
        for mode in ("qa", "extract"):
            with self.subTest(mode=mode):
                result = core.analyze_image(self.path, "内容？", mode=mode, max_tokens=32)
                self.assertEqual(result["status"], "truncated")
                self.assertTrue(result["error"])

    def test_rejects_cloud_remote_nonvision_and_uninstalled_models(self):
        for name, installed_info, show_info in (
            ("qwen3-vl:cloud", {}, {"capabilities": ["vision"]}),
            ("remote:2b", {}, {"capabilities": ["vision"], "remote_host": "https://example.com"}),
            ("remote:2b", {"remote_model": "other"}, {"capabilities": ["vision"]}),
            ("text:2b", {}, {"capabilities": ["completion"]}),
        ):
            with self.subTest(model=name):
                self.client.list.return_value = {"models": [{"model": name, **installed_info}]}
                self.client.show.return_value = show_info
                with self.assertRaises(ValueError):
                    core.analyze_image(self.path, "内容？", model=name)
        with self.assertRaises(ValueError):
            core.analyze_image(self.path, "内容？", model="missing:2b")
        self.client.chat.assert_not_called()

    def test_model_listing_filters_remote_and_nonvision(self):
        self.client.list.return_value = {"models": [
            {"model": "vision:2b"}, {"model": "text:2b"},
            {"model": "vision:cloud"}, {"model": "remote:2b"},
        ]}
        self.client.show.side_effect = lambda model: {
            "vision:2b": {"capabilities": ["vision"]},
            "text:2b": {"capabilities": ["completion"]},
            "remote:2b": {"capabilities": ["vision"], "remote_model": "cloud-vision"},
        }[model]
        self.assertEqual(core.list_vision_models(), ["vision:2b"])

    def test_qwen3vl_thinking_parser_gets_closed_thinking_prefill(self):
        self.client.show.return_value = {
            "capabilities": ["completion", "vision", "thinking"],
            "details": {"family": "qwen3vl"},
        }
        result = core.analyze_image(self.path, "内容？")
        messages = self.client.chat.call_args.kwargs["messages"]
        self.assertEqual(messages[-1], {"role": "assistant", "content": "<think>\n\n</think>\n\n"})
        self.assertEqual(result["assistant_prefill"], messages[-1]["content"])
        self.client.show.return_value["details"]["family"] = "other"
        result = core.analyze_image(self.path, "内容？")
        self.assertEqual(len(self.client.chat.call_args.kwargs["messages"]), 2)
        self.assertIsNone(result["assistant_prefill"])

    def test_invalid_inputs_rejected_before_inference(self):
        bad_file = Path(self.temp.name) / "not-image.png"
        bad_file.write_text("not an image")
        for image_path, question, options in (
            (None, "内容？", {}), (self.path, "  ", {}),
            (bad_file, "内容？", {}), (self.path, "内容？", {"max_edge": 1537}),
            (self.path, "内容？", {"max_tokens": 0}),
            (self.path, "内容？", {"mode": "unknown"}),
        ):
            with self.subTest(image=image_path, options=options):
                with self.assertRaises(ValueError):
                    core.analyze_image(image_path, question, **options)
        self.client.chat.assert_not_called()

    def test_image_byte_and_pixel_limits(self):
        with patch.object(core, "MAX_IMAGE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "20 MB"):
                core.analyze_image(self.path, "内容？")
        with patch.object(core, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaisesRegex(ValueError, "2500"):
                core.analyze_image(self.path, "内容？")
        self.client.chat.assert_not_called()

    def test_connection_failure_is_readable(self):
        self.client.list.side_effect = ConnectionError("internal detail")
        with self.assertRaisesRegex(ConnectionError, "本机 Ollama"):
            core.analyze_image(self.path, "内容？")
        with self.assertRaisesRegex(ConnectionError, "本机 Ollama"):
            core.list_vision_models()

    def test_client_is_pinned_to_loopback_and_ignores_proxy_environment(self):
        self.factory.stop()
        with patch.object(core.ollama, "Client") as constructor:
            core._create_client()
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["host"], "http://127.0.0.1:11434")
        self.assertFalse(kwargs["trust_env"])
        self.assertFalse(kwargs["follow_redirects"])
        self.assertEqual(kwargs["timeout"].connect, 5)


if __name__ == "__main__":
    unittest.main()
