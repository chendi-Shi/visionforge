"""CLI for one real local image inference."""
import argparse
import json
from pathlib import Path

from core import DEFAULT_MODEL, analyze_image, list_vision_models


def main():
    parser = argparse.ArgumentParser(description="本地 Ollama 图片理解")
    parser.add_argument("image", nargs="?")
    parser.add_argument("--question", default="请用中文描述图中清晰可见的内容。")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["qa", "extract"], default="qa")
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    try:
        if args.list_models:
            print(json.dumps(list_vision_models(), ensure_ascii=False))
            return
        if not args.image:
            parser.error("需要图片路径，或者使用 --list-models")
        result = analyze_image(args.image, args.question, args.model, args.mode, args.max_edge, args.max_tokens)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload)
        if result["status"] != "ok":
            raise SystemExit(2)
    except (RuntimeError, ConnectionError, ValueError, OSError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
