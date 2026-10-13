"""Exercise the running local app with a synthetic Chinese receipt.

Run after starting app.py and installing qwen3-vl:2b. This invokes real inference.
"""
import json
from pathlib import Path
import time

from gradio_client import Client, handle_file


ROOT = Path(__file__).resolve().parents[1]


def main():
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    client = Client(
        "http://127.0.0.1:7860", verbose=False, analytics_enabled=False,
        httpx_kwargs={"trust_env": False, "timeout": 600},
        download_files=ROOT / ".runtime" / "client-downloads",
    )
    client.predict(api_name="/refresh_models")
    started = time.perf_counter()
    status, text, record, files = client.predict(
        handle_file(str(ROOT / "examples" / "receipt_zh.png")),
        "结构化提取", "请抄录店名、商品、单价、数量与合计，并概述这张收据。看不清不要猜。",
        "qwen3-vl:2b", 768, 384, api_name="/run_analysis",
    )
    (artifacts / "structured-receipt.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    downloaded_json = [Path(file) for file in files if Path(file).suffix == ".json"]
    downloaded_text = [Path(file) for file in files if Path(file).suffix == ".txt"]
    json_matches = len(downloaded_json) == 1 and json.loads(downloaded_json[0].read_text(encoding="utf-8")) == record
    text_matches = len(downloaded_text) == 1 and text in downloaded_text[0].read_text(encoding="utf-8")
    evidence = {
        "status": status, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "api_name": "/run_analysis", "sample": "examples/receipt_zh.png",
        "record_status": record.get("status"), "download_count": len(files),
        "json_export_matches_record": json_matches, "text_export_matches_result": text_matches,
        "scope": "Real local app API, image upload, queue, inference and export; no browser rendering check",
    }
    (artifacts / "ui-inference.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(json.dumps(record.get("structured"), ensure_ascii=False, indent=2))
    if record.get("status") != "ok" or not json_matches or not text_matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
