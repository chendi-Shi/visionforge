"""Offline temporal evidence report, portable export, and stdlib verification.

Only report/export load the prepared data or a local transcoder. Verification
imports no model, image, numeric or HDF5 library and never executes snapshots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from robotbench import temporal_evaluation as evaluation


SNAPSHOTS = {"manifest.json": "dataset_manifest_sha256", "plan.json": "dataset_selection_sha256", "labels.jsonl": "labels_sha256"}
PREDICTION_SOURCES = {"temporal.py": "source_sha256", "temporal_data.py": "adapter_source_sha256", "core.py": "client_source_sha256"}
METHODS = ("uniform", "state_event")
INPUT_EVIDENCE_FIELDS = {"sample_id", "task_id", "robot", "split", "source_sha256", "task_description", "substages",
                         "timestamp", "state_eef", "head_view", "fps", "duration_seconds", "head_video_sha256"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def reject_local_paths(value):
    if isinstance(value, str) and re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|file://|/Users/|/home/", value, re.I):
        raise ValueError("Public metadata contains a local absolute path")
    if isinstance(value, dict):
        for child in value.values():
            reject_local_paths(child)
    elif isinstance(value, list):
        for child in value:
            reject_local_paths(child)


def safe(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative or ".." in Path(relative).parts:
        raise ValueError("Unsafe relative artifact path")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Artifact path escaped its root")
    return target


def save(path, content, *, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if path.read_bytes() != content:
            raise ValueError("Existing export contains different frozen bytes: " + path.name)
        return
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def load_frozen_labels(run):
    plan = read(run / "plan.json")
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256"):
        raise ValueError("Run plan checksum mismatch")
    for name, key in SNAPSHOTS.items():
        if sha(run / "dataset_snapshot" / name) != plan[key]:
            raise ValueError("Frozen dataset snapshot checksum mismatch")
    for name, key in PREDICTION_SOURCES.items():
        if sha(run / "prediction_sources" / name) != plan["protocol"][key]:
            raise ValueError("Prediction source snapshot differs from the frozen protocol")
    labels = rows(run / "dataset_snapshot" / "labels.jsonl")
    if len({r["sample_id"] for r in labels}) != len(labels):
        raise ValueError("Duplicate frozen annotation IDs")
    selection = read(run / "dataset_snapshot" / "plan.json")
    selected = {row["sample_id"]: row for row in selection["samples"]}
    # Independently reconstruct annotations from the separately frozen source
    # selection. No annotation is passed into the predictor or sampler here.
    for label in labels:
        original = selected[label["sample_id"]]
        annotation = original["annotation"]
        expected = {"sample_id": original["sample_id"], "task_id": original["task_id"], "split": original["split"],
                    "label": "failure" if annotation["verdict"] == "fail" else "success",
                    "onset_seconds": annotation.get("onset"), "failure_types": annotation.get("type", [])}
        if label != expected:
            raise ValueError("Frozen labels differ from the source author annotations")
    labels = [r for r in labels if r["split"] == plan["split"]]
    if set(METHODS) != set(plan["protocol"]["methods"]):
        raise ValueError("Unexpected frozen sampler methods")
    if {r["sample_id"] for r in labels} != {r["sample_id"] for r in plan["inputs"]}:
        raise ValueError("Frozen annotation IDs do not exactly cover the run")
    return plan, labels


def portable_inputs(inputs):
    result = []
    for row in inputs:
        result.append({k: row[k] for k in ("sample_id", "task_id", "robot", "split", "source_sha256", "task_description", "substages", "timestamp", "state_eef", "head_view", "fps", "duration_seconds")})
        result[-1]["head_video_sha256"] = row["video_sha256"][row["head_view"]]
    return result


def validate_inputs(plan, inputs):
    expected = {r["sample_id"]: r for r in plan["inputs"]}
    if len({r["sample_id"] for r in inputs}) != len(inputs) or {r["sample_id"] for r in inputs} != set(expected):
        raise ValueError("Input evidence IDs do not exactly match the frozen run")
    evaluation.episode_summaries(inputs)
    for row in inputs:
        identifier = row["sample_id"]
        if set(row) != INPUT_EVIDENCE_FIELDS:
            raise ValueError("Input evidence contains unexpected or annotation-bearing fields")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Unsafe sample ID")
        if not evaluation._finite(row["fps"]) or row["fps"] <= 0:
            raise ValueError("Valid encoded video fps required for navigation")
        if (not isinstance(row["state_eef"], list) or len(row["state_eef"]) != len(row["timestamp"])
                or any(not isinstance(state, list) or len(state) != 16 or any(not evaluation._finite(x) for x in state) for state in row["state_eef"])):
            raise ValueError("Invalid numeric state evidence")
        identity = {k: row[k] for k in ("sample_id", "task_id", "robot", "source_sha256", "split")}
        model_input = {k: row[k] for k in ("task_description", "substages", "timestamp", "state_eef", "head_view", "head_video_sha256")}
        identity["prediction_input_sha256"] = digest(model_input)
        if identity != expected[identifier]:
            raise ValueError("Prediction input evidence differs from its frozen identity")


def validate_record(record, episode, plan, root):
    identifier, method = episode["sample_id"], record.get("method")
    if method not in METHODS or any(record.get(k) != episode[k] for k in ("sample_id", "task_id", "robot", "split", "source_sha256")):
        raise ValueError("Prediction identity differs from frozen episode")
    if (record.get("plan_sha256") != plan["plan_sha256"] or record.get("model_digest") != plan["model_digest"]
            or record.get("model") != plan["model"] or record.get("options") != plan["protocol"]["model_options"]
            or record.get("system") != plan["protocol"]["system"] or record.get("schema") != plan["protocol"]["schema"]):
        raise ValueError("Prediction model or protocol differs from the plan")
    times = evaluation._selection_times(record, episode["timestamp"])
    selection = record.get("selection") or {}
    if (times is None or len(times) != plan["protocol"]["frame_budget"] or selection.get("method") != method
            or record.get("duration_seconds") != episode["timestamp"][-1]):
        raise ValueError("Sampling evidence differs from measured timestamps or frame budget")
    scores = selection.get("scores")
    if not isinstance(scores, list) or len(scores) != len(episode["timestamp"]) or any(not evaluation._finite(s) or s < 0 for s in scores):
        raise ValueError("Whole-recording state scores are missing or invalid")
    if json.loads(record["question"]) != {"task": episode["task_description"], "ordered_steps": episode["substages"], "frame_times_seconds": times}:
        raise ValueError("Prompt contains different task instructions or measured times")
    panel = f"panels/{identifier}--{method}.jpg"
    if record.get("panel_file") != panel or sha(safe(root, panel)) != record.get("panel_sha256"):
        raise ValueError("Original model input panel checksum differs")
    if record.get("status") == "ok":
        if record.get("done_reason") in {"length", "max_tokens", "max_length"}:
            raise ValueError("Truncated model output cannot be marked valid")
        raw = json.loads(record["raw_response"])
        if plan["protocol"].get("version") == 2:
            if not isinstance(raw, dict) or set(raw) != {"decision", "reason"} or not isinstance(raw["decision"], str):
                raise ValueError("Unexpected version 2 raw verdict schema")
            decision = raw["decision"]
            if decision in {"success", "abstain"}:
                verdict, frame = decision, -1
            elif re.fullmatch(r"failure_at_frame_[0-9]{2}", decision):
                verdict, frame = "failure", int(decision.rsplit("_", 1)[1])
            else:
                raise ValueError("Invalid joint verdict decision")
            raw = {"decision": decision, "prediction": verdict, "frame_id": frame, "reason": raw["reason"]}
        elif plan["protocol"].get("version") == 1:
            if not isinstance(raw, dict) or set(raw) != {"prediction", "frame_id", "reason"}:
                raise ValueError("Unexpected version 1 raw verdict schema")
        else:
            raise ValueError("Unsupported prediction protocol version")
        verdict, frame = raw["prediction"], raw["frame_id"]
        if (verdict not in {"success", "failure", "abstain"} or type(frame) is not int
                or not isinstance(raw["reason"], str) or not raw["reason"].strip()
                or (verdict == "failure" and not 0 <= frame < len(times)) or (verdict != "failure" and frame != -1)):
            raise ValueError("Invalid raw verdict")
        expected = {**raw, "onset_seconds": float(times[frame]) if verdict == "failure" else None}
        if any(record.get(k) != value for k, value in expected.items()):
            raise ValueError("Stored verdict no longer matches the raw model response")
    elif record.get("status") != "failed" or record.get("prediction") != "abstain" or record.get("onset_seconds") is not None or record.get("frame_id") != -1:
        raise ValueError("Failed inference must remain an abstention")


def load_records(run, inputs, plan, *, complete=False):
    by_id = {r["sample_id"]: r for r in inputs}
    expected = {(i, m) for i in by_id for m in METHODS}
    records, seen = [], set()
    for path in sorted((run / "predictions").glob("*.json")):
        record = read(path)
        key = record.get("sample_id"), record.get("method")
        if key not in expected or key in seen or path.name != f"{key[0]}--{key[1]}.json":
            raise ValueError("Duplicate, unknown or misnamed prediction record")
        validate_record(record, by_id[key[0]], plan, run)
        records.append(record)
        seen.add(key)
    if complete and seen != expected:
        raise ValueError("Public export requires every episode × sampler record, including failed runs")
    return records


def sample_rows(inputs, labels, media):
    indexed = {row["sample_id"]: row for row in labels}
    return [{"sample_id": row["sample_id"], "task_id": row["task_id"], "task_description": row["task_description"],
             "substages": row["substages"], "robot": row["robot"], "split": row["split"], "label": indexed[row["sample_id"]]["label"],
             "onset_seconds": indexed[row["sample_id"]]["onset_seconds"], "failure_types": indexed[row["sample_id"]]["failure_types"],
             "video_file": media[row["sample_id"]], "video_fps": row["fps"], "frame_count": len(row["timestamp"]),
             "duration_seconds": row["duration_seconds"], "measured_end_seconds": row["timestamp"][-1],
             "head_view": row["head_view"], "source_sha256": row["source_sha256"]} for row in inputs]


def render_html(plan, samples, records, result):
    esc = lambda value: html.escape(str(value), quote=True)
    percentage = lambda value: "—" if value is None else f"{100 * value:.1f}%"
    seconds = lambda value: "—" if value is None else f"{value:.2f}s"
    names = {"uniform": "均匀抽帧", "state_event": "状态事件抽帧", "clock_midpoint": "全判失败 · 时钟中点", "state_peak": "全判失败 · 状态峰值"}
    table = []
    for method, metrics in result["methods"].items():
        values = [names[method], percentage(metrics["accuracy"]), percentage(metrics["failure_recall"]), percentage(metrics["success_false_alarm_rate"]),
                  percentage(metrics["hit_at_1s"]), percentage(metrics["hit_at_2s"]), percentage(metrics["hit_at_5s"]),
                  seconds(metrics["conditional_onset_mae_seconds"]), percentage(metrics["valid_onset_coverage"]), percentage(metrics["abstention_rate"])]
        table.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>")
    by_key = {(r["sample_id"], r["method"]): r for r in records}
    cards = []
    for number, sample in enumerate(samples):
        identifier = sample["sample_id"]
        ground = "成功" if sample["label"] == "success" else "失败 · 作者起点 " + seconds(sample["onset_seconds"])
        panels = []
        for method in METHODS:
            record = by_key.get((identifier, method))
            if not record:
                panels.append(f'<section><h4>{esc(names[method])}</h4><p class="bad">缺失输出，仍计入分母。</p></section>')
                continue
            times = record["selection"]["timestamps"]
            tiles = []
            for index, measured in enumerate(times):
                playback = record["selection"]["indices"][index] / sample["video_fps"]
                tiles.append(f'<button class="tile" data-video="v-{esc(identifier)}" data-seek="{playback:.9f}" title="实测 {measured:.3f}s；视频 {playback:.3f}s"><span>{index:02d} · {measured:.2f}s</span></button>')
            verdict = esc(record["prediction"]) + " · " + seconds(record.get("onset_seconds"))
            panels.append(f'<section><h4>{esc(names[method])} <small>{verdict}</small></h4><div class="panel"><img src="{esc(record["panel_file"])}" alt="原始模型输入面板"><div class="tiles" style="grid-template-rows:repeat({math.ceil(len(times)/4)},1fr)">{"".join(tiles)}</div></div><p>{esc(record.get("reason") or record.get("error") or "")}</p><a href="predictions/{esc(identifier)}--{esc(method)}.json">原始响应与时间记录</a><details><summary>查看未经改写的输出</summary><pre>{esc(record.get("raw_response", ""))}</pre></details></section>')
        gt_button = "" if sample["onset_seconds"] is None else f'<button data-video="v-{esc(identifier)}" data-seek="{sample["onset_seconds"]:.9f}">跳到作者标注 {sample["onset_seconds"]:.2f}s</button>'
        cards.append(f'<details class="episode" {"open" if number == 0 else ""}><summary><strong>{number+1:02d} · {esc(sample["task_id"])}</strong><span>{esc(ground)}</span></summary><p>{esc(sample["task_description"])}</p><p class="muted">{esc(sample["robot"])} · {esc(identifier)} · {sample["duration_seconds"]:.2f}s · {esc(sample["head_view"])}</p><video id="v-{esc(identifier)}" src="{esc(sample["video_file"])}" controls preload="metadata"></video><div class="actions">{gt_button}<a href="{esc(sample["video_file"])}">完整头部视角视频</a></div><div class="comparisons">{"".join(panels)}</div></details>')
    paired = result["paired"]["uniform_to_state_event"]
    delta = paired["delta"]["hit_at_2s"]
    ci = paired["bootstrap_delta"]["metrics"]["hit_at_2s"]
    difference = "—" if delta is None else f"{delta*100:+.1f} 个百分点"
    interval = "未定义" if ci["lower"] is None else f'[{ci["lower"]*100:+.1f}, {ci["upper"]*100:+.1f}] 个百分点'
    body = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionForge · 时序失败证据</title><style>
body{{margin:0;background:#f2f4f7;color:#182637;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1440px;margin:auto;padding:32px}}h1{{font-size:34px;margin:8px 0}}h2{{font-size:22px}}.eyebrow{{color:#47718f;letter-spacing:.08em}}.muted,small{{color:#607085}}.intro,.metrics,.episode{{background:white;border:1px solid #dbe1e8;border-radius:12px;padding:22px;margin:18px 0}}.chips{{display:flex;gap:14px;flex-wrap:wrap}}.chips span{{background:#edf2f8;padding:8px 14px;border-radius:7px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #e5e9ef}}th:first-child,td:first-child{{text-align:left}}th{{font-weight:600;background:#f6f8fa}}a{{color:#1b6399}}summary{{cursor:pointer}}.episode>summary{{display:flex;justify-content:space-between;gap:18px}}.episode video{{display:block;max-width:800px;width:100%;max-height:520px;background:#111;margin:16px auto;border-radius:8px}}.actions{{display:flex;gap:14px;align-items:center;justify-content:center}}button{{cursor:pointer;border:1px solid #b8cadb;background:#f3f7fb;color:#164d77;border-radius:5px;padding:7px 12px}}.comparisons{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:22px}}h4{{display:flex;justify-content:space-between;margin:6px 0 10px}}.panel{{position:relative;line-height:0}}.panel img{{width:100%;height:auto;display:block}}.tiles{{position:absolute;inset:0;display:grid;grid-template-columns:repeat(4,1fr)}}.tile{{background:transparent;border:1px solid transparent;border-radius:0;position:relative;padding:0}}.tile:hover,.tile:focus{{background:#44b7ff24;border:2px solid #1f91da}}.tile span{{position:absolute;bottom:2px;left:2px;background:#122333dd;color:white;font:10px system-ui;padding:2px 4px;border-radius:3px}}pre{{white-space:pre-wrap;word-break:break-word;background:#f4f6f8;padding:12px}}.bad{{color:#a43030}}footer{{color:#607085;padding:24px 0}}@media(max-width:850px){{main{{padding:15px}}.comparisons{{grid-template-columns:1fr}}.episode>summary{{display:block}}h1{{font-size:26px}}}}
</style></head><body><main><div class="eyebrow">VISIONFORGE / TEMPORAL EVIDENCE</div><h1>失败发生在哪里？</h1><p class="muted">{esc(plan["model"])} · {esc(plan["split"])} · 相同视角、每段 {plan["protocol"]["frame_budget"]} 帧的离线回顾对照</p><section class="intro"><div class="chips"><span>{len(samples)} 段完整记录</span><span>{len({s["task_id"] for s in samples})} 个任务簇</span><span>采样器与标注隔离</span></div><p>点击输入面板里的任一小图，在对应完整视频中检查证据。面板文件保留模型实际看到的原始字节；公开视频只为审阅压缩。面板标记实测时间，播放器按原视频帧位置定位。</p><p>这是离线失败检测与起点定位实验，使用整段录制信息；不代表在线预警或机器人控制能力。</p><a href="plan.json">冻结协议</a> · <a href="evaluation.json">全部指标与逐样本结果</a> · <a href="samples.json">样本索引</a></section><section class="metrics"><h2>全分母结果</h2><div class="scroll"><table><thead><tr><th>方法</th><th>准确率</th><th>失败召回</th><th>成功误报</th><th>Hit@1s</th><th>Hit@2s</th><th>Hit@5s</th><th>条件 MAE</th><th>有效定位覆盖</th><th>弃权率</th></tr></thead><tbody>{"".join(table)}</tbody></table></div><p>状态事件相对均匀抽帧的 Hit@2s 差值：<strong>{difference}</strong>；任务簇配对区间 {interval}。小样本区间仅作描述。</p><p class="muted">Hit 分母是全部真实失败，包括漏检与弃权。MAE 只针对已检出且时间有效的失败，必须与覆盖率一起看。两个控制不读取标签，均默认全判失败；不能把失败召回单独解释为能力。采样器最近帧覆盖仅反映时间分辨率，详见原始指标。</p></section><h2>逐段证据</h2>{"".join(cards)}<footer>模型 digest：{esc(plan["model_digest"])}<br>计划 SHA256：{esc(plan["plan_sha256"])}<br>数据：FailTime / CC-BY-4.0。保留作者标注与模型原始输出；两者可能存在可见性或起点定义差异。</footer></main><script>
document.addEventListener('click', function(event){{const button=event.target.closest('button[data-video]');if(!button)return;const video=document.getElementById(button.dataset.video);const target=Number(button.dataset.seek);if(!video||!Number.isFinite(target))return;video.dataset.pendingSeek=String(target);const seek=()=>{{const value=Number(video.dataset.pendingSeek);video.currentTime=Math.max(0,Math.min(value,Number.isFinite(video.duration)?video.duration:value));video.pause();}};if(video.readyState>=1)seek();else video.addEventListener('loadedmetadata',seek,{{once:true}});video.scrollIntoView({{behavior:'smooth',block:'center'}});}});
</script></body></html>'''
    return body


def local_report(dataset, run):
    from robotbench.temporal_data import load_inputs
    plan, labels = load_frozen_labels(run)
    actual = load_inputs(dataset, split=plan["split"])
    inputs = portable_inputs(actual)
    validate_inputs(plan, inputs)
    records = load_records(run, inputs, plan)
    result = evaluation.evaluate_all(labels, records, inputs)
    media = {r["sample_id"]: Path(r["videos"][r["head_view"]]).as_uri() for r in actual}
    samples = sample_rows(inputs, labels, media)
    save(run / "evaluation.json", json_bytes(result))
    save(run / "samples.json", json_bytes(samples))
    save(run / "report.html", render_html(plan, samples, records, result).encode())
    return result


def _transcode(source, destination, expected_sha):
    if sha(source) != expected_sha:
        raise ValueError("Source video changed before review transcode")
    receipt_path = destination.with_suffix(".receipt.json")
    if destination.exists() and receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["source_sha256"] == expected_sha and receipt["review_sha256"] == sha(destination):
            return receipt
        raise ValueError("Review media receipt mismatch")
    import imageio_ffmpeg
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.stem + ".tmp.mp4")
    options = ["-map", "0:v:0", "-an", "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
               "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart"]
    subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source), *options, "-y", str(temporary)], check=True, timeout=300)
    temporary.replace(destination)
    receipt = {"source_sha256": expected_sha, "review_sha256": sha(destination), "review_bytes": destination.stat().st_size,
               "transform": options, "purpose": "Review-only preview. Original model input panel bytes remain unchanged."}
    save(receipt_path, json_bytes(receipt), frozen=True)
    return receipt


def export_bundle(dataset, run, destination):
    from robotbench.temporal_data import load_inputs
    plan, labels = load_frozen_labels(run)
    manifest = read(run / "dataset_snapshot" / "manifest.json")
    if plan["split"] != "test" or len(plan["inputs"]) != 16 or not manifest.get("complete") or manifest.get("samples_prepared") != 19:
        raise ValueError("Public test export requires the complete 19-episode preparation and fixed 16-test run")
    actual = load_inputs(dataset, split="test")
    inputs = portable_inputs(actual)
    validate_inputs(plan, inputs)
    records = load_records(run, inputs, plan, complete=True)
    for value in (plan, inputs, records, read(run / "dataset_snapshot" / "plan.json")):
        reject_local_paths(value)
    result = evaluation.evaluate_all(labels, records, inputs)
    if destination.exists():
        verified = verify_bundle(destination)
        if verified["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("Destination belongs to another run")
        return verified
    stage = destination.with_name("." + destination.name + ".staging")
    stage.mkdir(parents=True, exist_ok=True)
    save(stage / "plan.json", (run / "plan.json").read_bytes(), frozen=True)
    if (run / "model-manifest.json").is_file():
        if sha(run / "model-manifest.json") != plan["model_digest"].removeprefix("sha256:"):
            raise ValueError("Local model manifest bytes do not match the frozen full model digest")
        save(stage / "model-manifest.json", (run / "model-manifest.json").read_bytes(), frozen=True)
    for name in SNAPSHOTS:
        save(stage / "dataset_snapshot" / name, (run / "dataset_snapshot" / name).read_bytes(), frozen=True)
    for name in PREDICTION_SOURCES:
        save(stage / "prediction_sources" / name, (run / "prediction_sources" / name).read_bytes(), frozen=True)
    for record in records:
        stem = f"{record['sample_id']}--{record['method']}"
        save(stage / "predictions" / (stem + ".json"), (run / "predictions" / (stem + ".json")).read_bytes(), frozen=True)
        save(stage / "panels" / (stem + ".jpg"), (run / "panels" / (stem + ".jpg")).read_bytes(), frozen=True)
    review_media = {}
    for row in actual:
        identifier = row["sample_id"]
        print(json.dumps({"stage": "review_video", "sample_id": identifier}), flush=True)
        review_media[identifier] = _transcode(Path(row["videos"][row["head_view"]]), stage / "videos" / (identifier + ".mp4"), row["video_sha256"][row["head_view"]])
    samples = sample_rows(inputs, labels, {row["sample_id"]: "videos/" + row["sample_id"] + ".mp4" for row in inputs})
    for name, content in (("input-evidence.json", inputs), ("samples.json", samples), ("evaluation.json", result), ("media-manifest.json", review_media)):
        save(stage / name, json_bytes(content), frozen=True)
    save(stage / "report.html", render_html(plan, samples, records, result).encode(), frozen=True)
    code = {"temporal_report.py": Path(__file__), "temporal_evaluation.py": Path(evaluation.__file__)}
    for name, source in code.items():
        save(stage / "verifier" / "robotbench" / name, source.read_bytes(), frozen=True)
    save(stage / "verifier" / "robotbench" / "__init__.py", b'"""Explicit portable verification implementation."""\n', frozen=True)
    entry = '''"""Read-only stdlib verifier. Review/trust this tool before executing it."""
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(root/'verifier'))
sys.argv=[str(root/'verify.py'),'verify','--bundle',str(root)]
runpy.run_module('robotbench.temporal_report',run_name='__main__')
'''
    save(stage / "verify.py", entry.encode(), frozen=True)
    attribution = '''FailTime data (selected episodes and unchanged model input panels)
Dataset: ChangUoA/FailTime
Paper: ProTracer: Proprioception-Guided Failure Diagnosis in Robot Manipulation
Authors: Chang Dong; Mehdi Hosseinzadeh; King Hang Wong; Lingqiao Liu;
Francois Fraysse; Feras Dayoub; Minh Hoai Nguyen (2026).
Revision: 07aea9142616cf1f8047512bd53a4d9fcd5e7cee
Fixed source: https://huggingface.co/datasets/ChangUoA/FailTime/tree/07aea9142616cf1f8047512bd53a4d9fcd5e7cee
Dataset card/citation: https://huggingface.co/datasets/ChangUoA/FailTime/blob/07aea9142616cf1f8047512bd53a4d9fcd5e7cee/README.md
Author code: https://github.com/Chang-AIML/ProTracer
License: Creative Commons Attribution 4.0 International (CC-BY-4.0)
License terms: https://creativecommons.org/licenses/by/4.0/legalcode
Original source identities and annotations are retained in
dataset_snapshot/plan.json. Video previews have
been resized to width 480 and encoded with H.264 CRF 30 for review; model input
panels and raw prediction records are copied unchanged. No endorsement implied.
'''
    save(stage / "DATA_ATTRIBUTION.txt", attribution.encode(), frozen=True)
    notes = '''VisionForge temporal failure evidence bundle

Open report.html for offline review. Panel cells seek the corresponding video
frame. Videos are compressed review media; panels are exact model input bytes.
Measured timestamps are used for every prediction and metric. Video playback
coordinates use source frame index / encoded video fps only for navigation.

Run: python -B verify.py
Repository equivalent: python -B -m robotbench.temporal_report verify --bundle PATH

The verifier requires only Python's standard library. It checks every manifest
file, frozen source/annotation snapshots, all 16 x 2 raw records and input-panel
hashes, reconstructs author labels and model verdicts, and recomputes the full
evaluation and both label-free controls. It does not execute prediction_sources,
refit anything, decode HDF5, call Ollama, or access the network. input-evidence.json
contains the full numeric model input evidence needed to recompute its frozen
identity. Original large HDF5 and original MP4 files are not duplicated; their
hashes and upstream source references are retained. Verification is consistency
checking, not independent rerunning of inference or authentication of authorship.
It recomputes controls and metrics from the saved frame selections/state scores;
it does not rerun Gaussian smoothing, peak detection or the sampling algorithm.

verifier/ contains explicitly shipped executable verification code. Inspect and
trust it before execution. The repository command uses already installed code
and requires version-matched evaluator/report SHA values. Prediction snapshots
are preserved as data and never imported. A checksum is not a signature.

All selected records remain in denominators. Missing or failed predictions
abstain. Conditional MAE is meaningful only with valid-onset coverage. Sampler
oracle coverage measures time resolution, not visible semantic failure evidence.
The clock and motion-peak controls never see labels and always predict failure.
Task-cluster intervals on this small stratified subset are descriptive. This is
offline retrospective diagnosis, not online failure detection or robot control.
'''
    save(stage / "README.txt", notes.encode(), frozen=True)
    payload = {p.relative_to(stage).as_posix(): sha(p) for p in stage.rglob("*") if p.is_file() and p.name not in {"bundle-manifest.json", "bundle-manifest.sha256"}}
    bundle_manifest = {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "files": dict(sorted(payload.items())),
                       "verifier_source_sha256": {name: sha(path) for name, path in code.items()},
                       "independent_of_previous_bundles": True, "excluded_envelope_files": ["bundle-manifest.json", "bundle-manifest.sha256"]}
    save(stage / "bundle-manifest.json", json_bytes(bundle_manifest), frozen=True)
    save(stage / "bundle-manifest.sha256", (sha(stage / "bundle-manifest.json") + "\n").encode(), frozen=True)
    verified = verify_bundle(stage)
    stage.replace(destination)
    return verified


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = read(bundle / "bundle-manifest.json")
    if (bundle / "bundle-manifest.sha256").read_text().strip() != sha(bundle / "bundle-manifest.json"):
        raise ValueError("Bundle manifest checksum mismatch")
    expected = set(manifest["files"]) | {"bundle-manifest.json", "bundle-manifest.sha256"}
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if expected != actual:
        raise ValueError("Bundle file inventory differs from manifest")
    for name, checksum in manifest["files"].items():
        if sha(safe(bundle, name)) != checksum:
            raise ValueError("Bundle payload checksum mismatch: " + name)
    for name, path in (("temporal_report.py", Path(__file__)), ("temporal_evaluation.py", Path(evaluation.__file__))):
        if sha(path) != manifest["verifier_source_sha256"][name]:
            raise ValueError("Trusted verifier version differs from the exported evaluator")
    plan, labels = load_frozen_labels(bundle)
    if (bundle / "model-manifest.json").is_file() and sha(bundle / "model-manifest.json") != plan["model_digest"].removeprefix("sha256:"):
        raise ValueError("Local model manifest differs from the frozen full digest")
    if manifest["plan_sha256"] != plan["plan_sha256"] or plan["split"] != "test" or len(plan["inputs"]) != 16:
        raise ValueError("Bundle is not the complete frozen 16-episode test run")
    dataset_manifest = read(bundle / "dataset_snapshot" / "manifest.json")
    if not dataset_manifest.get("complete") or dataset_manifest.get("samples_prepared") != 19:
        raise ValueError("Frozen dataset preparation was incomplete")
    inputs = read(bundle / "input-evidence.json")
    validate_inputs(plan, inputs)
    records = load_records(bundle, inputs, plan, complete=True)
    for value in (plan, inputs, records, read(bundle / "dataset_snapshot" / "plan.json")):
        reject_local_paths(value)
    media = read(bundle / "media-manifest.json")
    if set(media) != {row["sample_id"] for row in inputs}:
        raise ValueError("Review video inventory does not match selected inputs")
    for row in inputs:
        identifier = row["sample_id"]
        receipt = read(bundle / "videos" / (identifier + ".receipt.json"))
        if (receipt != media[identifier] or receipt["source_sha256"] != row["head_video_sha256"]
                or receipt["review_sha256"] != sha(bundle / "videos" / (identifier + ".mp4"))):
            raise ValueError("Review video provenance mismatch")
    samples = sample_rows(inputs, labels, {row["sample_id"]: "videos/" + row["sample_id"] + ".mp4" for row in inputs})
    if samples != read(bundle / "samples.json"):
        raise ValueError("Published sample labels or media links differ from frozen inputs")
    result = evaluation.evaluate_all(labels, records, inputs)
    if result != read(bundle / "evaluation.json"):
        raise ValueError("Recomputed metrics or label-free controls differ")
    if render_html(plan, samples, records, result) != (bundle / "report.html").read_text(encoding="utf-8"):
        raise ValueError("HTML report differs from the validated evidence")
    return {"verified": True, "plan_sha256": plan["plan_sha256"], "samples": len(inputs), "prediction_records": len(records),
            "payload_files": len(manifest["files"]), "inference_calls": 0, "network_calls": 0,
            "scope": "Frozen sources, full numeric input identities, exact panels/raw records, complete denominator metrics and controls; no inference rerun."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("report", "export"):
        child = sub.add_parser(command)
        child.add_argument("--data", type=Path, required=True)
        child.add_argument("--run", type=Path, required=True)
        if command == "export":
            child.add_argument("--out", type=Path, default=Path("artifacts/robot-temporal"))
    child = sub.add_parser("verify")
    child.add_argument("--bundle", type=Path, default=Path("artifacts/robot-temporal"))
    args = parser.parse_args()
    if args.command == "report":
        result = local_report(args.data, args.run)
        print(json.dumps({"report": str(args.run / "report.html"), "n": result["methods"]["uniform"]["n"]}))
    elif args.command == "export":
        print(json.dumps(export_bundle(args.data, args.run, args.out)))
    else:
        print(json.dumps(verify_bundle(args.bundle)))


if __name__ == "__main__":
    main()
