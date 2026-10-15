"""Fixed two-model temporal study comparison; no inference or model selection.

This module reads and verifies two complete standalone evidence bundles. It
uses the already-frozen standard-library evaluator on the same selected IDs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import html
from pathlib import Path
import json

from robotbench import temporal_evaluation as evaluation
from robotbench import temporal_report as report

MINI = "minicpm-v4.6:1b"
QWEN = "qwen3.5:4b"


def _model_parts(manifest):
    def component(value):
        return {key: value[key] for key in ("mediaType", "digest", "size")}
    return {"schemaVersion": manifest["schemaVersion"], "mediaType": manifest["mediaType"],
            "config": component(manifest["config"]), "layers": [component(row) for row in manifest["layers"]]}


def validate_study(study, bundles):
    """All checks precede comparison scoring; the two bundles were verified."""
    if report.digest({k: v for k, v in study.items() if k != "study_plan_sha256"}) != study.get("study_plan_sha256"):
        raise ValueError("Frozen study plan checksum mismatch")
    if set(bundles) != {MINI, QWEN} or study.get("expected_test_calls") != 64:
        raise ValueError("The fixed study requires both named models and 64 calls")
    ids = study["test_sample_ids"]
    if len(ids) != 16 or len(set(ids)) != 16 or set(ids) & set(study["development_sample_ids"]):
        raise ValueError("Study test membership is invalid or overlaps development")
    if study["uncertainty"].get("seed") != 1337 or study["uncertainty"].get("iterations") != 2000:
        raise ValueError("Unexpected fixed study uncertainty protocol")
    expected_models = {row["name"]: row for row in study["models"]}
    if set(expected_models) != {MINI, QWEN} or len(expected_models) != len(study["models"]):
        raise ValueError("Unexpected or repeated study model identity")
    trusted_eval = report.sha(Path(evaluation.__file__))
    if study["evaluation_source_sha256"] != trusted_eval:
        raise ValueError("Trusted evaluator differs from the study-frozen source")
    prepared = {}
    for model, bundle in bundles.items():
        plan, labels = report.load_frozen_labels(bundle)
        manifest = report.read(bundle / "bundle-manifest.json")
        if (plan["model"] != model or plan["protocol"] != study["protocol"]
                or manifest["verifier_source_sha256"]["temporal_evaluation.py"] != trusted_eval):
            raise ValueError("Bundle model, protocol or evaluator differs from the study")
        if set(r["sample_id"] for r in plan["inputs"]) != set(ids) or len(plan["inputs"]) != 16:
            raise ValueError("Bundle is not the exact frozen test cohort")
        if datetime.fromisoformat(plan["created_at"]) < datetime.fromisoformat(study["created_at"]):
            raise ValueError("Model run was frozen before the fixed two-model study")
        selection = report.read(bundle / "dataset_snapshot" / "plan.json")
        if (selection.get("plan_sha256") != study["selection_plan_sha256"]
                or report.digest({k: v for k, v in selection.items() if k != "plan_sha256"}) != selection["plan_sha256"]):
            raise ValueError("Dataset selection differs from the study")
        inputs = report.read(bundle / "input-evidence.json")
        records = report.load_records(bundle, inputs, plan, complete=True)
        prepared[model] = {"plan": plan, "labels": labels, "inputs": inputs, "records": records,
                           "evaluation": report.read(bundle / "evaluation.json"), "manifest": manifest}
    a, b = prepared[MINI], prepared[QWEN]
    if a["labels"] != b["labels"] or a["inputs"] != b["inputs"] or a["plan"]["inputs"] != b["plan"]["inputs"]:
        raise ValueError("Models were not evaluated on identical labels and numeric/pixel input identities")
    if a["plan"]["server"] != b["plan"]["server"]:
        raise ValueError("Ollama server versions differ in this same-server study")
    for name in report.SNAPSHOTS:
        if report.sha(bundles[MINI] / "dataset_snapshot" / name) != report.sha(bundles[QWEN] / "dataset_snapshot" / name):
            raise ValueError("Frozen dataset snapshots differ between models")
    if a["plan"]["model_digest"].removeprefix("sha256:") != expected_models[MINI]["digest"].removeprefix("sha256:"):
        raise ValueError("MiniCPM digest differs from its predeclared identity")
    official = expected_models[QWEN]["registry_manifest"]
    if report.digest(official) != expected_models[QWEN]["manifest_canonical_sha256"]:
        raise ValueError("Pinned Qwen registry manifest checksum mismatch")
    local_path = bundles[QWEN] / "model-manifest.json"
    if (not local_path.is_file() or report.sha(local_path) != b["plan"]["model_digest"].removeprefix("sha256:")
            or _model_parts(report.read(local_path)) != _model_parts(official)):
        raise ValueError("Qwen local full manifest digest/config/layers do not match the predeclared registry model")
    a_records = {(r["sample_id"], r["method"]): r for r in a["records"]}
    b_records = {(r["sample_id"], r["method"]): r for r in b["records"]}
    if set(a_records) != set(b_records) or len(a_records) != 32:
        raise ValueError("The fixed study must include all 16 x 2 records per model")
    for key, left in a_records.items():
        right = b_records[key]
        if left["selection"] != right["selection"] or left["panel_sha256"] != right["panel_sha256"] or left["question"] != right["question"]:
            raise ValueError("Models received different frame selections, panel bytes or task prompts")
    if a["evaluation"]["control_predictions"] != b["evaluation"]["control_predictions"]:
        raise ValueError("Label-free controls differ despite the same input episode/sampler set")
    return prepared


def comparison(study_path, minicpm_bundle, qwen_bundle):
    bundles = {MINI: Path(minicpm_bundle), QWEN: Path(qwen_bundle)}
    # Use trusted installed verifier code. Never import files from either bundle.
    verified = {name: report.verify_bundle(path) for name, path in bundles.items()}
    study = report.read(study_path)
    prepared = validate_study(study, bundles)
    labels, inputs = prepared[MINI]["labels"], prepared[MINI]["inputs"]
    paired = {}
    for method in report.METHODS:
        left = [r for r in prepared[MINI]["records"] if r["method"] == method]
        right = [r for r in prepared[QWEN]["records"] if r["method"] == method]
        paired[method] = evaluation.compare_paired(labels, left, right, inputs, bootstrap_iterations=2000, seed=1337)
    return {"schema_version": 1, "study_plan_sha256": study["study_plan_sha256"], "study_plan_file_sha256": report.sha(study_path),
            "comparison_source_sha256": report.sha(Path(__file__)), "evaluation_source_sha256": study["evaluation_source_sha256"],
            "n_samples": 16, "prediction_calls": 64, "test_sample_ids": study["test_sample_ids"],
            "models": {name: {"model_digest": prepared[name]["plan"]["model_digest"], "model_plan_sha256": prepared[name]["plan"]["plan_sha256"],
                               "bundle_directory": path.name, "bundle_manifest_sha256": report.sha(path / "bundle-manifest.json"), "verified": verified[name]}
                       for name, path in bundles.items()},
            "four_model_sampler_conditions": [{"model": model, "sampler": method, "metrics": prepared[model]["evaluation"]["methods"][method]}
                                               for model in (MINI, QWEN) for method in report.METHODS],
            "label_free_controls": {m: prepared[MINI]["evaluation"]["methods"][m] for m in evaluation.CONTROL_METHODS},
            "paired_minicpm_to_qwen": paired,
            "paired_uniform_to_state_event": {m: prepared[m]["evaluation"]["paired"]["uniform_to_state_event"] for m in (MINI, QWEN)},
            "limits": [*study["sampling_limits"], "Both models use the same frozen prompt, 12-frame budget, exact panel bytes and sampler selections.",
                       "Model identity differs; this is not an isolated parameter-count, architecture or training-data experiment.",
                       "No tuning, inference or new selection is performed by this comparison.",
                       "Task-cluster intervals are descriptive on this small stratified subset. Missing/failed outputs remain abstentions in all paired denominators.",
                       "Ground-truth onsets are author annotations; displayed visual evidence and annotated failure onset can differ."]}


def render_html(result):
    esc = lambda value: html.escape(str(value), quote=True)
    pct = lambda v: "—" if v is None else f"{v*100:.1f}%"
    sec = lambda v: "—" if v is None else f"{v:.2f}s"
    condition_rows = []
    for condition in result["four_model_sampler_conditions"]:
        m = condition["metrics"]
        values = [condition["model"], condition["sampler"], pct(m["accuracy"]), pct(m["failure_recall"]), pct(m["success_false_alarm_rate"]),
                  pct(m["hit_at_1s"]), pct(m["hit_at_2s"]), pct(m["hit_at_5s"]), sec(m["conditional_onset_mae_seconds"]), pct(m["valid_onset_coverage"]), pct(m["abstention_rate"])]
        condition_rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>")
    deltas = []
    for method, value in result["paired_minicpm_to_qwen"].items():
        delta, ci = value["delta"]["hit_at_2s"], value["bootstrap_delta"]["metrics"]["hit_at_2s"]
        effect = "—" if delta is None else f"{delta*100:+.1f} pp"
        interval = "未定义" if ci["lower"] is None else f'[{ci["lower"]*100:+.1f}, {ci["upper"]*100:+.1f}] pp'
        deltas.append(f'<li>{esc(method)}：Qwen − MiniCPM Hit@2s <strong>{esc(effect)}</strong>，任务簇配对区间 {esc(interval)}</li>')
    links = " · ".join(f'<a href="../{esc(v["bundle_directory"])}/report.html">{esc(k)} 完整视频证据</a>' for k, v in result["models"].items())
    limits = "".join(f"<li>{esc(v)}</li>" for v in result["limits"])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionForge · 固定双模型时序对照</title><style>body{{font:15px/1.65 system-ui;margin:0;background:#f3f5f8;color:#1b2939}}main{{max-width:1400px;margin:auto;padding:32px}}section{{background:white;padding:24px;margin:18px 0;border:1px solid #dce3eb;border-radius:12px}}h1{{font-size:30px}}a{{color:#1e6393}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;width:100%}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #e2e7ee}}th:first-child,td:first-child{{text-align:left}}.muted{{color:#617187}}</style><main><p class="muted">VISIONFORGE / FROZEN TWO-MODEL STUDY</p><h1>同一批视频、同一组面板，两种本地模型</h1><p>16 段测试记录 × 2 种采样器 × 2 个模型；64 次推理全部保留。此页只复核和比较已完成的固定实验。</p><p>{links}</p><section><h2>全部四种模型与采样器组合</h2><div class="scroll"><table><thead><tr><th>模型</th><th>采样器</th><th>准确率</th><th>失败召回</th><th>成功误报</th><th>Hit@1s</th><th>Hit@2s</th><th>Hit@5s</th><th>条件 MAE</th><th>定位覆盖</th><th>弃权率</th></tr></thead><tbody>{"".join(condition_rows)}</tbody></table></div><p>所有 Hit 指标以全部真实失败为分母；条件 MAE 必须和定位覆盖率一起阅读。</p></section><section><h2>固定样本上的配对比较</h2><ul>{"".join(deltas)}</ul><p>区间固定 seed=1337、2000 次任务簇重采样，只作描述，不作部署或显著性结论。</p><a href="comparison.json">完整指标、控制与配对结果</a> · <a href="study-plan.json">首个测试调用前冻结的研究计划</a></section><section><h2>解释边界</h2><ul>{limits}</ul></section><p class="muted">Study SHA256：{esc(result["study_plan_sha256"])}</p></main></html>'''


def export_comparison(study_path, minicpm_bundle, qwen_bundle, destination):
    study_path, destination = Path(study_path), Path(destination)
    bundles = {MINI: Path(minicpm_bundle), QWEN: Path(qwen_bundle)}
    if any(path.parent.resolve() != destination.parent.resolve() for path in bundles.values()):
        raise ValueError("Study report and its model evidence bundles must be sibling directories for portable links")
    result = comparison(study_path, *bundles.values())
    if destination.exists():
        if (destination / "study-plan.json").read_bytes() != study_path.read_bytes():
            raise ValueError("Existing comparison belongs to another frozen study")
        return verify_comparison(destination, *bundles.values())
    stage = destination.with_name("." + destination.name + ".staging")
    stage.mkdir(parents=True, exist_ok=True)
    report.save(stage / "study-plan.json", study_path.read_bytes(), frozen=True)
    report.save(stage / "comparison.json", report.json_bytes(result), frozen=True)
    report.save(stage / "report.html", render_html(result).encode(), frozen=True)
    for model, path in bundles.items():
        prefix = "minicpm" if model == MINI else "qwen"
        report.save(stage / "references" / (prefix + "-bundle-manifest.json"), (path / "bundle-manifest.json").read_bytes(), frozen=True)
        report.save(stage / "references" / (prefix + "-run-plan.json"), (path / "plan.json").read_bytes(), frozen=True)
    report.save(stage / "references" / "qwen-model-manifest.json", (bundles[QWEN] / "model-manifest.json").read_bytes(), frozen=True)
    sources = {"temporal_comparison.py": Path(__file__), "temporal_report.py": Path(report.__file__), "temporal_evaluation.py": Path(evaluation.__file__)}
    for name, path in sources.items():
        report.save(stage / "verifier" / "robotbench" / name, path.read_bytes(), frozen=True)
    report.save(stage / "verifier" / "robotbench" / "__init__.py", b'"""Explicit standard-library comparison verification code."""\n', frozen=True)
    entry = '''"""Review/trust this verifier before executing a downloaded copy."""
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(root/'verifier'))
sys.argv=[str(root/'verify.py'),'verify','--out',str(root),'--minicpm',str(root.parent/'robot-temporal-minicpm'),'--qwen',str(root.parent/'robot-temporal')]
runpy.run_module('robotbench.temporal_comparison',run_name='__main__')
'''
    # Use the actual sibling basenames, keeping the default wrapper portable.
    entry = entry.replace("root.parent/'robot-temporal-minicpm'", "root.parent/" + repr(bundles[MINI].name))
    entry = entry.replace("root.parent/'robot-temporal'", "root.parent/" + repr(bundles[QWEN].name))
    report.save(stage / "verify.py", entry.encode(), frozen=True)
    notes = '''Fixed two-model temporal study comparison

This metadata-only bundle requires both sibling model evidence bundles listed
in comparison.json. Open report.html; inspect raw comparisons and study-plan.json.
Run python -B verify.py to verify both complete model bundles, enforce the study
membership/protocol/model-manifest pins, and recompute all paired comparisons.
No network, inference, HDF5, numeric package or untrusted prediction-source
execution is required. The explicitly shipped verifier code must be reviewed
and trusted before execution. Checksums are consistency checks, not signatures.

Qwen's local manifest bytes must hash to its full frozen model digest and match
the registry config and ordered layer identities pinned before any test call.
MiniCPM's digest is compared directly to the study plan. All model inputs, frame
selections, panels, labels, server version and evaluator source are held fixed.
No methods are chosen or retuned from the comparison. Small-subset task-cluster
intervals are descriptive; conditional MAE must be read with coverage.
'''
    report.save(stage / "README.txt", notes.encode(), frozen=True)
    files = {p.relative_to(stage).as_posix(): report.sha(p) for p in stage.rglob("*") if p.is_file() and p.name not in {"bundle-manifest.json", "bundle-manifest.sha256"}}
    manifest = {"study_plan_sha256": result["study_plan_sha256"], "files": dict(sorted(files.items())),
                "verifier_source_sha256": {name: report.sha(path) for name, path in sources.items()},
                "requires_bundles": {name: {"directory": path.name, "manifest_sha256": report.sha(path / "bundle-manifest.json")} for name, path in bundles.items()}}
    report.save(stage / "bundle-manifest.json", report.json_bytes(manifest), frozen=True)
    report.save(stage / "bundle-manifest.sha256", (report.sha(stage / "bundle-manifest.json") + "\n").encode(), frozen=True)
    checked = verify_comparison(stage, *bundles.values())
    stage.replace(destination)
    return {**checked, "exported": True}


def verify_comparison(destination, minicpm_bundle, qwen_bundle):
    destination = Path(destination)
    manifest = report.read(destination / "bundle-manifest.json")
    if (destination / "bundle-manifest.sha256").read_text().strip() != report.sha(destination / "bundle-manifest.json"):
        raise ValueError("Comparison manifest checksum mismatch")
    if {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} != set(manifest["files"]) | {"bundle-manifest.json", "bundle-manifest.sha256"}:
        raise ValueError("Comparison bundle file inventory mismatch")
    for name, checksum in manifest["files"].items():
        if report.sha(report.safe(destination, name)) != checksum:
            raise ValueError("Comparison payload checksum mismatch")
    for name, path in (("temporal_comparison.py", Path(__file__)), ("temporal_report.py", Path(report.__file__)), ("temporal_evaluation.py", Path(evaluation.__file__))):
        if report.sha(path) != manifest["verifier_source_sha256"][name]:
            raise ValueError("Trusted comparison verifier version differs")
    for model, path in ((MINI, Path(minicpm_bundle)), (QWEN, Path(qwen_bundle))):
        expected = manifest["requires_bundles"][model]
        if path.name != expected["directory"] or report.sha(path / "bundle-manifest.json") != expected["manifest_sha256"]:
            raise ValueError("Required model evidence bundle differs")
        prefix = "minicpm" if model == MINI else "qwen"
        for filename, origin in ((prefix + "-bundle-manifest.json", "bundle-manifest.json"), (prefix + "-run-plan.json", "plan.json")):
            if (destination / "references" / filename).read_bytes() != (path / origin).read_bytes():
                raise ValueError("Copied evidence reference differs from the model bundle")
    if (destination / "references" / "qwen-model-manifest.json").read_bytes() != (Path(qwen_bundle) / "model-manifest.json").read_bytes():
        raise ValueError("Copied Qwen manifest reference differs")
    result = comparison(destination / "study-plan.json", minicpm_bundle, qwen_bundle)
    if result != report.read(destination / "comparison.json") or manifest["study_plan_sha256"] != result["study_plan_sha256"]:
        raise ValueError("Recomputed fixed-study comparison differs")
    if render_html(result) != (destination / "report.html").read_text(encoding="utf-8"):
        raise ValueError("Comparison HTML differs from validated results")
    return {"verified": True, "samples": 16, "model_calls": 64, "inference_calls": 0, "study_plan_sha256": result["study_plan_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify"))
    parser.add_argument("--study", type=Path, default=Path("data/temporal-study/plan.json"))
    parser.add_argument("--minicpm", type=Path, default=Path("artifacts/robot-temporal-minicpm"))
    parser.add_argument("--qwen", type=Path, default=Path("artifacts/robot-temporal"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/robot-temporal-study"))
    args = parser.parse_args()
    if args.command == "export":
        result = export_comparison(args.study, args.minicpm, args.qwen, args.out)
    else:
        result = verify_comparison(args.out, args.minicpm, args.qwen)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
