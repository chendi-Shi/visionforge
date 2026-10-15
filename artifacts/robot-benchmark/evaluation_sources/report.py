"""Portable, escaped experiment report containing observed results and failures."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import shutil

from robotbench.data import model_input, sha256_file


NAMES = {"train_majority": "训练集多数类", "text_only": "仅子任务文本 · CLIP", "after": "结束状态 · CLIP + 线性分类器",
         "transition": "前后变化 · CLIP + 线性分类器", "after_only": "Qwen · 仅结束状态", "ordered": "Qwen · 前后状态",
         "reversed": "Qwen · 前后反转（损坏对照）"}


def esc(value):
    return html.escape(str(value), quote=True)


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def _table(metrics):
    rows = []
    for name, m in metrics.items():
        interval = m["bootstrap"]["metrics"]["accuracy"]
        ci = "—" if interval["lower"] is None else f"{percent(interval['lower'])}–{percent(interval['upper'])}"
        rows.append(f"<tr><td>{esc(NAMES.get(name, name))}</td><td>{m['correct']}/{m['n']}</td><td>{percent(m['accuracy'])}</td>"
                    f"<td>{percent(m['balanced_accuracy'])}</td><td>{percent(m['macro_f1'])}</td><td>{percent(m['per_class']['failure']['recall'])}</td><td>{percent(m['coverage'])}</td><td>{ci}</td></tr>")
    return "<div class='scroll'><table><thead><tr><th>方法</th><th>正确 / 全部</th><th>Accuracy</th><th>Balanced acc.</th><th>Macro F1</th><th>失败召回</th><th>回答覆盖率</th><th>Accuracy 95% 区间</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _failure_table(metrics):
    rows = []
    before, after = (metrics[method]["per_failure_category"] for method in ("after", "transition"))
    for category in sorted(set(before) | set(after)):
        values = [before.get(category), after.get(category)]
        cells = ["—" if value is None else f"{value['detected']}/{value['n']} · {percent(value['failure_recall'])}" for value in values]
        rows.append(f"<tr><td>{esc(category)}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>")
    return "<div class='scroll'><table><thead><tr><th>作者标注的失败类型</th><th>结束状态检出</th><th>前后变化检出</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def _counterfactual_table(diagnostics):
    rows = []
    for method, result in diagnostics.items():
        n = result["eligible_pairs"]
        rows.append(f"<tr><td>{esc(NAMES.get(method, method))}</td><td>{result['both_correct']}/{n}</td>"
                    f"<td>{result['same_binary_prediction']}/{n}</td><td>{percent(result['pair_coverage'])}</td></tr>")
    return "<div class='scroll'><table><thead><tr><th>方法</th><th>配对两条均正确</th><th>给出相同二元判定</th><th>配对回答覆盖率</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def write_report(report, samples, root, output, *, prediction_records=None):
    output, root = Path(output), Path(root)
    manifest = report["dataset"]
    blocks = [f"<p class='eyebrow'>VISIONFORGE / ROBOT ACTION-EFFECT BENCHMARK</p><h1>机器人做完动作，<br>目标状态真的改变了吗？</h1>"
              f"<p class='lead'>固定真实机器人数据、按基础任务组验证的轻量学习实验、本地 VLM 消融。展示全部结果和错误。</p>"
              f"<div class='cards'><div><b>{manifest['samples']}</b>标注样本</div><div><b>{manifest['task_groups']}</b>基础任务组</div>"
              f"<div><b>{manifest['episode_groups']}</b>发布方 episode 组</div><div><b>{manifest['success']} / {manifest['failure']}</b>成功 / 失败</div></div>"
              f"<p class='note'>数据来自 <a href='{esc(manifest['source'])}'>Guardian UR5-Fail</a>，revision <code>{esc(manifest['revision'][:12])}</code>。"
              "包含反事实指令和修改后的状态对；样本行不能当作相互独立的机器人执行次数。视角以发布顺序编号，不假设统一物理方向。</p>"]
    csv_rows = []
    if report.get("cv"):
        cv = report["cv"]
        blocks.append("<h2>01 / 轻量学习：按基础任务组验证</h2><p>每次留出一个完整基础任务组，剩余组拟合 StandardScaler 和逻辑回归；训练行若与测试共享任意原图，也会剔除。分组取发布方 taskvar 的 + 前缀；相近语义动作仍可能跨组。CLIP 编码器冻结，超参数固定。以下是本项目的探索性交叉验证，不是官方零样本排行榜，也不保证语义完全不同的新任务泛化。</p>")
        blocks.append(_table(cv["metrics"]))
        pair = cv["paired_after_to_transition"]
        delta = pair["bootstrap_delta"]["metrics"]["accuracy"]
        blocks.append(f"<p class='note'>前后变化相对结束状态的 Accuracy 差值：{percent(pair['delta']['accuracy'])}；基础任务组配对 bootstrap 95% 区间 "
                      f"{percent(delta['lower'])}–{percent(delta['upper'])}。区间跨越零时，这个实验不足以证明稳定提升。"
                      "前后方案同时增加特征维度、包含线性冗余差分并改变 L2 正则化几何；分数变化不能单独归因于时序推理。"
                      "区间对固定逐折预测重采样，未覆盖重新训练的方差。</p>")
        for method, m in cv["metrics"].items():
            csv_rows.append({"experiment": "task_family_cv", "method": method, "failure_recall": m["per_class"]["failure"]["recall"], **{k: m[k] for k in ("n", "correct", "accuracy", "balanced_accuracy", "macro_f1", "coverage")}})
        blocks.append("<h3>哪些失败仍然漏判？</h3><p>下面仅按作者的失败类别分解二元检出率，不代表模型解释对了失败原因。类别样本很少时不能外推。</p>" + _failure_table(cv["metrics"]))
        if cv.get("counterfactual_pairs"):
            blocks.append("<h3>相同画面，不同指令</h3><p>对完全相同的有序六图、不同子任务、相反标签，要求配对两条都答对。所有配对保留，可能复用样本和观测，不是相互独立的试验。</p>" + _counterfactual_table(cv["counterfactual_pairs"]))
    vlm = report.get("vlm")
    if vlm:
        blocks.append(f"<h2>02 / 本地 VLM：同一组样本的输入消融</h2><p>每个基础任务组按固定哈希选一个样本，选择时不看标签或模型成绩。"
                      f"已保存 {vlm['completed_calls']} / {vlm['expected_calls']} 次调用，结构有效 {vlm['successful_calls']} 次。"
                      f"模型调用累计 {vlm['inference_seconds'] / 60:.1f} 分钟（本机负载影响，非标准速度基准）。</p>")
        blocks.append(_table(vlm["metrics"]))
        sensitivity = vlm["order_sensitivity"]
        blocks.append(f"<p class='note'>前后反转属于输入损坏对照：相对原始标签的分数仅用于诊断。"
                      f"结构有效配对 {sensitivity['paired_valid_outputs']} 条，其中 {sensitivity['prediction_flips']} 条改变了三状态判定（含显式弃权）。"
                      "这不能替代连续时序理解的证据。超时、结构错误和弃权都保留在指标分母中。</p>")
        changed = vlm.get("order_sensitivity_changed_inputs")
        if changed:
            blocks.append(f"<p>其中 {len(changed['identical_before_after_ids'])} 条的动作前后图片逐一相同。排除这些输入不变的记录后，"
                          f"结构有效配对 {changed['paired_valid_outputs']} 条，判定翻转 {changed['prediction_flips']} 条；原始分类分母仍不改变。</p>")
        if vlm.get("cv_on_same_pilot"):
            blocks.append("<details><summary>在完全相同的 VLM 子集上查看轻量分类器成绩</summary>" + _table(vlm["cv_on_same_pilot"]) + "</details>")
        for method, m in vlm["metrics"].items():
            csv_rows.append({"experiment": "local_vlm_pilot", "method": method, "failure_recall": m["per_class"]["failure"]["recall"], **{k: m[k] for k in ("n", "correct", "accuracy", "balanced_accuracy", "macro_f1", "coverage")}})
        blocks.append("<h2>03 / 逐条核对：证据、原判与人工标签</h2><p>图片只展示原始状态对。标注来自数据集作者；以下原因是模型回答，不是因果真相。</p>"
                      "<label class='filter'><input type='checkbox' id='errors'> 只看前后状态模型答错／弃权的样本</label>")
        by_id = {s["sample_id"]: s for s in samples}
        assets = output / "assets"
        assets.mkdir(exist_ok=True)
        snapshot = None if prediction_records is None else {condition: {row["sample_id"]: row for row in rows} for condition, rows in prediction_records.items()}
        for sample_id in vlm["plan"]["sample_ids"]:
            sample = by_id[sample_id]
            records = {}
            for condition in vlm["plan"]["conditions"]:
                if snapshot is not None:
                    records[condition] = snapshot.get(condition, {}).get(sample_id, {"prediction": "pending", "status": "missing"})
                else:
                    path = output / "predictions" / f"{sample_id}.{condition}.json"
                    records[condition] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"prediction": "pending", "status": "missing"}
            correct = records["ordered"]["prediction"] == sample["label"] and records["ordered"]["status"] == "ok"
            blocks.append(f"<article class='case' data-correct='{str(correct).lower()}'><div class='case-head'><span>{esc(sample['task_group'])}</span>"
                          f"<span class='tag {'good' if correct else 'bad'}'>前后状态模型：{'正确' if correct else '错误 / 弃权 / 未完成'}</span></div>"
                          f"<h3>{esc(sample['subtask_instruction'])}</h3><p>{esc(sample['task_instruction'])}</p>"
                          f"<p>作者标签：<strong>{esc(sample['label'])}</strong> · 类型：{esc(sample['failure_category'])}</p><div class='observations'>")
            for frame in model_input(sample, root, view=vlm["plan"]["view"])["frames"]:
                filename = frame["sha256"] + ".png"
                dest = assets / filename
                if not dest.exists():
                    shutil.copyfile(frame["path"], dest)
                if sha256_file(dest) != frame["sha256"]:
                    raise ValueError("Report asset checksum mismatch")
                blocks.append(f"<figure><img loading='lazy' src='assets/{filename}' alt='{esc(frame['phase'])} camera {esc(frame['view'])}'>"
                              f"<figcaption>{esc(frame['phase'])} · camera slot {esc(frame['view'])}</figcaption></figure>")
            blocks.append("</div><div class='verdicts'>")
            for condition, record in records.items():
                blocks.append(f"<div><h4>{esc(NAMES[condition])}</h4><b>{esc(record['prediction'])}</b><p>{esc(record.get('evidence') or record.get('error') or '尚未完成')}</p>"
                              f"<small>{esc(record.get('elapsed_seconds', '—'))} s · {esc(record['status'])}</small></div>")
            audit = {"sample_id": sample_id, "records": records}
            blocks.append("</div><details><summary>查看原始响应、输入哈希和模型身份</summary><pre>" + esc(json.dumps(audit, ensure_ascii=False, indent=2)) + "</pre></details></article>")
    blocks.append("<h2>范围与复现</h2><ul>" + "".join(f"<li>{esc(note)}</li>" for note in report["limits"]) + "</ul>"
                  "<p><a href='metrics.json'>完整指标 JSON</a> · <a href='metrics.csv'>指标 CSV</a> · "
                  "<a href='https://github.com/chendi-Shi/visionforge/blob/codex/initial-release/docs/ROBOT_BENCHMARK.md'>实验协议与复现命令</a></p>"
                  "<p class='note'>Data: Guardian authors / UR5-Fail, Apache-2.0 per publisher. See repository THIRD_PARTY.md for attribution. "
                  "Image crops, camera order and state pairs follow the publisher release. No model weights were trained or fine-tuned here beyond the linear probes.</p>")
    css = """
*{box-sizing:border-box}body{margin:0;background:#f3f5f4;color:#14231f;font:15px/1.65 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1200px;margin:auto;padding:54px 28px}h1{font-size:44px;line-height:1.22;letter-spacing:-1px;margin:15px 0 22px}h2{margin-top:52px;font-size:25px;border-top:1px solid #cbd5d0;padding-top:28px}h3{font-size:21px;margin:16px 0 0}h4{margin:0 0 10px;font-size:14px}.eyebrow{letter-spacing:2px;color:#27705d;font-size:12px;font-weight:700}.lead{font-size:18px;max-width:850px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:32px 0}.cards>div{background:white;border:1px solid #d9e2dd;border-radius:10px;padding:20px}.cards b{display:block;font-size:31px;color:#145a46}table{width:100%;border-collapse:collapse;background:white;white-space:nowrap}th,td{padding:14px 13px;border-bottom:1px solid #e0e7e2;text-align:left}th{font-size:12px;color:#527363}.scroll{overflow:auto;border:1px solid #d9e2dd;border-radius:10px}.note{background:#e9efeb;border-left:3px solid #809b88;padding:14px 18px;font-size:13px;color:#405c4c}a{color:#09644c}code{word-break:break-all}details{margin-top:15px}summary{cursor:pointer;color:#1a654e}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto;background:#14231f;color:#e2eee7;padding:18px;font-size:12px}.case{margin:25px 0;background:white;border:1px solid #d9e2dd;border-radius:12px;padding:24px}.case-head{display:flex;justify-content:space-between;gap:16px;font-size:12px;color:#527363}.tag{padding:3px 10px;border-radius:30px}.good{background:#e4f2e9;color:#246b42}.bad{background:#f8e6de;color:#914c2e}.observations{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}figure{margin:0}figure img{width:100%;background:#eee;border-radius:6px}figcaption{font-size:12px;color:#527363}.verdicts{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.verdicts>div{background:#f2f5f3;padding:16px;border-radius:8px}.verdicts p{font-size:13px}.filter{display:block;margin:16px 0}small{color:#527363}@media(max-width:700px){h1{font-size:32px}.cards{grid-template-columns:repeat(2,1fr)}.observations{grid-template-columns:repeat(3,1fr)}.verdicts{grid-template-columns:1fr}main{padding:26px 16px}}
"""
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>VisionForge · 机器人动作效果实验</title><style>" + css + "</style><main>" + "".join(blocks) + "</main><script>document.getElementById('errors')?.addEventListener('change',e=>document.querySelectorAll('.case').forEach(c=>c.hidden=e.target.checked&&c.dataset.correct==='true'));</script></html>"
    (output / "index.html").write_text(page, encoding="utf-8", newline="\n")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "method", "n", "correct", "accuracy", "balanced_accuracy", "macro_f1", "failure_recall", "coverage"))
        writer.writeheader()
        writer.writerows(csv_rows)
