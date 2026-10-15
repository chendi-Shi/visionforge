"""Read-only viewer for the public robot benchmark export.

Only the caller's fixed artifact directory is readable here. The UI never
loads private benchmark data, alters an experiment plan, or starts inference.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import gradio as gr


CV_METHODS = ("train_majority", "text_only", "after", "transition")
VLM_CONDITIONS = ("after_only", "ordered", "reversed")
NAMES = {
    "train_majority": "训练集多数类", "text_only": "只读子任务 · CLIP",
    "after": "动作后状态 · CLIP", "transition": "前后变化 · CLIP",
    "after_only": "VLM · 仅动作后", "ordered": "VLM · 正确前后顺序",
    "reversed": "VLM · 交换前后（损坏诊断）",
}
HEADERS = ["方法", "正确 / 全部", "Accuracy", "Balanced accuracy", "失败召回", "Macro F1", "覆盖率", "任务宏平均", "Accuracy 95% CI"]


def _file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("公开实验文件路径超出导出目录")
    return path


def _read(root: Path, name: str):
    path = _file(root, name)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("公开实验 JSON 超出读取上限")
    return json.loads(path.read_text(encoding="utf-8"))


def _samples(root: Path) -> list[dict]:
    rows = _read(root, "samples.json")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("samples.json 应为公开样本数组")
    ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", sid) for sid in ids):
        raise ValueError("公开样本 ID 无效")
    if len(ids) != len(set(ids)):
        raise ValueError("公开样本 ID 重复")
    return rows


def _percent(value) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _metric_rows(section: dict | None, methods: tuple[str, ...]) -> list[list[str]]:
    if section is not None and not isinstance(section, dict):
        raise ValueError("公开实验指标结构无效")
    scores = (section or {}).get("metrics") or {}
    rows = []
    for method in methods:
        result = scores.get(method)
        if not result:
            rows.append([NAMES[method], "未导出", *(["—"] * (len(HEADERS) - 2))])
            continue
        interval = (result.get("bootstrap") or {}).get("metrics", {}).get("accuracy", {})
        ci = "—" if interval.get("lower") is None else f"{_percent(interval['lower'])} – {_percent(interval['upper'])}"
        rows.append([NAMES[method], f"{result.get('correct', '—')} / {result.get('n', '—')}",
                     _percent(result.get("accuracy")), _percent(result.get("balanced_accuracy")),
                     _percent((result.get("per_class") or {}).get("failure", {}).get("recall")),
                     *[_percent(result.get(key)) for key in ("macro_f1", "coverage", "task_macro_accuracy")], ci])
    return rows


def load_overview(artifact_root: str | Path):
    """Plain values for UI and offline smoke checks; absent data stays absent."""
    root = Path(artifact_root).resolve()
    try:
        report = _read(root, "metrics.json")
        if not isinstance(report, dict):
            raise ValueError("metrics.json 应为指标对象")
        samples = _samples(root)
        dataset, vlm = report.get("dataset") or {}, report.get("vlm") or {}
        cv_rows = _metric_rows(report.get("cv"), CV_METHODS)
        vlm_rows = _metric_rows(vlm, VLM_CONDITIONS)
        status = f"数据：{dataset.get('samples', '—')} 条标注记录 · {dataset.get('task_groups', '—')} 个基础任务组。"
        if vlm:
            done, total = vlm.get("completed_calls", 0), vlm.get("expected_calls", "—")
            status += f"\n\n本地 VLM 已保存 **{done} / {total}** 次调用，结构有效 {vlm.get('successful_calls', '—')} 次。"
            status += f"公开诊断样本 {len(samples)} 条；全量 CV 与 VLM 子集的分母不同。"
            if isinstance(total, int) and done < total:
                status += " **实验尚未全部完成；未回答样本保留在分母内。**"
            elif isinstance(total, int) and total > 0 and done == total:
                collapsed = []
                for condition in VLM_CONDITIONS:
                    score = (vlm.get("metrics") or {}).get(condition) or {}
                    matrix = score.get("confusion_matrix") or {}
                    count = score.get("n", 0)
                    for label in ("success", "failure"):
                        if count > 0 and sum(row.get(label, 0) for row in matrix.values()) == count:
                            collapsed.append(f"{NAMES[condition]}：全部预测 {label}")
                if collapsed:
                    status += "\n\n**单一类别输出：** " + "；".join(collapsed) + "。请结合失败召回判断，不能将结构有效等同于任务有效。"
        else:
            status += "\n\nVLM 实验尚未导出，不显示推测成绩或完成进度。"
        choices = [(f"{s.get('task_group', '—')} · {s['sample_id']}", s["sample_id"]) for s in samples]
        files = [str(_file(root, name)) for name in ("index.html", "metrics.json", "metrics.csv", "cv.json", "robot-benchmark.zip", "report.zip")
                 if _file(root, name).is_file()]
        return status, cv_rows, vlm_rows, choices, files
    except FileNotFoundError:
        return "**实验尚未导出。** 等待公开的 metrics.json 和 samples.json；不会用演示数字代替真实结果。", [], [], [], []
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return f"公开实验结果读取失败：{type(exc).__name__}: {exc}", [], [], [], []


def load_sensitivity(artifact_root: str | Path):
    """Read the separately exported post-hoc analysis, never replace the main CV."""
    root = Path(artifact_root).resolve()
    try:
        report = _read(root, "evaluation.json")
        if not isinstance(report, dict) or report.get("post_hoc") is not True or report.get("coarse_groups") != 4:
            raise ValueError("需要已标记后验性质的四粗组导出")
        scores = report.get("metrics") or {}
        rows = _metric_rows({"metrics": scores}, CV_METHODS)
        per_group = (scores.get("after") or {}).get("per_task") or {}
        counts = "/".join(str(per_group[name]["n"]) for name in sorted(per_group))
        status = "**独立后验分析，主16组结果保持原样。** "
        status += f"四组大小为 {counts}，分组极不平衡。" if counts else "分组大小未在导出中提供。"
        delta = ((report.get("paired_after_to_transition") or {}).get("delta") or {}).get("accuracy")
        if delta is not None:
            status += f" 前后变化相对动作后的 Accuracy 差值：{float(delta) * 100:+.2f}个百分点。"
            if float(delta) <= 0:
                status += "本次没有准确率增益。"
        status += " 仅4个粗组，仍共享抓取等动作；区间只作探索，不代表严格语义OOD。"
        return status, rows
    except FileNotFoundError:
        return "后验分组压力测试尚未导出。", []
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return f"后验分组压力测试读取失败：{type(exc).__name__}: {exc}", []


def sample_detail(artifact_root: str | Path, sample_id: str | None):
    root = Path(artifact_root).resolve()
    if not sample_id:
        return [], "选择一个公开样本查看原始观测与判断。", [], {}
    try:
        sample = next((s for s in _samples(root) if s["sample_id"] == sample_id), None)
        if sample is None:
            raise ValueError("所选样本不在公开导出中")
        gallery = []
        for frame in sample.get("frames", []):
            relative = frame.get("path", "")
            if not re.fullmatch(r"assets/[0-9a-f]{64}\.png", relative):
                raise ValueError("图片必须来自公开 assets 目录")
            image = _file(root, relative)
            if not image.is_file():
                raise ValueError("公开样本图片缺失")
            gallery.append((str(image), f"{frame.get('phase', '—')} · view slot {frame.get('view', '—')}"))
        if len(gallery) != 6:
            raise ValueError("公开样本应有动作前后各三张图")
        text = (f"任务：{sample.get('task_instruction', '—')}\n子任务：{sample.get('subtask_instruction', '—')}\n"
                f"作者标签：{sample.get('label', '—')}\n作者失败类型：{sample.get('failure_category', '—')}\n"
                "图像顺序遵循发布方列表；view slot 不代表统一物理相机方向。")
        predictions = {}
        if _file(root, "cv.json").is_file():
            cv = _read(root, "cv.json")
            for method in CV_METHODS:
                predictions[method] = next((row for row in cv.get("predictions", {}).get(method, [])
                                            if row.get("sample_id") == sample_id), None)
        for condition in VLM_CONDITIONS:
            name = f"predictions/{sample_id}.{condition}.json"
            if _file(root, name).is_file():
                row = _read(root, name)
                if row.get("sample_id") != sample_id or row.get("condition") != condition:
                    raise ValueError("公开预测与样本或实验条件不一致")
                predictions[condition] = row
        rows = []
        for method in (*CV_METHODS, *VLM_CONDITIONS):
            row = predictions.get(method)
            if not row:
                rows.append([NAMES[method], "未导出", "—", "—", "无已保存预测"])
                continue
            status, prediction = row.get("status", "missing"), row.get("prediction", "abstain")
            if method == "reversed":
                correct = "原标签不定义反向真值"
            elif status != "ok" or prediction == "abstain":
                correct = "未答对（保留分母）"
            else:
                correct = "正确" if prediction == sample.get("label") else "错误"
            evidence = row.get("evidence") or row.get("error") or ("线性分类器无文字解释；见原始记录的折与分数。" if method in CV_METHODS else "无文字证据")
            rows.append([NAMES[method], prediction, status, correct, evidence])
        return gallery, text, rows, {"sample_id": sample_id, "predictions": predictions}
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        return [], f"公开样本读取失败：{type(exc).__name__}: {exc}", [], {}


def build_benchmark_tab(artifact_root: str | Path):
    """Build inside an existing Gradio tab; no inference or data mutation."""
    root = Path(artifact_root).resolve()
    initial = load_overview(root)
    initial_id = initial[3][0][1] if initial[3] else None
    initial_detail = sample_detail(root, initial_id)
    sensitivity_root = root.parent / "robot-sensitivity"
    sensitivity_initial = load_sensitivity(sensitivity_root)
    gr.Markdown("### 机器人执行后，目标状态真的改变了吗？\n"
                "冻结 CLIP 的按基础任务组留一的线性分类，与本地 VLM 的动作后 / 前后顺序消融。这里只读取已导出的真实结果，不会更改协议或启动推理。\n\n"
                "**这是动作前后状态验证，没有秒级失败发生标签，不代表连续视频失败定位。** 反转输入只检验顺序敏感性；140条数据记录不等于140次独立执行。")
    status = gr.Markdown(initial[0])
    refresh = gr.Button("刷新已导出结果", variant="secondary")
    gr.Markdown("#### 按基础任务组留一验证 · 四种固定方法\n组 ID 只去掉 taskvar 的 variation 后缀，语义相近任务仍可能跨组，不代表严格的新任务泛化。全量预测包含在分母中；失败召回不能由整体准确率代替。特征维度同时变化，不能把差异单独归因于时序理解。")
    cv_table = gr.Dataframe(value=initial[1], headers=HEADERS, datatype=["str"] * len(HEADERS), interactive=False, wrap=True)
    gr.Markdown("#### 后验分组压力测试 · 4粗组")
    sensitivity_status = gr.Markdown(sensitivity_initial[0])
    sensitivity_table = gr.Dataframe(value=sensitivity_initial[1], headers=HEADERS, datatype=["str"] * len(HEADERS), interactive=False, wrap=True, visible=bool(sensitivity_initial[1]))
    gr.Markdown("#### 本地 VLM · 同一固定诊断子集\n正常顺序与动作后输入为主要配对比较；交换顺序相对原标签的分数仅供损坏诊断。")
    vlm_table = gr.Dataframe(value=initial[2], headers=HEADERS, datatype=["str"] * len(HEADERS), interactive=False, wrap=True)
    picker = gr.Dropdown(choices=initial[3], value=initial_id, label="选择公开诊断样本", interactive=True)
    detail = gr.Textbox(label="任务与发布方原始标签", value=initial_detail[1], lines=5, interactive=False)
    gallery = gr.Gallery(value=initial_detail[0], label="原始观测 · 第一行动作前，第二行动作后", columns=3, rows=2, height=840, object_fit="contain", interactive=False)
    predictions = gr.Dataframe(value=initial_detail[2], headers=["方法", "预测", "状态", "与原标签比较", "模型证据 / 错误"], datatype=["str"] * 5, interactive=False, wrap=True)
    with gr.Accordion("原始预测、模型身份与折记录", open=False):
        raw = gr.JSON(value=initial_detail[3])
    downloads = gr.File(value=initial[4] or None, label="下载实验报告与指标", file_count="multiple", interactive=False)
    gr.Markdown("HTML 报告依赖同目录的 assets 图片；完整离线查看请使用已导出的 ZIP。JSON 保留原始分母与完成数。置信区间固定现有逐折预测后按任务重采样，不涵盖重新训练的方差。")

    def reload(selected):
        overview = load_overview(root)
        selected = selected if selected in {value for _, value in overview[3]} else next((value for _, value in overview[3]), None)
        sensitivity = load_sensitivity(sensitivity_root)
        return (*overview[:3], gr.Dropdown(choices=overview[3], value=selected), overview[4] or None,
                sensitivity[0], gr.Dataframe(value=sensitivity[1], headers=HEADERS, datatype=["str"] * len(HEADERS),
                                            interactive=False, visible=bool(sensitivity[1])))

    refresh.click(reload, inputs=picker, outputs=[status, cv_table, vlm_table, picker, downloads, sensitivity_status, sensitivity_table]).then(
        lambda sid: sample_detail(root, sid), inputs=picker, outputs=[gallery, detail, predictions, raw])
    picker.change(lambda sid: sample_detail(root, sid), inputs=picker, outputs=[gallery, detail, predictions, raw])
