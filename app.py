"""Robot action-effect experiments, video library, and evidence review."""
from __future__ import annotations
import html
import json
import os
from pathlib import Path
from video_jobs import VideoEngine, default_data_dir

ROOT = Path(__file__).resolve().parent
DATA_ROOT = default_data_dir()
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_TEMP_DIR'] = str(DATA_ROOT / 'uploads')
import gradio as gr
from core import list_vision_models
from search_jobs import SearchEngine, DEFAULT_SEARCH_MODEL
from library_jobs import LibraryEngine
from search_reviews import ReviewStore
from search_report import export_search
from workbench import STATUS, VERDICT, ACTOR, metrics, library_rows, evidence_rows, interval_rows, cross_video_rows, verdict
from video_report import clock
from robotbench.ui import build_benchmark_tab
from robotbench.temporal_ui import build_temporal_tab

ROBOT_ARTIFACT_ROOT = ROOT / 'artifacts' / 'robot-benchmark'
TEMPORAL_ARTIFACT_ROOT = ROOT / 'artifacts' / 'robot-temporal'
TEMPORAL_MINICPM_ROOT = ROOT / 'artifacts' / 'robot-temporal-minicpm'

CSS = """
.gradio-container{max-width:1520px!important;background:#f5f7fa}
.gradio-container,.gradio-container input,.gradio-container textarea,.gradio-container button,.gradio-container table{font-family:'Segoe UI','Microsoft YaHei',sans-serif!important}
.gradio-container button.primary{background:#087f70!important;border-color:#087f70!important}
#hero{background:#102737;color:white;padding:24px 28px;border-radius:14px;margin-bottom:14px}
#hero h1{color:white;font-size:27px;margin:4px 0 9px}#hero p{color:#c9dbe5;margin:0;line-height:1.7}
.eyebrow{color:#65ddbc;font:600 11px monospace;letter-spacing:2px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0 14px}
.metric{background:white;border:1px solid #dce5eb;border-radius:10px;padding:15px}
.metric b{display:block;font-size:22px;color:#18394b}.metric span,.subtle{font-size:12px;color:#647b8b}
.notice{padding:12px 16px;background:#fff2d6;border-radius:8px;line-height:1.7}
#evidence-frame img{max-height:360px;object-fit:contain}
@media(max-width:700px){.metrics{grid-template-columns:repeat(2,1fr)}}
"""
SEEK_JS = """(seconds, previewKey) => {
 const target=Number(seconds);if(!Number.isFinite(target)||target<0)return [];
 const expected=String(previewKey||'').split('|').slice(1).join('|').replaceAll('\\\\','/');
 const request=window.__visionforgeSeek=(window.__visionforgeSeek||0)+1;
 const deadline=Date.now()+8000;
 const seek=()=>{const v=document.querySelector('#source-player video');
 if(window.__visionforgeSeek!==request)return;
 const source=v?decodeURIComponent(v.currentSrc||v.src||'').replaceAll('\\\\','/'):'';
 if(!v||v.readyState<1||(expected&&!source.includes(expected))){if(Date.now()<deadline)setTimeout(seek,100);return;}
 v.currentTime=Math.min(target,Number.isFinite(v.duration)?v.duration:target);v.pause();};
 seek();return [];
}"""


def build_app(engine=None, searches=None, library=None, reviews=None):
    engine = engine or VideoEngine(DATA_ROOT)
    searches = searches or SearchEngine(engine)
    library = library or LibraryEngine(engine, searches)
    reviews = reviews or ReviewStore(engine.root)
    engine.start()
    searches.start()

    def guarded(fn):
        def call(*args):
            try:
                return fn(*args)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
        return call

    def model_options():
        try:
            names = list_vision_models()
            selected = DEFAULT_SEARCH_MODEL if DEFAULT_SEARCH_MODEL in names else next(iter(names), None)
            return gr.Dropdown(choices=names, value=selected), f'本地 Ollama · {len(names)} 个视觉模型 · CPU 串行队列'
        except Exception as exc:
            return gr.Dropdown(choices=[], value=None), f'Ollama 暂不可用；仍可导入视频和复核已有证据。{exc}'

    def refresh_library(query, archived, selected):
        videos = library.list_videos(include_archived=True)
        rows, options = library_rows(videos, query, archived)
        selected = [x for x in (selected or []) if x in {value for _, value in options}]
        summary = metrics([(sum(not v.get('archived') for v in videos), '库中视频'), (len(options), '当前筛选可搜索'),
                           (len(selected), '已选搜索范围'), (sum(v['status'] in {'preparing', 'queued_prepare'} for v in videos), '等待 / 正在预处理')])
        return [r[1:] for r in rows], gr.CheckboxGroup(choices=options, value=selected), selected, summary, rows

    def poll_library(query, archived, selected):
        # A tick may have read the old selection before a newer user input.
        # Background progress must never write choices, checkbox value or State.
        rows, _, _, summary, records = refresh_library(query, archived, selected)
        return rows, summary, records

    @guarded
    def import_files(paths):
        if not paths:
            raise ValueError('请选择一个或多个视频。')
        engine.start()
        result = library.import_videos(paths, DEFAULT_SEARCH_MODEL)
        lines = [f"已接收 {result['succeeded']} 个视频；失败 {result['failed']} 个。预处理在后台继续，可离开此页。"]
        lines += [f"{Path(item['path']).name}：{item['error']}" for item in result['items'] if item.get('error')]
        return '\n'.join(lines)

    def select_video(rows, evt: gr.SelectData):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        meta = next(v for v in library.list_videos(include_archived=True) if v['id'] == rows[idx][0])
        return meta['id'], meta['title'], ', '.join(meta.get('tags', [])), meta.get('archived', False)

    @guarded
    def save_video(job_id, title, tags, archived):
        if not job_id:
            raise ValueError('先点击视频库中的一行。')
        library.update_video(job_id, title=title, tags=[x.strip() for x in tags.replace('，', ',').split(',') if x.strip()], archived=archived)
        return '视频资料已保存。归档仅从搜索范围隐藏，已有证据仍可打开。'

    def run_choices(selected=None):
        options = [(f"{x['target'][:35]} · {x['id'][:8]}", x['id']) for x in library.list_searches()]
        return options, selected if selected in {v for _, v in options} else next((v for _, v in options), None)

    def run_options(selected=None):
        options, value = run_choices(selected)
        return gr.Dropdown(choices=options, value=value), value

    def evidence_choices(selected=None):
        items = searches.list_searches()
        titles = {v['id']: v['title'] for v in library.list_videos(include_archived=True)}
        options = [(f"{titles.get(x['job_id'], x['job_name'])} · {x['target'][:25]} · {x['id'][:8]}", x['id']) for x in items]
        default = next((x['id'] for x in items if x['status'] == 'completed'), next((v for _, v in options), None))
        return options, selected if selected in {v for _, v in options} else default

    def evidence_options(selected=None):
        options, value = evidence_choices(selected)
        return gr.Dropdown(choices=options, value=value), value

    def remember_run(selected):
        options = run_choices()[0]
        if selected is not None and selected not in {value for _, value in options}:
            gr.Warning('该任务已不可用，已清空选择。')
            return gr.Dropdown(choices=options, value=None), None
        return gr.skip(), selected

    def remember_evidence(selected):
        options = evidence_choices()[0]
        if selected is not None and selected not in {value for _, value in options}:
            gr.Warning('该检索记录已不可用，已清空选择。')
            return gr.Dropdown(choices=options, value=None), None
        return gr.skip(), selected

    def remember_scope(selected, query, archived):
        options = library_rows(library.list_videos(include_archived=True), query, archived)[1]
        available = {value for _, value in options}
        selected = [value for value in (selected or []) if value in available]
        return gr.CheckboxGroup(choices=options, value=selected), selected

    def select_filtered(query, archived):
        options = library_rows(library.list_videos(include_archived=True), query, archived)[1]
        selected = [value for _, value in options]
        return gr.CheckboxGroup(choices=options, value=selected), selected

    @guarded
    def start_run(job_ids, target, english, model, total, top_k, radius, step, per_video, edge, tokens):
        if not model:
            raise ValueError('请启动 Ollama 并选择一个已安装的视觉模型。')
        if not job_ids:
            raise ValueError('先在视频库选择要搜索的视频。')
        run_id = library.create_search(job_ids, target, retrieval_query=english or '', model=model,
            max_total_frames=int(total), top_k=int(top_k), radius_seconds=float(radius), step_seconds=float(step),
            max_frames=int(per_video), max_edge=int(edge), max_tokens=int(tokens))
        searches.start()
        return *run_options(run_id), '任务已持久化。按视频依次检索和核验，可暂停后继续。', gr.Tabs(selected='search')

    @guarded
    def run_action(run_id, action):
        if not run_id:
            raise ValueError('先选择一个跨视频任务。')
        data = library.pause(run_id) if action == 'pause' else library.resume(run_id)
        searches.start()
        return f"{STATUS.get(data['status'], data['status'])} · 已保存的证据保留。"

    def run_view(run_id, visible_only):
        if not run_id:
            return "<p class='subtle'>在视频库勾选搜索范围，再提交目标。每个视频独立保存证据。</p>", [], [], {}, [], []
        try:
            data = library.get_search(run_id)
            progress = data['progress']
            summary = metrics([(STATUS.get(data['status'], data['status']), '任务状态'),
                (f"{progress['videos_completed']} / {progress['videos_total']}", '已完成视频'),
                (progress['frames_verified'], '已核验帧'), (progress['videos_failed'], '失败视频')])
            rows = [[c['search_id'], c['title'], STATUS.get(c['status'], c['status']),
                     f"{c['verified_frames']} / {c['planned_frames']}", c.get('error') or c.get('message') or ''] for c in data['children']]
            hits = cross_video_rows(data['children'], reviews, visible_only)
            return summary, [r[1:] for r in rows], [r[3:] for r in hits], data, rows, hits
        except Exception as exc:
            return "<p class='notice'>" + html.escape(str(exc)) + '</p>', [], [], {}, [], []

    def open_child(rows, evt: gr.SelectData):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return *evidence_options(rows[idx][0]), gr.Tabs(selected='evidence')

    def open_hit(rows, evt: gr.SelectData):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        row = rows[idx]
        return *evidence_options(row[0]), gr.Tabs(selected='evidence'), {'search_id': row[0], 'frame_id': row[2]}

    def evidence_view(search_id, only, old_key, old_preview):
        empty = ("<p class='subtle'>打开一个视频的检索记录后，选择证据帧复核。</p>", [], [], [], '', {}, '', gr.skip(), old_preview)
        if not search_id:
            return empty
        try:
            data = reviews.apply(searches.get_search(search_id))
            key = f"{search_id}|{data['updated_at']}|{json.dumps(data.get('review_current', {}), sort_keys=True)}|{only}"
            if key == old_key:
                return tuple(gr.skip() for _ in empty)
            results = data.get('effective_results', data.get('results', []))
            current = [r for r in data.get('review_current', {}).values() if r.get('state') == 'active']
            summary = metrics([(STATUS.get(data['status'], data['status']), '核验状态'),
                (f"{len(data.get('results', []))} / {len(data.get('frames', []))}", '已核验 / 计划帧'),
                (sum(verdict(r) == 'visible' for r in results), '当前判定可见'), (len(current), '有复核记录')])
            summary += "<p class='subtle'>" + html.escape(data.get('error') or data.get('message') or '') + '</p>'
            for rejection in data.get('review_rejections', []):
                summary += "<p class='notice'>复核暂未应用：" + html.escape(rejection['frame_id'] + ' · ' + rejection['reason']) + '</p>'
            if data.get('review_interval_warning'):
                summary += "<p class='notice'>" + html.escape(data['review_interval_warning']) + '</p>'
            translation = data.get('translation') or {}
            query = f"目标：{translation.get('visual_target') or data['target']}\n检索词：{translation.get('query') or '等待生成'}"
            candidates = [(f['path'], f"{clock(f['time_seconds'])} · 相似度 {f.get('score',0):.3f}") for f in data.get('candidates', [])]
            preview = data['video'].get('preview_path')
            # Gradio serves a content-addressed cached copy, not the input path.
            # Bind seek to that exact source so switching videos cannot seek the old one.
            cached_preview = gr.processing_utils.save_file_to_cache(preview, os.environ['GRADIO_TEMP_DIR']) if preview else None
            preview_key = f"{data['job_id']}|{cached_preview}"
            return (summary, evidence_rows(data, only), interval_rows(data.get('reviewed_intervals', data.get('intervals', {}))),
                    candidates, query, data, key, gr.Video(value=cached_preview) if preview_key != old_preview else gr.skip(), preview_key)
        except Exception as exc:
            return ("<p class='notice'>" + html.escape(str(exc)) + '</p>', *empty[1:])

    def frame_detail(search_id, fid):
        data = reviews.apply(searches.get_search(search_id))
        frame = next(f for f in data.get('frames', []) if f['id'] == fid)
        result = next((r for r in data.get('results', []) if r['frame']['id'] == fid), {})
        review = data.get('review_current', {}).get(fid, {})
        description = f"{data['job_name']} · {clock(frame['time_seconds'])} · {fid}\n模型原判：{VERDICT.get(verdict(result), verdict(result))}\n{(result.get('structured') or {}).get('reason') or result.get('error') or ''}"
        if review:
            description += f"\n最近复核：{ACTOR.get(review.get('actor'), review.get('actor', ''))} · 版本 {review['revision']}"
            if review.get('state') in {'stale', 'deferred'}:
                description += '\n该复核当前未应用，请查看上方原因。'
        value = review.get('verdict', verdict(result))
        if value not in {'visible', 'not_visible', 'uncertain', 'reset'}:
            value = 'uncertain'
        return (frame['path'], description, {'search_id': search_id, 'frame_id': fid, 'revision': review.get('revision', 0)},
                value, review.get('note', ''), {'model_original': result, 'review_history': [r for r in data.get('review_history', []) if r['frame_id'] == fid]}, frame['time_seconds'])

    def select_frame(search_id, rows, evt: gr.SelectData):
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return frame_detail(search_id, rows[idx][0])

    def requested_frame(search_id, request):
        if request and request.get('search_id') == search_id:
            return frame_detail(search_id, request['frame_id'])
        return None, '选择左侧一帧，查看原图与模型判断。', {}, 'uncertain', '', {}, -1

    def reload_selected(search_id, selected):
        return requested_frame(search_id, selected)

    @guarded
    def save_review(search_id, selection, decision, note):
        if not selection or selection.get('search_id') != search_id:
            raise ValueError('视频或任务已切换，请重新选择证据帧。')
        saved = reviews.save(searches.get_search(search_id), selection['frame_id'], decision, note,
                             actor='user', expected_revision=selection['revision'])
        return '复核已保存；当前时间证据已重算，模型原始输出保留。', {**selection, 'revision': saved['revision']}, ''

    @guarded
    def export_evidence(search_id):
        if not search_id:
            raise ValueError('先选择检索记录。')
        return export_search(reviews.apply(searches.get_search(search_id)), engine.root / 'exports')

    @guarded
    def export_run(run_id):
        if not run_id:
            raise ValueError('先选择跨视频任务。')
        from library_report import export_library_search
        return export_library_search(library.get_search(run_id), reviews, engine.root / 'exports')

    with gr.Blocks(title='VisionForge · 机器人失败时间定位') as demo:
        gr.HTML('<div id="hero"><div class="eyebrow">VISIONFORGE / ROBOT FAILURE LOCALIZATION</div><h1>失败发生在哪一秒，证据是什么？</h1><p>连续机器人轨迹、独立人工时间标注与等预算抽帧实验。直接回到录像核对模型判断、漏检和误报。</p></div>')
        with gr.Row():
            model = gr.Dropdown(label='视频搜索模型（不更改已保存实验）', choices=[], scale=2)
            connection = gr.Textbox(label='运行环境', interactive=False, scale=4)
            refresh_model = gr.Button('刷新模型', scale=1)
        with gr.Tabs(selected='temporal') as tabs:
            with gr.Tab('连续轨迹 · 失败定位', id='temporal'):
                with gr.Tabs(selected='qwen' if (TEMPORAL_ARTIFACT_ROOT / 'evaluation.json').is_file() else 'minicpm'):
                    with gr.Tab('Qwen3.5 · 4B', id='qwen'):
                        build_temporal_tab(TEMPORAL_ARTIFACT_ROOT)
                    with gr.Tab('MiniCPM-V 4.6 · 1B', id='minicpm'):
                        build_temporal_tab(TEMPORAL_MINICPM_ROOT, video_id='temporal-video-minicpm')
            with gr.Tab('机器人算法实验', id='robot'):
                build_benchmark_tab(ROBOT_ARTIFACT_ROOT)
            with gr.Tab('01  视频库', id='library'):
                library_summary = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            filter_text = gr.Textbox(label='按名称 / 标签筛选', placeholder='会议、路口、巡检…')
                            archived = gr.Checkbox(label='显示已归档', value=False)
                            refresh = gr.Button('刷新视频库')
                        videos_table = gr.Dataframe(headers=['名称', '标签', '时长', '状态'], datatype=['str']*4, type='array', interactive=False, wrap=True)
                        with gr.Accordion('整理选中的视频', open=False):
                            edit_id = gr.State('')
                            title = gr.Textbox(label='名称')
                            tags = gr.Textbox(label='标签，用逗号分隔')
                            archive_video = gr.Checkbox(label='归档')
                            save_metadata = gr.Button('保存资料')
                        library_message = gr.Textbox(label='处理记录', interactive=False)
                        with gr.Accordion('批量导入视频', open=True):
                            uploads = gr.File(label='拖入多个本地视频 · 每个最大 1 GB', file_count='multiple', file_types=['video'], type='filepath')
                            import_button = gr.Button('导入视频库', variant='primary')
                            gr.Markdown('预处理不需要 Ollama 在线。原文件复制到本地资料库；视频预览无音轨。')
                    with gr.Column(scale=2):
                        scope = gr.CheckboxGroup(label='本次搜索范围 · 勾选已准备的视频',
                                                 choices=library_rows(library.list_videos(include_archived=True))[1], value=[])
                        gr.Markdown('新增视频预处理完成后，点击“刷新视频库”更新可选搜索范围。')
                        select_all = gr.Button('选择当前筛选下的全部视频')
                        target = gr.Textbox(label='要找什么', placeholder='例如：带 Red Bull 图案的货车', lines=2)
                        english = gr.Textbox(label='英文检索词（可选）', placeholder='留空由本地模型转换')
                        total_budget = gr.Slider(12, 192, value=48, step=3, label='整次任务最多核验帧数（所有视频合计）')
                        budget_note = gr.Markdown()
                        with gr.Accordion('采样与推理设置', open=False):
                            top_k = gr.Slider(1, 4, value=2, step=1, label='每视频候选数')
                            radius = gr.Slider(1, 6, value=2, step=.5, label='候选前后范围 / 秒')
                            step = gr.Slider(.5, 2, value=1, step=.5, label='核验间隔 / 秒')
                            per_video = gr.Slider(3, 48, value=12, step=1, label='单视频核验帧数上限')
                            edge = gr.Slider(224, 768, value=448, step=32, label='核验图像最长边')
                            tokens = gr.Slider(96, 512, value=160, step=16, label='每帧输出 Token 上限')
                        submit = gr.Button('搜索所选视频', variant='primary')
                        submit_message = gr.Textbox(label='提交状态', interactive=False)
            with gr.Tab('02  搜索任务', id='search'):
                with gr.Row():
                    run_picker = gr.Dropdown(label='已保存的跨视频任务', choices=run_choices()[0], value=None, scale=4)
                    refresh_runs = gr.Button('刷新任务', scale=1)
                    pause_run = gr.Button('暂停整次任务', scale=1)
                    resume_run = gr.Button('继续 / 重试失败项', scale=1)
                run_message = gr.Textbox(label='任务操作', interactive=False)
                run_summary = gr.HTML()
                gr.Markdown('每个视频分别检索并核验。点击进度行打开该视频；失败项保留错误信息。')
                children_table = gr.Dataframe(headers=['视频', '状态', '已核验 / 计划', '进度 / 错误'], datatype=['str']*4, type='array', interactive=False, wrap=True)
                only_visible = gr.Checkbox(label='结果只显示当前判定可见的帧', value=True)
                hits_table = gr.Dataframe(headers=['视频', '时间', '当前判断', '来源'], datatype=['str']*4, type='array', interactive=False, wrap=True)
                gr.Markdown('点击结果会切换到对应视频和证据帧。跨视频相似度不作为事件概率；候选范围之外没有作出不存在的判断。')
                export_case = gr.Button('导出整次任务证据包')
                case_files = gr.File(label='跨视频报告 / CSV / 证据包', file_count='multiple', interactive=False)
                run_state = gr.State({})
            with gr.Tab('03  证据复核', id='evidence'):
                with gr.Row():
                    evidence_picker = gr.Dropdown(label='视频检索记录（含历史单视频任务）', choices=evidence_choices()[0], value=None, scale=5)
                    refresh_evidence = gr.Button('刷新记录', scale=1)
                evidence_summary = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=3):
                        player = gr.Video(label='原时间轴预览', elem_id='source-player', interactive=False)
                        query_detail = gr.Textbox(label='实际检索与核验目标', interactive=False, lines=2)
                        evidence_filter = gr.Radio(['全部', '待复核', '可见', '不确定 / 失败'], label='证据筛选', value='全部')
                        frames_table = gr.Dataframe(headers=['帧 ID', '时间', '模型原判', '当前判断', '来源'], datatype=['str']*5, type='array', interactive=False, wrap=True)
                    with gr.Column(scale=2):
                        frame_image = gr.Image(label='选中的证据原帧', type='filepath', interactive=False, elem_id='evidence-frame')
                        frame_description = gr.Textbox(label='证据详情', interactive=False, lines=4)
                        decision = gr.Radio(choices=[(VERDICT[x], x) for x in ('visible', 'not_visible', 'uncertain', 'reset')], value='uncertain', label='复核判断')
                        review_note = gr.Textbox(label='复核说明', placeholder='写明画面依据；修正判断必须填写。', lines=2)
                        save = gr.Button('保存复核并重算时间证据', variant='primary')
                        review_message = gr.Textbox(label='保存状态', interactive=False)
                        with gr.Accordion('原始模型输出与复核历史', open=False):
                            frame_raw = gr.JSON()
                gr.HTML('<p class="notice">下方按当前逐帧判断推导。复核过的帧与未经复核的模型判断可能同时参与；采样之间不证明连续可见，区间也不是精确事件边界。</p>')
                intervals_table = gr.Dataframe(headers=['时间证据', '起点', '终点', '依据帧'], datatype=['str']*4, type='array', interactive=False, wrap=True)
                with gr.Accordion('视觉检索候选', open=False):
                    candidates_gallery = gr.Gallery(columns=4, height=230, interactive=False)
                with gr.Accordion('完整记录（原始结果与复核分开保存）', open=False):
                    evidence_raw = gr.JSON()
                export_button = gr.Button('导出此视频证据包')
                evidence_files = gr.File(label='报告、原始记录与复核审计', file_count='multiple', interactive=False)
        timer = gr.Timer(4)
        evidence_key, preview_key = gr.State(''), gr.State('')
        selection, requested = gr.State({}), gr.State({})
        library_records, child_records, hit_records = gr.State([]), gr.State([]), gr.State([])
        # Dynamic choices belong to a Gradio session. After a restart, an old
        # browser can still send a selected ID while its new session has none
        # of the previous component updates. Poll only canonical session State;
        # load/refresh/navigation update the widget and State together.
        selected_scope, selected_run, selected_evidence = gr.State([]), gr.State(None), gr.State(None)
        seek = gr.Number(value=-1, visible=False)
        lib_in, lib_out = [filter_text, archived, selected_scope], [videos_table, scope, selected_scope, library_summary, library_records]
        ev_in = [selected_evidence, evidence_filter, evidence_key, preview_key]
        ev_out = [evidence_summary, frames_table, intervals_table, candidates_gallery, query_detail, evidence_raw, evidence_key, player, preview_key]
        detail_out = [frame_image, frame_description, selection, decision, review_note, frame_raw, seek]
        choice_events = {'concurrency_id': 'workbench_choices', 'concurrency_limit': 1}
        demo.load(model_options, outputs=[model, connection])
        demo.load(refresh_library, inputs=lib_in, outputs=lib_out, **choice_events)
        demo.load(run_options, outputs=[run_picker, selected_run], **choice_events)
        demo.load(evidence_options, outputs=[evidence_picker, selected_evidence], **choice_events)
        refresh_model.click(model_options, outputs=[model, connection])
        refresh.click(refresh_library, inputs=lib_in, outputs=lib_out, **choice_events)
        filter_text.submit(refresh_library, inputs=lib_in, outputs=lib_out, **choice_events)
        archived.change(refresh_library, inputs=lib_in, outputs=lib_out, **choice_events)
        timer.tick(poll_library, inputs=lib_in, outputs=[videos_table, library_summary, library_records], show_progress='hidden')
        import_button.click(import_files, inputs=uploads, outputs=library_message).then(refresh_library, inputs=lib_in, outputs=lib_out, **choice_events)
        videos_table.select(select_video, inputs=library_records, outputs=[edit_id, title, tags, archive_video])
        save_metadata.click(save_video, inputs=[edit_id, title, tags, archive_video], outputs=library_message).then(refresh_library, inputs=lib_in, outputs=lib_out, **choice_events)
        select_all.click(select_filtered, inputs=[filter_text, archived], outputs=[scope, selected_scope], **choice_events)
        scope.input(remember_scope, inputs=[scope, filter_text, archived], outputs=[scope, selected_scope], **choice_events)
        def budget_text(ids, total, cap):
            n = len(ids or [])
            return f'已选 **{n}** 个视频；本次最多核验 **{min(int(total), n * int(cap))}** 帧。索引与核验串行执行，耗时随视频数量和时长增加。'
        for component in (selected_scope, total_budget, per_video):
            component.change(budget_text, inputs=[selected_scope, total_budget, per_video], outputs=budget_note)
        # Actions consume the current browser choice, not an asynchronously
        # mirrored State. This also preserves the explicit named API inputs.
        submit.click(start_run, inputs=[scope, target, english, model, total_budget, top_k, radius, step, per_video, edge, tokens], outputs=[run_picker, selected_run, submit_message, tabs], api_name='create_library_search', **choice_events)
        run_picker.input(remember_run, inputs=run_picker, outputs=[run_picker, selected_run], **choice_events)
        evidence_picker.input(remember_evidence, inputs=evidence_picker, outputs=[evidence_picker, selected_evidence], **choice_events)
        refresh_runs.click(run_options, inputs=selected_run, outputs=[run_picker, selected_run], **choice_events)
        pause_run.click(lambda rid: run_action(rid, 'pause'), inputs=run_picker, outputs=run_message)
        resume_run.click(lambda rid: run_action(rid, 'resume'), inputs=run_picker, outputs=run_message)
        for event in (selected_run.change, only_visible.change, timer.tick):
            event(run_view, inputs=[selected_run, only_visible], outputs=[run_summary, children_table, hits_table, run_state, child_records, hit_records], show_progress='hidden')
        children_table.select(open_child, inputs=child_records, outputs=[evidence_picker, selected_evidence, tabs], **choice_events)
        hits_table.select(open_hit, inputs=hit_records, outputs=[evidence_picker, selected_evidence, tabs, requested], **choice_events).then(evidence_view, inputs=ev_in, outputs=ev_out).then(requested_frame, inputs=[selected_evidence, requested], outputs=detail_out).then(None, inputs=[seek, preview_key], js=SEEK_JS)
        refresh_evidence.click(evidence_options, inputs=selected_evidence, outputs=[evidence_picker, selected_evidence], **choice_events)
        selected_evidence.change(evidence_view, inputs=ev_in, outputs=ev_out).then(requested_frame, inputs=[selected_evidence, requested], outputs=detail_out)
        evidence_filter.change(evidence_view, inputs=ev_in, outputs=ev_out)
        timer.tick(evidence_view, inputs=ev_in, outputs=ev_out, show_progress='hidden')
        frames_table.select(select_frame, inputs=[selected_evidence, frames_table], outputs=detail_out).then(None, inputs=[seek, preview_key], js=SEEK_JS)
        save.click(save_review, inputs=[evidence_picker, selection, decision, review_note], outputs=[review_message, selection, evidence_key], api_name='save_evidence_review').then(evidence_view, inputs=ev_in, outputs=ev_out).then(reload_selected, inputs=[selected_evidence, selection], outputs=detail_out)
        export_button.click(export_evidence, inputs=evidence_picker, outputs=evidence_files)
        export_case.click(export_run, inputs=run_picker, outputs=case_files)
    return demo


if __name__ == '__main__':
    build_app().queue(default_concurrency_limit=4).launch(server_name='127.0.0.1', server_port=7860,
        css=CSS, allowed_paths=[str(DATA_ROOT), str(ROBOT_ARTIFACT_ROOT), str(TEMPORAL_ARTIFACT_ROOT), str(TEMPORAL_MINICPM_ROOT)], max_file_size='1gb', inbrowser=False, share=False)
