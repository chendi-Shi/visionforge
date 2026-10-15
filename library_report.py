"""A portable dossier linking separate video timelines without merging clocks."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import uuid
import zipfile

from search_report import export_search
from video_report import _safe_csv, clock
from workbench import STATUS, VERDICT, ACTOR, verdict


def export_library_search(run, reviews, outputdir):
    folder = Path(outputdir).expanduser().resolve() / uuid.uuid4().hex
    folder.mkdir(parents=True)
    videos, rows, cards = [], [], []
    for child in run.get('children', []):
        data = reviews.apply(child['search'])
        item = {k: child.get(k) for k in ('job_id', 'search_id', 'title', 'source_sha256', 'status', 'message', 'error')}
        # A missing child remains an explicit failure; corrupted present assets
        # cause export_search to fail instead of silently dropping evidence.
        if data.get('video'):
            files = export_search(data, folder / 'videos')
            report_folder = Path(files[1]).parent
            Path(files[0]).unlink()  # Avoid a redundant nested ZIP in the dossier.
            item['report'] = (report_folder / 'report.html').relative_to(folder).as_posix()
            item['evidence'] = Path(files[1]).relative_to(folder).as_posix()
        else:
            item['report'] = item['evidence'] = None
            item['export_note'] = '子检索缺少视频记录，未生成子报告；失败记录保留。'
        videos.append(item)
        current = data.get('review_current', {})
        raw = {r['frame']['id']: r for r in data.get('results', [])}
        for result in data.get('effective_results', []):
            frame = result['frame']
            review = current.get(frame['id'], {})
            rows.append([run['id'], child['job_id'], child['search_id'], child['title'],
                         frame['id'], frame['time_seconds'], clock(frame['time_seconds']),
                         verdict(raw.get(frame['id'], {})), verdict(result), result.get('judgment_source', 'model'),
                         review.get('revision', 0), frame.get('sha256'), child.get('source_sha256')])
        link = '<a href="' + escape(item['report'], quote=True) + '">打开该视频的时间证据与复核记录 →</a>' if item['report'] else '<p>该视频没有可导出的证据，详见错误。</p>'
        cards.append('<article><h2>' + escape(child['title']) + '</h2><p>' +
                     escape(STATUS.get(child['status'], child['status'])) + '</p><p>' +
                     escape(child.get('error') or child.get('message') or '') + '</p>' + link + '</article>')
    manifest = {k: run.get(k) for k in ('id', 'target', 'retrieval_query', 'model', 'config', 'status', 'progress', 'created_at', 'updated_at')}
    manifest.update(videos=videos, exported_at_utc=datetime.now(timezone.utc).isoformat(),
                    scope='每段视频具有独立时间轴；结果是各视频候选范围内的采样核验，不是全库穷尽性检测。')
    manifest_path, csv_path = folder / 'case.json', folder / 'timeline.csv'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'job_id', 'search_id', 'video', 'frame_id', 'time_seconds', 'time',
                         'model_verdict', 'effective_verdict', 'judgment_source', 'review_revision', 'frame_sha256', 'preview_sha256'])
        writer.writerows([[_safe_csv(value) for value in row] for row in rows])
    title = escape(run.get('target') or '视频库检索')
    (folder / 'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · VisionForge</title><style>
body{{font:15px system-ui;background:#f3f6f8;color:#173143;margin:0}}main{{max-width:1050px;margin:auto;padding:40px 24px}}h1{{font-size:28px}}
article{{padding:20px;background:white;border:1px solid #dbe5eb;border-radius:12px;margin:16px 0}}p{{line-height:1.7;overflow-wrap:anywhere}}a{{color:#087963}}
</style></head><body><main><p>VISIONFORGE / 跨视频证据包</p><h1>{title}</h1>
<p>状态：{escape(STATUS.get(run['status'], run['status']))} · 共 {len(videos)} 个视频。</p>
<p>{escape(manifest['scope'])} 每个报告分别保留模型原判、复核人与修订记录；没有复核的判断仍来自模型。</p>
<p><a href="timeline.csv">全部采样判断 CSV</a> · <a href="case.json">任务清单 JSON</a></p>
{''.join(cards)}</main></body></html>''', encoding='utf-8')
    archive = folder / 'video-library-evidence.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path != archive:
                compression = zipfile.ZIP_DEFLATED if path.suffix in {'.html', '.json', '.csv'} else zipfile.ZIP_STORED
                bundle.write(path, path.relative_to(folder).as_posix(), compress_type=compression)
    return [str(archive), str(csv_path), str(manifest_path)]
