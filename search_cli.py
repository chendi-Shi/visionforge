"""CLI for resumable local visual retrieval and evidence verification."""
import argparse
import json
from pathlib import Path
import time

from search_jobs import SearchEngine, DEFAULT_SEARCH_MODEL, ACTIVE
from video_jobs import VideoEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('job_id')
    create.add_argument('target')
    create.add_argument('--query', default='', help='Optional editable English CLIP query; otherwise translated locally')
    create.add_argument('--model', default=DEFAULT_SEARCH_MODEL)
    create.add_argument('--top-k', type=int, default=2)
    create.add_argument('--radius', type=float, default=2)
    create.add_argument('--step', type=float, default=1)
    create.add_argument('--max-frames', type=int, default=12)
    create.add_argument('--max-edge', type=int, default=448)
    create.add_argument('--max-tokens', type=int, default=160)
    create.add_argument('--wait', action='store_true')
    commands.add_parser('list').add_argument('--job-id')
    commands.add_parser('status').add_argument('search_id')
    commands.add_parser('pause').add_argument('search_id')
    resume = commands.add_parser('resume')
    resume.add_argument('search_id')
    resume.add_argument('--wait', action='store_true')
    export = commands.add_parser('export')
    export.add_argument('search_id')
    export.add_argument('--output', type=Path)
    args = parser.parse_args()
    engine = SearchEngine(VideoEngine(args.data_dir))
    if args.command == 'create':
        search_id = engine.create_search(args.job_id, args.target, args.query, args.model, args.top_k,
            args.radius, args.step, args.max_frames, args.max_edge, args.max_tokens)
    elif args.command == 'resume':
        search_id = args.search_id
        engine.resume(search_id)
    elif args.command == 'pause':
        print(json.dumps(engine.pause(args.search_id), ensure_ascii=False, indent=2))
        return
    elif args.command == 'list':
        print(json.dumps([{key: row[key] for key in ('id','job_id','status','target','message')}
                          for row in engine.list_searches(args.job_id)], ensure_ascii=False, indent=2))
        return
    elif args.command == 'status':
        print(json.dumps(engine.get_search(args.search_id), ensure_ascii=False, indent=2))
        return
    else:
        from search_report import export_search
        print(json.dumps(export_search(engine.get_search(args.search_id), args.output or engine.root / 'exports'),
                         ensure_ascii=False, default=str))
        return
    if args.wait:
        engine.start()
        try:
            while engine.get_search(search_id)['status'] in ACTIVE:
                time.sleep(.5)
        except KeyboardInterrupt:
            engine.pause(search_id)
        finally:
            engine.stop()
    print(json.dumps(engine.get_search(search_id), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
