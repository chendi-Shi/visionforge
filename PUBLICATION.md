# Public release preparation

This Git checkout began as an independent publication copy at commit `4957ab2`. The notes below document that initial publication and the preserved historical snapshots. Subsequent commits add visual retrieval and single-frame grounding; their new measurements are recorded separately from the old dual-frame results.

## Initial publication redaction

Only machine-specific path fields in existing JSON evidence were replaced with repository-relative paths; the recorded FFmpeg executable path was replaced with `ffmpeg`. Paths in these published snapshots are resolved from the repository root. The standalone report retains its existing package-relative paths. Historical metadata may still identify the original ZIP export; the duplicate ZIP is deliberately excluded from Git.

The affected fields are evaluation `manifest`, paused/completed job `source_path`, normalized-video `preview_path`, evidence frame `path`, validation `exports`, exploratory extraction `command`, and pipeline-plan paths. Source-video paths map to `examples/video/street-traffic.webm`; normalized previews map to the existing `artifacts/video-evidence/` package; pipeline frames map to the matching `examples/video/pipeline-evidence/` bytes.

Path redaction changes JSON file hashes. The manual reference's `plan_sha256` and documentation references to changed snapshots were updated to identify the published bytes. Original video, preview, image, model, and code hashes were preserved. Original model responses, observations, timing, measured status, review decisions, and human/assistant reference annotations were not changed. No successful run or accuracy claim was added.

At the initial publication, the only change to existing Python code was the missing-sample message in `app.py`; `tools/fetch_video_sample.py` was added to fetch the original on demand. The later retrieval upgrade changes the application, adds search modules and an inference lock in `video_jobs.py`, and extends installation and documentation. Historical JSON results are not rewritten as new-model results. `core.py` and `core_video.py` retain the identities recorded with the original outputs. `.gitignore` excludes secrets, databases, environments, runtime data, caches and optional large binaries.

## Intentional binary exclusions

The 26,490,572-byte original `examples/video/street-traffic.webm` is not stored in Git. The downloader streams into a temporary file, enforces the recorded byte count and SHA256, and only then atomically installs the file. An existing verified file is reused; an existing different file is never overwritten. The official source, expected original hashes and CC BY attribution are retained.

The 10,148,953-byte prebuilt `video-review.zip` is also excluded because its complete extracted assets remain in `artifacts/video-evidence/0354ceb6ccaf4e3eb7857a7368ff0b67/`: `report.html`, JSON, CSV, the normalized H264 MP4 and all frames. After cloning, open that local HTML directly without downloading the original or running a model. The UI can still export a new ZIP. These exclusions reduce transfer size; they are intentional, not missing uploads. They do not change original model evidence or evaluation outcomes.

## Attribution

The sample [Street traffic.webm](https://commons.wikimedia.org/wiki/File:Street_traffic.webm) is by [Editor](https://www.youtube.com/user/Editor), originally published at [YouTube](https://www.youtube.com/watch?v=_7sCYyxw4Ic), licensed under [CC BY 3.0 Unported](https://creativecommons.org/licenses/by/3.0/). Cropped/resized evidence frames and the normalized silent preview are derivatives. Full provenance and transformation notes remain in `examples/video/source.json` and in the exported report's `source_attribution`. Project MIT licensing does not replace third-party media/model/dependency licenses; see `THIRD_PARTY.md` and `licenses/`.

## Snapshot integrity after path redaction

| Published file | Original local snapshot SHA-256 | Published SHA-256 |
| --- | --- | --- |
| `artifacts/evaluation-32-tokens.json` | `dc64a85d0f72a86739723af37ff6e963e31a2e51bbe4d2ddb31493afea84df59` | `f4dc6570159dff73b466358d89b274b44a2f1389e32e55c23253cdc87b93725a` |
| `artifacts/evaluation.json` | `4109d4d9d2522a0a89655a3bbec90b967abfb543d5a8e3906af6fe323e7ec487` | `0b62fcf3c758c4caf56564391580c8335d25102da29f1a654387dd22048217bf` |
| `artifacts/video-completed.json` | `828ab8a8924cb6f99a0a065e5e153f9d3f4fed43c799d787bf21eede3bd6affb` | `73b2f57e912ebae7ed8031ed4a6c8cff160b8dc4d405c5495cb3a1a521235d31` |
| `artifacts/video-paused.json` | `c0d340b635d6bbdbeff781a721d7df5aa2397a6c2e73edbc04215c99b5ceae2d` | `ee58ca64177dab032b410defd40044f318a73ecbdb17fba012eee69b4c536b11` |
| `artifacts/video-validation.json` | `e535d2cf3618affc8ea6793ab832ece2f0699fedf8ffe4d2d65b98a304b5a8df` | `5eacae88b961196bb7deecaefd42ed7584376e308aa9301afaf560925d461e56` |
| `examples/video/frames.json` | `a6d62f43dc9977f2431e83b91109f72c3174a5f6722fc2ea98235ab465c858b0` | `91dbaa5058aa3b44fb3090727083b889f274c7c30e68f221d51ee0ca1dd26b0b` |
| `examples/video/pipeline-plan.json` | `8b4e81dedbeb9bc7e4527998c8d28fd4009ea8c18f93bc9968096d3f015545a8` | `a823bed9d9cbf119c24e54d71b6e8f750e8a6534a4ab06c5ee41228ca3922fd9` |
| `examples/video/manual-reference.json` | `76125afaf1546cd048d40a83358d6afabe9fdcb1d36eae421d8e34177f934b11` | `84207cab39b8b93a1fbbe7dbe380e942550a511eae395e4c3c4470c7f1a3753e` |

## Initial publication checks

Publication checks passed on 2026-10-13: all 40 offline tests passed in 13.010 seconds using the existing development environment (`python -B -m unittest discover -s tests -v`). No Ollama inference was run. The independent scan found no private machine paths or token-like secrets in release files or the 10 members of the existing ZIP. No database, virtual environment, runtime cache or log was included. Published snapshot video/frame hashes, preserved core-code hashes, and updated documentation references were verified. That scan covered the original ZIP before its intentional exclusion from Git. The preserved extracted report and normalized video remain unchanged. The downloader is checked offline using the retained original sample and mocked byte streams; no sample is re-downloaded for publication.
