# 视频理解：复用与样本来源

调研日期：2026-10-13。通过 GitHub 插件检查 README、许可证与关键源码，并核对官方文档和 PyPI 元数据。目标是这台 16 GB RAM、i5-1130G7、无独显的 Windows 电脑上的本地 Ollama 视频理解。

采用 **PySceneDetect 的 CPU 镜头检测 + FFmpeg 视频处理 + 已有 Ollama 调用层**。新增部分限于采样预算、任务与缓存、中文观察/事件结构、证据时间线、人工标注和评测；不重新实现视频解码、场景检测算法或模型推理。

## 已检查的方案

| 组件 / 仓库 | 许可与具体入口 | 取舍 |
| --- | --- | --- |
| [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | [BSD-3-Clause](https://github.com/Breakthrough/PySceneDetect/blob/main/LICENSE)；[AdaptiveDetector](https://github.com/Breakthrough/PySceneDetect/blob/main/scenedetect/detectors/adaptive_detector.py)、[SceneManager](https://github.com/Breakthrough/PySceneDetect/blob/main/scenedetect/scene_manager.py)、[save_images](https://github.com/Breakthrough/PySceneDetect/blob/main/scenedetect/output/image.py) | 直接依赖。传统 CPU 检测，无须额外神经网络模型；支持降采样、镜头起止时间和关键帧。`get_scene_list(start_in_scene=True)` 可为无切镜头视频保留完整场景。场景切换只用于采样，不能直接当成事件发生。 |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) / [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) 分发 | FFmpeg 按[官方说明](https://ffmpeg.org/legal.html)为 LGPL-2.1-or-later，启用 GPL 组件的构建适用 GPL；Python 包装器为 [BSD-2-Clause](https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE)。[get_ffmpeg_exe](https://github.com/imageio/imageio-ffmpeg/blob/main/imageio_ffmpeg/_utils.py) 提供本机可执行文件路径。 | 直接复用解码、缩放、抽帧与预览视频转码。PyPI Windows wheel 附带 FFmpeg，可避免要求用户手工配置系统 PATH；不假定同包附带 ffprobe。具体二进制许可须以实际构建为准，包装器许可不覆盖 FFmpeg 本体。 |
| [starks-design/video-debrief](https://github.com/starks-design/video-debrief) | [Apache-2.0](https://github.com/starks-design/video-debrief/blob/main/LICENSE)；[scene_frames.py](https://github.com/starks-design/video-debrief/blob/main/scripts/scene_frames.py) 的 `detect_scene_changes`、`add_heartbeats`、`extract_all`；[render_report.py](https://github.com/starks-design/video-debrief/blob/main/scripts/render_report.py) 的时间线与播放器定位 | 参考“切镜头 + 长镜头定期间隔补帧 + 帧清单 + 时间定位”的流程，不复制整套应用。README 明确 Ollama 未做真实验证；默认清理证据帧；[describe_all](https://github.com/starks-design/video-debrief/blob/main/scripts/describe_local.py) 的缓存只按整秒时间戳匹配，没有视频/模型/提示词指纹。这些行为不满足可追溯评测要求。 |
| [win4r/VideoFinder-Llama3.2-vision-Ollama](https://github.com/win4r/VideoFinder-Llama3.2-vision-Ollama) | [Apache-2.0](https://github.com/win4r/VideoFinder-Llama3.2-vision-Ollama/blob/main/LICENSE)；[main.py](https://github.com/win4r/VideoFinder-Llama3.2-vision-Ollama/blob/main/main.py) 的 `analyze_video`、`analyze_image` | 不作为实现底座。默认 Llama3.2-Vision、每秒一次推理不适合本机预算；FPS 被转成整数后计算时间，可能漂移；未返回或无法解析置信度时默认满分。仅作为已有同类功能的调研证据。 |

以上两个应用仓库是流程参考，不据此宣称其完整方案已在本机验证。没有把上游项目的演示效果或性能当成本项目结果。

## 依赖元数据与实现边界

本日 [PyPI scenedetect-headless 元数据](https://pypi.org/pypi/scenedetect-headless/0.7.2/json)确认 `0.7.2` 未撤回，要求 Python ≥ 3.10，依赖 `click!=8.3.0,~=8.0`、`numpy`、`opencv-python-headless`、`platformdirs`、`tqdm`。无需启用 `pyav` 或 `moviepy` 扩展。`scenedetect` 与 `scenedetect-headless` 提供相同模块，应只安装一个。

[PyPI imageio-ffmpeg 元数据](https://pypi.org/pypi/imageio-ffmpeg/0.6.0/json)确认 `0.6.0` 未撤回，要求 Python ≥ 3.9，无其他 Python 必需依赖；提供 31,246,824 字节的 `py3-none-win_amd64` wheel。这是发行包信息，不是本机性能测试。

- 镜头边界与定期间隔采样合并、去重后，按明确帧数预算串行送入现有本地 VLM。固定机位长镜头也必须采样。
- 每个证据帧关联源视频哈希、帧编号、实际时间戳和帧哈希；不得把请求的 seek 时间直接冒充已解码帧的实际时间。
- 观察只描述可见内容；单帧状态和由多帧支持的变化分开记录。稀疏采样不能证明目标持续存在，也不能保证找全短暂事件。
- 缓存需要视频/图像哈希、模型 digest、提示词和采样参数。保留证据帧供回看，不使用参考仓库的整秒时间戳缓存。
- 仅连接本地 Ollama；公共样片和依赖的下载需要联网，视频分析本身不需要向远端发送内容。

## 真实世界公开视频

样片为 [Wikimedia Commons: Street traffic.webm](https://commons.wikimedia.org/wiki/File:Street_traffic.webm)，作者 [Editor](https://www.youtube.com/user/Editor)，[原始发布页面](https://www.youtube.com/watch?v=_7sCYyxw4Ic)，许可 [CC BY 3.0 Unported](https://creativecommons.org/licenses/by/3.0/)。Commons 记录在 2013-02-03 审核了原始 YouTube 许可。来源页描述为旧金山街道交通。

源文件名 `Street traffic.webm`；Commons 记录为 1920×1080、35.004 秒、26,490,572 字节，SHA-1 为 `7f916030f14d84a65aa92077339f472897915fef`。原文件不随 Git 仓库分发；运行 `tools/fetch_video_sample.py` 校验下载到 `examples/video/street-traffic.webm`；完整下载 URL、原文件实算哈希、归属和许可保存在同目录 `source.json`。

归属文字：**Street traffic — Editor，CC BY 3.0；来源 Wikimedia Commons。证据图由本项目从原视频抽取并缩放。** 原视频下载文件不转码；抽帧、缩放或预览转码的产物应明确标记为衍生文件。

人工证据只依据本地实际解码并查看的画面，记录在 `examples/video/annotations.json`，与模型输出分开。它们是少量定点观察，不能作为完整视频事件清单或总体准确率的依据。本轮配套工作不运行模型推理，不报告尚未测得的速度或识别质量。

## 本轮实际验证记录

- 原文件下载后，字节数和 SHA-1 与 Commons 记录一致；实算 SHA-256 为 `e5facb24baf755f0c1193999d823e99f6032661ad2bdef5cc28cc1cbbfde03e2`。
- 已用安装后的 `imageio-ffmpeg 0.6.0` 所提供的 FFmpeg 7.1 Gyan essentials 解码并保存 7 张 960×540 证据图。该可执行文件的 `ffmpeg -L` 明确显示 GPL-3.0-or-later；构建配置包含 `--enable-gpl --enable-version3`。
- 抽帧用 `-copyts` 保留输入时间戳，并从 `showinfo` 读取实际 PTS：0、5、10、15、20、25、30 秒，时间基为 1/1000。日志为 `examples/video/extraction.log`，命令、版本、帧路径和每帧 SHA-256 为 `frames.json`。本片探测信息显示 `1k fps, 30 tbr, 1k tbn`，不能把名义 FPS 直接当作可靠的帧序号换算依据。
- 助手已逐张目视查看这些图片，形成 3 条探索性参考观察：0→5 秒前景浅色卡车的画面占比变化；25→30 秒红色头盔骑车人的可见性变化；30 秒道路中景行人与右侧公交车、出租车状态。早期 `annotations.json` 仅对应这组探索性图片，不能拿来评判采样时刻不同的正式流水线。
- 标注由助手目视整理，尚未经用户人工复核；`annotations.json` 如实记录 `human_reviewed: false`。样本调研阶段未运行 Ollama。随后主流程已完成三窗真实推理，耗时与错误见 [VIDEO_VALIDATION.md](VIDEO_VALIDATION.md)，不要把参考标注当成准确率成绩。

## 与正式三窗采样一致的参考

正式导入任务 `23958a06958844548df8a195ef49b005` 的 `plan.json` 已生成三窗；其公开副本（仅替换本地路径）为 `examples/video/pipeline-plan.json`，SHA-256 为 `a823bed9d9cbf119c24e54d71b6e8f750e8a6534a4ab06c5ee41228ca3922fd9`。6 张对应 PNG 已按原字节复制到 `pipeline-evidence/`，并逐张核验与计划内 SHA-256 相符。参考标注为 `manual-reference.json`。

| 时间窗 | 实际证据时间（秒） | 目视支持的状态或变化 |
| --- | --- | --- |
| S0001：0–15 | 3、11.916667 | 前者中央有浅色 SKYLINE 卡车；后者为左侧公交车及中央偏右的小汽车、出租车，卡车在第二帧不可见。 |
| S0002：15–30 | 18、26.916667 | 前者公交车占据左侧与中央近景；后者道路露出更多，中央近景可见红头盔骑车人。不可见不等于不存在，前一帧可能存在遮挡。 |
| S0003：30–35 | 31、33.916667 | 前者中景横向可见 Red Bull 厢式货车；后者该车不可见、前景可见带包行人及另一名穿浅色长裤的行人。不据此建立跨帧身份或连续路线。 |

这些时间来自正式流水线的 **12 fps、从零开始的标准化预览**，按实际解码帧号除以 12 计算；帧号分别为 36、143、216、323、372、407。它们不是原 WebM 名义 FPS 换算值，也不是事件发生的精确边界。标准化预览 SHA-256 为 `35af362b6c2e58fa6d3d8ae1483815805fea0ec61e3296f85248c94702ab8045`。

助手已目视查看这 6 张实际图片；`manual-reference.json` 逐帧记录状态、逐窗记录能确认的变化和不应声称的变化，并绑定源视频、预览、计划与每帧哈希。此参考独立于模型输出、尚未经用户人工复核，也不代表已完成模型准确率评测。旧 `.runtime/video-media-check/plan.json` 后来只选首尾两窗，不能替代这里的三窗快照。
