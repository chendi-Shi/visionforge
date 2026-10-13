# 来源与复用边界

调研日期：2026-10-13。没有把第三方项目整体复制到此目录，也没有重新实现模型架构、权重加载或 Gradio UI 框架。

|组件|复用方式|许可/来源|
|---|---|---|
|ollama-python 0.6.3|安装官方 SDK；使用 Client、chat、show、list 与图片序列化|[MIT](licenses/ollama-python-MIT.txt)|
|官方 structured-outputs-image.py|参考图片 + JSON Schema + Pydantic 验证的应用模式，替换为中文任务和本机限制|[固定 commit 8785556](https://github.com/ollama/ollama-python/blob/8785556559ec1d045d27a1ba9d9cc7330c3d3cb1/examples/structured-outputs-image.py)|
|Gradio 6.30.0|使用上传、队列、组件、文件下载|[Apache-2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE)|
|Qwen3-VL 2B|Ollama 下载量化预训练权重；不重新分发权重|[模型页及 Apache-2.0](https://ollama.com/library/qwen3-vl:2b)|
|MiniCPM-V 4.6 1B|通过本地 Ollama 进行单帧目标核验；不重新分发权重|[官方模型库](https://ollama.com/library/minicpm-v4.6:1b)、[OpenBMB / Apache-2.0](https://github.com/OpenBMB/MiniCPM-V)|
|FastEmbed 0.9.0|直接复用 CPU 图文双编码器和预处理，不自行实现 CLIP|[Apache-2.0](https://github.com/qdrant/fastembed/blob/v0.9.0/LICENSE)|
|Qdrant CLIP ViT-B/32 ONNX 图像/文本模型|固定 revision、校验权重 SHA256；在用户本地生成画面向量；不把权重提交到仓库|模型卡均标 [MIT：text](https://huggingface.co/Qdrant/clip-ViT-B-32-text)、[MIT：vision](https://huggingface.co/Qdrant/clip-ViT-B-32-vision)；上游 [OpenAI CLIP MIT](https://github.com/openai/CLIP/blob/main/LICENSE)|
|ONNX Runtime|CPUExecutionProvider 执行已有 ONNX 模型|[MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)|
|Pillow|图像解码、EXIF 方向与缩放|[HPND](https://github.com/python-pillow/Pillow/blob/main/LICENSE)|
|Pydantic / httpx|Schema 验证、SDK 超时配置|[Pydantic MIT](https://github.com/pydantic/pydantic/blob/main/LICENSE)、[HTTPX BSD-3-Clause](https://github.com/encode/httpx/blob/master/LICENSE.md)|
|PySceneDetect / scenedetect-headless 0.7.2|直接使用 AdaptiveDetector、SceneManager 和 OpenCV adapter 做 CPU 镜头检测；不复制检测算法|[BSD-3-Clause](https://github.com/Breakthrough/PySceneDetect/blob/main/LICENSE)|
|imageio-ffmpeg 0.6.0|通过 get_ffmpeg_exe 定位 Windows wheel 附带的 FFmpeg，不要求手工配置系统 PATH|Python 包装器 [BSD-2-Clause](https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE)；其许可不覆盖 FFmpeg 可执行文件|
|FFmpeg 7.1 Gyan essentials 实际构建|以独立子进程进行探测、H.264 标准化预览、去音频和抽帧；没有重写编解码器|本机 `ffmpeg -L` 确认为 **GPL-3.0-or-later**，构建含 `--enable-gpl --enable-version3`；FFmpeg 项目的一般许可与构建差异见[官方说明](https://ffmpeg.org/legal.html)|
|opencv-python-headless 5.0.0.93 / OpenCV|解码标准化预览、核验实际帧位置、保存 PNG，并供 PySceneDetect 使用|包装层 [MIT](https://github.com/opencv/opencv-python/blob/main/LICENSE.txt)；OpenCV 本体 [Apache-2.0](https://github.com/opencv/opencv/blob/main/LICENSE)。wheel 的 `LICENSE-3RD-PARTY.txt` 另列所带第三方组件|
|filelock 4.1.1|同一数据目录的视频 worker 跨进程互斥|[MIT](https://github.com/tox-dev/py-filelock/blob/main/LICENSE)，已核对本机安装包元数据与许可证|
|Street traffic 真实公开视频|本地下载原视频，保留来源与哈希；抽取并缩放证据图，标准化预览作为衍生文件|作者 [Editor](https://www.youtube.com/user/Editor)，[Wikimedia Commons 来源](https://commons.wikimedia.org/wiki/File:Street_traffic.webm)，[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)；完整记录见 [source.json](examples/video/source.json)|

上游库许可证由安装分发包保留。项目新增部分是任务编排、参数约束、结果记录、评分和本机启动脚本。自制 smoke 图由本项目程序绘制，不属于真实公开数据集。不同来源不合并宣称为自研 VLM。

视频层具体取舍与参考仓库见 [VIDEO_REUSE.md](docs/VIDEO_REUSE.md)，实际数据流与状态机见 [VIDEO_ARCHITECTURE.md](docs/VIDEO_ARCHITECTURE.md)。Video Debrief 与 VideoFinder 为源码调研和流程比较对象，没有把其整套应用复制进本项目。

公开样片归属：**Street traffic — Editor，CC BY 3.0；来源 Wikimedia Commons。证据图由本项目从原视频抽取并缩放，预览由原视频转码并移除音频。** 原视频文件保持下载字节不变；来源许可、原文件哈希及证据处理记录随样片保留。
