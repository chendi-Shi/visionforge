# VisionForge 复用决策

调研日期：2026-10-13。先通过 GitHub 插件搜索并检查仓库 README、许可证与关键源码，再确定实现方案。

选择 **Ollama 官方 Python SDK + Gradio + Pillow**，在本机运行图片理解应用。项目新增代码限定为应用、数据和评测层；模型、推理引擎、通信 SDK、界面组件和图片解码均复用现有项目。

## 候选比较

| 项目 | 已有能力 | 许可证及采用判断 |
| --- | --- | --- |
| [ollama/ollama-python](https://github.com/ollama/ollama-python) | 官方图片问答、流式响应、JSON Schema 输出；依赖主要是 httpx 和 Pydantic | MIT。采用为推理接口，直接参考官方样例，不自行实现 HTTP 协议或模型加载 |
| [imanoop7/Ollama-OCR](https://github.com/imanoop7/Ollama-OCR) | Streamlit 界面、图片/PDF、批量、自定义提示词、OCR 与多种导出 | MIT。参考上传、预览、批量与导出流程；不引入整套依赖或复制处理器 |
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | 完整 Ollama 聊天平台、会话管理、多模态及知识库 | [当前自定义许可证](https://github.com/open-webui/open-webui/blob/main/LICENSE)，含品牌条件。适合直接使用完整平台；本项目不 fork 全量前后端 |
| [CherryHQ/cherry-studio](https://github.com/CherryHQ/cherry-studio) | Windows 桌面客户端、Ollama、图片附件、中文交互 | [AGPL-3.0](https://github.com/CherryHQ/cherry-studio/blob/main/LICENSE)。适合免开发安装；本项目不扩展完整 Electron 客户端 |

Ollama-OCR 的[处理器源码](https://github.com/imanoop7/Ollama-OCR/blob/main/src/ollama_ocr/ocr_processor.py)存在无请求超时、异常转普通文本使批处理成功统计失真、PDF 页码前缀破坏 JSON 等问题；默认灰度与反色也会丢失通用图片理解所需的颜色信息。[界面源码](https://github.com/imanoop7/Ollama-OCR/blob/main/src/ollama_ocr/app.py)硬编码旧模型列表，[依赖文件](https://github.com/imanoop7/Ollama-OCR/blob/main/requirements.txt)未锁版本且包含本项目不需要的 transformers、OpenCV 等。因此采用官方 SDK 更适合此处的轻量实现。

## 复用边界

- **模型与推理**：由本地 Ollama 执行；不训练模型、不开发推理引擎。
- **图片问答与结构化输出**：依据官方[多模态样例](https://github.com/ollama/ollama-python/blob/main/examples/multimodal-chat.py)和[图片结构化输出样例](https://github.com/ollama/ollama-python/blob/main/examples/structured-outputs-image.py)，通过 SDK 和 Pydantic Schema 完成。直接改用代码时保留上游 MIT 许可与归属。
- **界面与图片处理**：复用 [Gradio](https://github.com/gradio-app/gradio) 的组件和队列、[Pillow](https://github.com/python-pillow/Pillow) 的解码、方向校正和缩放能力。
- **项目新增部分**：中文任务提示词、图片与模型校验、串行任务调度、结果落盘和导出、样本及评测记录。这些是具体使用场景需要的应用、数据和评测层。

## 本机与模型基线

本机检查结果：Windows、Intel i5-1130G7、16 GB RAM、Iris Xe 集成显卡，无独立显卡；Ollama 服务版本为 **0.34.4**。

首选本地模型 **`qwen3-vl:2b`**。[Ollama 模型页](https://ollama.com/library/qwen3-vl:2b)标注约 **1.9 GB**、**Q4_K_M** 量化，并要求 **Ollama ≥ 0.12.7**；本机版本满足该要求。1.9 GB 是模型下载体积，不代表运行时总内存。此选择优先控制单机资源占用；中文识别质量、延迟和内存峰值需以本机样本评测确认，本文不报告尚未测得的性能。

推理连接固定到本机 `http://127.0.0.1:11434`，仅使用下载到本机的视觉模型；不选择 `*-cloud` 模型或远程推理端点。模型下载需要联网，图片推理无需云端服务。界面部署采用本机监听和关闭公开分享的配置。
