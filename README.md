# VisionForge · 本地视频事件检索

用这台 Windows 笔记本和本地 Ollama，查找视频中某个**可见目标出现的时段**，回看证据，并检查出现/消失的采样区间。

例如输入「戴红色头盔的人什么时候出现？」或「车身有 Red Bull 图案的货车」。系统先直接检索画面，再在候选附近补帧，逐帧判断“可见 / 本帧未见 / 无法判断”。旧版遗漏在摘要里的目标也能进入检索。

## 工作流程

1. **导入视频。** FFmpeg 生成时间轴一致的本地预览。无需先跑旧版逐段摘要。
2. **建立画面索引。** 复用 FastEmbed / ONNX Runtime 的 CLIP 图文编码器，默认每 2 秒采样，覆盖视频首尾。最多 900 帧；超过预算会调整间隔并记录实际覆盖率。向量、模型身份及证据哈希保存在本地并复用。
3. **按目标检索。** 中文查询可由本地 Ollama 转为可检查的可见目标与英文检索词，也可手动填写英文词。直接对画面向量排序，并对相近时刻去重。相似度仅用于选候选，不是发生概率。
4. **局部补取证。** 默认选择 2 个候选中心，各补取前后 2 秒、1 秒间隔的帧；最多核验 12 帧。每帧独立问同一个目标，避免把前帧内容带到后帧。
5. **查看时间证据。** 相邻“本帧未见→可见”或“可见→本帧未见”给出可见性变化区间；不确定、失败和过大间隔会阻断推导。点击证据可定位视频。
6. **保存与导出。** 后台任务支持暂停、重启后继续、仅重试失败帧。模型、代码、设置、视频或证据发生变化时拒绝混用结果。可导出带预览、证据图、原始响应和时间区间的离线 HTML / JSON / CSV / ZIP。

旧版双帧摘要、人工复核和导出仍在折叠区域中。它的文字筛选与新的画面语义检索分开。

## 本机运行

```powershell
git clone https://github.com/chendi-Shi/visionforge.git
cd visionforge

# Python 3.12+；先安装并启动本地 Ollama
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -WithSearch

# 可选：下载公开街景原片并核对大小和 SHA256；上传自己的视频无需此步骤
.\.venv\Scripts\python.exe tools/fetch_video_sample.py

powershell -ExecutionPolicy Bypass -File .\start.ps1
```

打开 <http://127.0.0.1:7860/>。选择已有视频或导入新视频，预处理完成后即可创建目标检索。

首次安装/下载模型需要网络；画面嵌入与 VLM 推理都在本机运行。CLIP 固定权重缓存位于 `%LOCALAPPDATA%/VisionForge/models`，视频、索引和作业位于 `%LOCALAPPDATA%/VisionForge/video-data`，可用 `VISIONFORGE_DATA_DIR` 更改作业目录。上传的画面不会发给在线模型。服务只监听回环地址，不创建公网分享链接。

`-WithSearch` 安装锁定的搜索依赖并下载 `qwen3-vl:2b`。MiniCPM 4.6 1B 也已实测，可用 `ollama pull minicpm-v4.6:1b` 安装后在界面选择。本次四帧诊断中 Qwen 匹配 4/4，MiniCPM 匹配 1/4、另三帧返回不确定，因此默认保留 Qwen；小样本不能代表通用准确率。第一轮画面索引还会下载约 606 MB 的固定 CLIP 双编码器，之后复用缓存。模型下载或校验失败会明确停止，不用假向量或文字搜索替代画面索引。

本项目按 **i5-1130G7 / 16 GB / Iris Xe、CPU** 串行推理设计。画面索引和 VLM 共享推理锁，阶段切换释放模型，避免两个工作流同时争抢内存；其他项目的任务仍会影响本机速度。帧核验可能较慢；预算越大越慢。输入上限 1 GB / 30 分钟，不承诺实时处理。

## 命令行

```powershell
$env:PYTHONUTF8 = '1'
# 只做视频预处理，不自动调用 VLM
.\.venv\Scripts\python.exe video_cli.py import examples/video/street-traffic.webm --max-segments 3 --wait

# 替换 VIDEO_JOB_ID；不加 --wait 时入队，运行中的界面 worker 会处理
.\.venv\Scripts\python.exe search_cli.py create VIDEO_JOB_ID "戴红色头盔的人" --query "a person wearing a red helmet" --wait
.\.venv\Scripts\python.exe search_cli.py list
.\.venv\Scripts\python.exe search_cli.py status SEARCH_ID
.\.venv\Scripts\python.exe search_cli.py pause SEARCH_ID
.\.venv\Scripts\python.exe search_cli.py resume SEARCH_ID --wait
.\.venv\Scripts\python.exe search_cli.py export SEARCH_ID
```

英文检索词可缩短到对象和外观描述，不宜填写长指令。CLIP 的 77-token 上限会显式检查；中文问题的本地翻译也保留，方便发现语义偏差。

## 能力边界与验证

这是**视觉目标检索及采样可见性核验**。单帧无法证明连续动作、身份、意图、因果或速度；画面中“未见”不等于整个视频“没有”。相邻采样点只限定可见性差异，不保证只发生一次出现/消失，也不提供精确事件边界。遮挡或切镜头可能产生可见性变化。稀疏索引仍会漏掉短暂目标；候选外的时间不会被自动宣判为负样本。

没有语音识别或音频检索；预览无声。会议、讲课等依赖音轨的任务不在本轮能力范围内。生成的判断仍需结合原始证据复核。

- [检索架构与复用](docs/SEARCH_REUSE.md)：直接复用 FastEmbed、ONNX Runtime 和固定 CLIP 模型，使用 NumPy 做小规模相似度排序，无需数据库服务。
- [任务与证据架构](docs/SEARCH_ARCHITECTURE.md)：暂停恢复、身份校验、资源互斥及时间区间的推导条件。
- [新版实测与原始记录](docs/SEARCH_VALIDATION.md)：模型对照、真实画面检索、误召回与自动化检查。
- [旧版真实视频验证](docs/VIDEO_VALIDATION.md)与[内容错误复核](docs/VIDEO_CONTENT_REVIEW.md)：保留旧版跨帧误认，不能当成新版成绩。
- [四例固定诊断集](examples/video/grounding-cases.json)：来自同一段公开视频，由助手目视标注，**未经用户审核，不是独立 benchmark**。程序结构通过与内容标签匹配分别报告。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe tools/benchmark_grounding.py --manifest examples/video/grounding-cases.json --models qwen3-vl:2b minicpm-v4.6:1b --max-edge 448 --max-tokens 128
```

旧版完整离线报告仍可在克隆后打开 [report.html](artifacts/video-evidence/0354ceb6ccaf4e3eb7857a7368ff0b67/report.html)。原始 WebM 按需下载，重复的预生成 ZIP 不入 Git。

## 实现入口

|文件|职责|
|---|---|
|video_index.py|完整时间轴采样、CLIP画面索引、缓存校验与时间去重|
|core_grounding.py|单帧可见性核验与相邻采样区间推导|
|search_jobs.py / search_cli.py|持久检索、预算、暂停续跑与模型身份隔离|
|app.py / search_report.py|目标检索界面、证据定位与离线报告|
|video_media.py / video_jobs.py|视频预处理、已有作业及旧版摘要工作流|
|core.py / core_video.py|共用Ollama客户端及保留的双帧摘要|

项目新增代码使用 [MIT](LICENSE)。依赖、模型与公开样片各自保留许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)；尤其当前 FFmpeg 构建启用 GPL。样片 **Street traffic — Editor，CC BY 3.0，来源 Wikimedia Commons**；帧图和无声预览为衍生产物，完整归属见 [source.json](examples/video/source.json)。公开副本的路径清理与历史记录见 [PUBLICATION.md](PUBLICATION.md)。
