# VisionForge · 本地视频证据库

用这台 Windows 笔记本和本地 Ollama，管理一组视频，搜索**可见目标出现的时段**，回到原始画面复核，并导出包含判断依据的证据包。

例如给一组路口视频打上标签，勾选本次范围，输入「戴红色头盔的人什么时候出现？」。系统对每个视频检索画面、在候选附近补帧，再逐帧判断“可见 / 本帧未见 / 无法判断”。发现误判后可以填写依据修正；时间证据随之重算，模型原始输出和全部复核版本仍保留。

[视频库使用与接口说明](docs/LIBRARY_WORKFLOW.md) · [跨视频与复核验收](docs/LIBRARY_VALIDATION.md) · [历史单视频实测](docs/SEARCH_VALIDATION.md)

![视频库：批量导入、筛选与限定搜索范围](artifacts/video-library.png)

## 工作流程

1. **视频库。** 批量导入本地文件，后台生成时间轴一致的无声预览。预处理不需要 Ollama 在线。可改名称、加标签、筛选和归档；归档后仍保留原文件与已有证据。勾选已经准备好的视频作为搜索范围。
2. **搜索任务。** 一次提交目标，为所选视频建立持久子任务，逐个检索和核验。默认整次任务最多核验 **48 帧**，单视频最多 **12 帧**；界面显示本次总预算。每个视频的状态、进度和错误分别展示，支持整批暂停、继续及重试失败项。
3. **证据复核。** 点击结果切换到对应视频、时间和原帧，比较“模型原判”和“当前判断”。填写依据后可确认、纠正、标记不确定，或撤销当前复核。每次保存新增版本并重算时间证据；开发助手复核与用户复核分别标记。
4. **导出交付。** 导出单视频报告，或把整次任务打包为离线 ZIP；跨视频索引页链接各视频报告，包含无声预览、证据图、原始响应、复核历史和汇总 CSV。每个视频保留独立时间轴。

画面索引复用 FastEmbed / ONNX Runtime 的固定 CLIP 双编码器，默认每 2 秒采样、每视频最多 900 帧，超限时均匀调整间隔。中文查询由本地 Ollama 转为可检查的目标与英文检索词；也可填写英文词。默认每视频选择 2 个候选，补取前后 2 秒、1 秒间隔的帧，再按预算核验。候选来自原始画面向量，旧版摘要文字不进入索引。

跨视频搜索目前是**同一查询分派到所选视频，再汇总核验结果**；各视频独立排序，没有全库统一 Top K。相似度只是候选排名，不能当成事件概率。相邻采样点给出可见性变化区间；不确定、失败和过大间隔会阻断推导。

历史单视频检索可直接在“证据复核”页打开。旧版完整界面保留为 `legacy_app.py`，包含双帧摘要等功能；需要时单独启动，不能与主界面同时占用 7860 端口。旧 CLI 继续可用。

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

打开 <http://127.0.0.1:7860/>，按“视频库 → 搜索任务 → 证据复核”使用。导入完成后勾选范围，提交目标即可。Ollama 暂不可用时仍可整理视频、预处理和复核已有证据；新检索的模型核验需要已安装的本地视觉模型。

首次安装/下载模型需要网络；画面嵌入与 VLM 推理都在本机运行。CLIP 固定权重缓存位于 `%LOCALAPPDATA%/VisionForge/models`，视频、索引和作业位于 `%LOCALAPPDATA%/VisionForge/video-data`，可用 `VISIONFORGE_DATA_DIR` 更改作业目录。上传的画面不会发给在线模型。服务只监听回环地址，不创建公网分享链接。

`-WithSearch` 安装锁定的搜索依赖并下载 `qwen3-vl:2b`。MiniCPM 4.6 1B 也已实测，可用 `ollama pull minicpm-v4.6:1b` 安装后在界面选择。2026-10-13 四帧诊断中 Qwen 匹配 4/4，MiniCPM 匹配 1/4、另三帧返回不确定，因此默认保留 Qwen；小样本不能代表通用准确率。第一轮画面索引还会下载约 606 MB 的固定 CLIP 双编码器，之后复用缓存。模型下载或校验失败会明确停止，不用假向量或文字搜索替代画面索引。

本项目按 **i5-1130G7 / 16 GB / Iris Xe、CPU** 串行推理设计。画面索引和 VLM 共享推理锁，阶段切换释放模型，避免两个工作流同时争抢内存；其他项目的任务仍会影响本机速度。帧核验可能较慢；预算越大越慢。输入上限 1 GB / 30 分钟，不承诺实时处理。

## 命令行与 Python 接口

原有 CLI 处理单视频；跨视频任务通过界面或 `LibraryEngine` Python 接口使用，完整示例见[视频库说明](docs/LIBRARY_WORKFLOW.md#python-接口)。

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

2026-10-15：110 项离线测试通过；同一公开样片的两个独立片段完成跨视频实跑，共 6 次真实本地核验。浏览器验证了跨视频定位；真实 32 秒漏检经过修正、撤销、恢复修正，原始记录哈希保持不变。[报告、界面与验收数据](docs/LIBRARY_VALIDATION.md)。这不是独立数据集准确率评估。

这是**视觉目标检索及采样可见性核验**。单帧无法证明连续动作、身份、意图、因果或速度；画面中“未见”不等于整个视频“没有”。相邻采样点只限定可见性差异，不保证只发生一次出现/消失，也不提供精确事件边界。遮挡或切镜头可能产生可见性变化。稀疏索引仍会漏掉短暂目标；候选外的时间不会被自动宣判为负样本。

没有语音识别或音频检索；预览无声。会议、讲课等依赖音轨的任务不在本轮能力范围内。生成的判断仍需结合原始证据复核。

2026-10-13 的单视频中文端到端实测完成10帧及服务重启恢复。**其中仍有真实漏检：模型认出31秒的品牌货车，却漏掉32秒，导致错误的消失区间。** [历史报告、原始响应和错误复核](docs/SEARCH_VALIDATION.md)保持原样公开。本轮新增复核流程能记录纠正并更新当前时间证据，不能据此宣称模型准确率提高。

复核仅在任务完成、暂停或停止于失败状态时生效。继续执行期间暂缓应用，完成后重新校验证据；目标、模型、原始结果或图片内容变化时，旧复核会显式标记失效。若不同模型身份混入当前判读，逐帧记录保留，但不合并推导时间区间。复核不会训练模型，也不会自动纠正未复核的其他帧。

- [视频库、跨视频任务与复核](docs/LIBRARY_WORKFLOW.md)：预算分配、原子提交、复核审计、导出与复用边界。
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
|library_jobs.py|视频资料、批量导入、跨视频父子任务与整批预算|
|search_reviews.py|追加式复核历史、证据绑定与时间证据重算|
|app.py / workbench.py|视频库、搜索任务、证据复核三个页面及显示逻辑|
|search_report.py / library_report.py|单视频报告和多视频离线证据包|
|legacy_app.py|保留的旧版完整界面，按需单独启动|
|video_media.py / video_jobs.py|视频预处理、已有作业及旧版摘要工作流|
|core.py / core_video.py|共用Ollama客户端及保留的双帧摘要|

项目新增代码使用 [MIT](LICENSE)。依赖、模型与公开样片各自保留许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)；尤其当前 FFmpeg 构建启用 GPL。样片 **Street traffic — Editor，CC BY 3.0，来源 Wikimedia Commons**；帧图和无声预览为衍生产物，完整归属见 [source.json](examples/video/source.json)。公开副本的路径清理与历史记录见 [PUBLICATION.md](PUBLICATION.md)。
