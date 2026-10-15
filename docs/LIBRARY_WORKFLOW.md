# 视频库、跨视频检索与证据复核

本轮入口为 `app.py`，按“视频库 → 搜索任务 → 证据复核”组织。已有单视频检索仍可打开，原始模型响应与历史实测保留；新增资料整理、持久父子任务、复核版本和整批导出。

## 视频库与搜索范围

在“视频库”中一次选择多个本地视频，点击“导入视频库”。每批最多50个文件，每文件上限1 GB；视频预处理沿用30分钟上限。每个文件独立报告接收或失败，已接收文件继续在后台预处理，不因另一文件失败而丢弃。导入仅复制原文件并生成本地预览与采样资料，**不调用 Ollama**。本地依赖安装好后，Ollama 不在线也能预处理、整理资料和查看已有证据。

点击表格中的视频，可修改显示名称、逗号分隔的标签和归档状态。名称与标签不改变原文件，不参与图像嵌入。归档只把视频移出可选搜索范围，保留文件、索引及既有任务；勾选“显示已归档”后可以恢复。归档不会取消已提交任务，暂停任务需使用任务页的暂停按钮。

名称和标签筛选同时缩小可选范围。“选择当前筛选下的全部视频”只勾选已准备且未归档的视频。筛选变化会保留当前范围内的已选项；提交前可看到所选数量和总核验预算。尚未准备完成、重复或已经归档的视频会被明确拒绝，整次搜索不会只提交其中一部分。

新检索需要启动 Ollama 并选择已安装的视觉模型。即使手动填写英文检索词，后续逐帧核验仍需要模型；当前界面没有单独的“只检索不核验”任务模式。

## 整批预算与持久任务

一次查询分别交给每个选中的视频。每视频直接从 CLIP 画面索引选候选，再由本地 VLM 核验候选附近帧；复用已有 `SearchEngine`、索引和推理互斥锁，按队列串行执行。

这是**持久的跨视频工作流与结果汇总**，没有跨库统一向量排名或全库 Top K。某视频没有“可见”结果，只能说明本次核验范围未得到可见判断，不能推断视频中绝不存在目标。不同视频各用自己的相对时间，不能拼成连续事件轨迹。

| 设置 | 默认值 | 含义 |
|---|---:|---|
| 整次任务最多核验帧数 `max_total_frames` | 48 | 所选视频合计的核验上限 |
| 单视频上限 `max_frames` | 12 | 分配给某视频的帧数还受到整批上限约束 |
| 每视频候选数 `top_k` | 2 | 来自该视频内部的时间去重候选 |
| 补帧范围 `radius_seconds` | 2秒 | 候选中心前后范围 |
| 补帧间隔 `step_seconds` | 1秒 | 在候选范围内规划采样 |

例如选6个视频，整批48帧、单视频12帧，则每个视频最多8帧。实际计划可能因视频长度、候选重叠或去重而更少。预算按选择顺序均分，余数逐个分配；每视频至少需要 `max(3, top_k)` 帧，整批预算不足会拒绝提交。该预算限制 VLM 帧核验，**不包含**视频预处理、CLIP 索引构建和查询转换所耗时间。

`LibraryEngine` 把父任务、全部子检索和它们的映射一次写入同一个 SQLite 事务。中途写入失败会整体回滚；其他客户端不会读到仅有父任务或部分映射的半成品。重启后读取持久映射，不重新派发同一子任务。一次新提交会创建一个新父任务；没有把用户重复提交的相同查询自动合并。

任务页分别显示每个视频的状态、已核验帧数和错误。整批暂停会立即暂停尚未开始的子任务；正在核验的子任务在当前帧结束后保存。继续操作保留完成的子任务和已成功核验的帧，只继续暂停项或重试失败项。程序中断后正在执行的子任务转为暂停，需要明确继续。检索代码、模型、设置或证据变化会触发现有身份校验，不能把新的推理结果混入旧任务；此时可保留历史记录并新建检索。

汇总结果保留 `job_id`、`search_id`、帧 ID、视频内容 SHA-256 和帧内时间。即使不同视频都存在 `QF00000012`，也不会将它们视为同一证据。失败子任务和部分结果仍显示在原父任务下。

## 原判、复核与时间证据

点击任务的某一视频或结果行，会进入该视频的“证据复核”，加载相应预览。点击证据帧可定位时间并查看原图。表格同时显示模型原判、当前判断及判断来源；“待复核”“可见”“不确定 / 失败”筛选用于逐帧检查。

任务必须完成、暂停或处于部分完成/失败状态，才能保存复核。选择“可见”“未见”或“不确定”均须填写画面依据。保存后新增一条版本记录，重算当前时间证据；原 `results`、模型原始响应和 `attempts` 不被覆写。

复核记录有明确来源：

| 来源 | 保存方式 | 显示含义 |
|---|---|---|
| `user` | 用户在界面保存 | 用户复核 |
| `developer_assistant` | 开发验证通过 Python 接口明确标记 | 开发助手复核；不能计为用户审核 |
| 无有效复核 | 原始模型结果 | 模型 · 待复核 |

`ReviewStore` 使用追加式表和逐帧修订号，禁止原地修改、删除历史。保存时提交读到的 `expected_revision`；若另一个页面已保存新版本，本次会提示刷新，避免覆盖他人刚做的复核。它是本地记录来源与版本的机制，没有多人账号或权限系统。

“恢复模型判断”写入一条 `reset` 版本，撤销当前复核覆盖，历史仍可查看。复核绑定目标、模型身份、源视频哈希、实际帧号/时间、证据图片哈希及原始结果哈希。应用前重新核对，界面会明确显示下列情况：

| 状态 | 当前处理 |
|---|---|
| `active` | 复核与原证据匹配，覆盖该帧当前判断并参与重算 |
| `reset` | 使用原始模型判断，保留此前复核历史 |
| `deferred` | 子任务正在执行，暂缓应用复核；暂停或完成后重新校验 |
| `stale` | 目标、模型、原始结果或证据发生变化，拒绝套用旧复核 |

证据图片缺失或哈希改变时，该点作为失败点阻断时间推导。若只是原始结果/身份改变，旧复核仍保留并标记失效；填写原因后 `reset`，才可针对当前证据重新复核。若有效逐帧判读包含不同目标或模型身份，保留逐帧记录并显示警告，不合并推导时间区间。

重算仍使用原来的相邻采样规则：不确定、失败或过大间隔阻断推导。“可见采样组”不证明两点之间持续可见，出现/消失区间不是精确事件边界。未经复核的模型判断可能与复核判断共同参与当前时间证据；修正一帧不会训练模型或自动纠正其他帧。

## 离线证据包

“导出此视频证据包”保存该视频预览、证据图片、原始响应、复核历史、原始区间与复核后区间。导出校验实际资源哈希，遇到损坏资产会报错。

“导出整次任务证据包”提供 ZIP、汇总 CSV 和任务清单 JSON。解压后打开根目录 `index.html`，从每个视频卡片进入其独立报告。包内通过相对路径引用视频、帧图和报告，无需继续运行 Ollama 或 VisionForge。汇总 CSV 包含原判、当前判断、来源、修订号、视频/检索/帧 ID 与哈希；没有把多个视频的时钟拼成一条时间线。缺失的子检索保留错误记录；已有视频但证据损坏时明确中止导出，避免悄悄漏掉证据。

## Python 接口

已有单视频 `video_cli.py`、`search_cli.py` 仍可使用。跨视频任务使用下面的 Python API 或主界面：

```python
from video_jobs import VideoEngine
from search_jobs import SearchEngine
from library_jobs import LibraryEngine
from search_reviews import ReviewStore
from library_report import export_library_search

videos = VideoEngine()
searches = SearchEngine(videos)
library = LibraryEngine(videos, searches)
reviews = ReviewStore(videos.root)

# 导入返回每个文件的接收状态/错误；启动现有视频 worker 做预处理。
receipt = library.import_videos(["clip-a.mp4", "clip-b.mp4"])
videos.start()
print(receipt)
print(library.list_videos())

# 先确认所选视频 ready_for_search=True，再用实际 job ID 提交。
run_id = library.create_search(
    ["VIDEO_JOB_ID_A", "VIDEO_JOB_ID_B"],
    "戴红色头盔的人", retrieval_query="a person wearing a red helmet",
    model="qwen3-vl:2b", max_total_frames=48, max_frames=12,
)
searches.start()
print(library.get_search(run_id))
```

独立脚本必须保持进程运行以承载 worker；主界面运行时也能处理已持久化的队列。上例 ID 是占位符，不能直接作为实际输入。`LibraryEngine.start/stop` 仅代理现有搜索 worker，不创建第二个推理进程。

| 接口 | 返回与用途 |
|---|---|
| `list_videos(include_archived=False)` | 原视频字段加 `title`、`tags`、`archived`、`ready_for_search` |
| `update_video(job_id, title=None, tags=None, archived=None)` | 更新资料并返回当前视频；省略字段保持原值 |
| `import_videos(paths, model="qwen3-vl:2b")` | `job_ids`、逐文件 `items`、`succeeded`、`failed` |
| `create_search(job_ids, target, retrieval_query="", model=..., max_total_frames=48, **config)` | 原子创建父子任务，返回父任务 ID |
| `get_search(run_id)` / `list_searches()` | 状态、`children`、`results`、各视频 `intervals`、进度及错误 |
| `pause(run_id)` / `resume(run_id)` | 原子更新该父任务映射到的子检索，返回当前汇总 |
| `ReviewStore.save(search, frame_id, verdict, note, actor="user", expected_revision=0)` | 追加一次复核并返回新增版本；传入新读取的检索快照 |
| `ReviewStore.apply(search)` | 返回保留原判、增加 `effective_results`、`reviewed_intervals` 和审计字段的副本 |
| `export_library_search(run, reviews, outputdir)` | 返回 ZIP、汇总 CSV、清单 JSON 的路径 |

视频资料和父子映射保存在 `searches.sqlite3`，复核历史保存在 `search-reviews.sqlite3`；与已有视频数据一起位于默认作业目录或 `VISIONFORGE_DATA_DIR` 指定目录。

## 开源参考与复用边界

本轮继续直接使用 FFmpeg、OpenCV、Gradio、FastEmbed、ONNX Runtime 和现有固定 CLIP 权重。具体模型身份、许可及调用边界见[画面检索复用说明](SEARCH_REUSE.md)和[第三方许可](../THIRD_PARTY.md)。没有重新实现解码器、模型编码器或推理服务。

2026-10-15 对以下一手项目资料核查后，采用的是工作流设计参考；本轮没有安装其整套服务、导入其代码或声称与其功能等价：

| 项目 | 参考点 | 本项目的实现范围 |
|---|---|---|
| [MaterialSearch](https://github.com/chn-lee-yumi/MaterialSearch) | 本地素材库、按画面描述查找视频片段 | 视频集合筛选与逐视频画面检索；没有以图搜图或素材目录自动扫描。上游 README 说明前端与核心包分离，部分 API 实现不开放，不能把整个系统描述为本项目已复用的开源核心。 |
| [FiftyOne](https://github.com/voxel51/fiftyone) / [官方 App 文档](https://docs.voxel51.com/user_guide/app.html) | 浏览媒体与标签、发现错误、修正标注的工作流 | 对检索帧逐条复核，保留模型原判和修订来源；没有引入其数据平台或完整标注能力。 |
| [Edit Mind](https://github.com/iliashad/edit-mind) | 本地视频知识库、后台处理、搜索结果回到视频素材 | 父子任务、逐视频状态与独立报告；沿用本地 SQLite 和串行 worker，没有部署其 PostgreSQL、ChromaDB、BullMQ 等服务，也没有增加其音频/人脸分析能力。 |

## 验证边界

离线测试覆盖原子提交/回滚、并发客户端、重启映射、预算、归档恢复、部分失败，以及复核版本冲突、证据绑定、恢复模型判断与导出。测试注入的模型输出只验证程序行为，不能作为视觉准确率。

此前公开的[单视频实测](SEARCH_VALIDATION.md)来自同一段35秒街景，保留真实漏检和错误原因说明。新增视频库与复核功能不会改变那次模型原判，也不会把开发助手复核升级为用户标注或独立基准。稀疏视觉采样、单帧目标判断、无声预览和 CPU 处理速度的限制仍然存在。
