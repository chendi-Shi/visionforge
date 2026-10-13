# 原始画面语义检索：复用与验证

调研及实现日期：2026-10-13。

## 复用边界

直接复用 [Qdrant FastEmbed 0.9.0](https://github.com/qdrant/fastembed/tree/v0.9.0) 的 ONNX 图文编码器和图像预处理；以 NumPy 计算归一化向量的余弦相似度。这一规模最多 900 张图，不引入独立向量数据库。没有 PyTorch 或 GPU 依赖。

| 组件 | 用途 | 许可／来源 |
|---|---|---|
| FastEmbed 0.9.0 | CPU 编码、分词、图像预处理 | [Apache-2.0](https://github.com/qdrant/fastembed/blob/v0.9.0/LICENSE) |
| Qdrant/clip-ViT-B-32-text | 英文查询，512 维 | [官方模型卡，MIT](https://huggingface.co/Qdrant/clip-ViT-B-32-text) |
| Qdrant/clip-ViT-B-32-vision | 原始视频帧，512 维 | [官方模型卡，MIT](https://huggingface.co/Qdrant/clip-ViT-B-32-vision) |
| OpenAI CLIP ViT-B/32 | 两个 ONNX 编码器的上游模型 | [官方代码与 MIT 许可](https://github.com/openai/CLIP) |
| ONNX Runtime | 两线程 CPU 推理 | [MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| 现有 video_media / OpenCV | 读取已规范化的 12 fps 预览、核对实际帧号 | [视频复用说明](VIDEO_REUSE.md) |

CLIP 是 2021 年的成熟图文检索模型；本轮使用它建立可复核的画面检索基线，不能把它表述为最新 VLM。视频摘要、caption、OCR 文本均不进入视觉索引。匹配的是图片与查询的向量；后续事件判断由调用方另行处理。

## 接口与证据

`video_index.build_index(preview_path, output_dir, interval_seconds=2.0, max_frames=900, progress_callback=None)` 返回可 JSON 序列化的 manifest。回调参数是 `(stage, message)`。

从整个预览时间轴每约 2 秒抽一帧，包含首帧与末帧，超过预算时均匀增大间隔。`sampling` 明确记录请求间隔、是否触发预算、平均有效间隔、实际最大／最小帧间隔、首尾时间和样本数量。最多 900 帧，不把前 900 帧之后的视频直接丢掉。实际时间是经解码位置核验的 `frame_index / fps`，尾帧位于视频结束时间之前。

每张 PNG 保存 `id`、实际帧号、实际时间和 SHA-256。`search_index(index_dir, query_text, top_k=3, min_separation_seconds=4.0)` 返回上述证据字段、图片路径和 `score`。先按余弦相似度排序，再按时间距离贪心去重；同分时取较早帧，刚好相隔指定秒数允许保留，结果可能少于 `top_k`。

这是一组覆盖完整时间跨度的稀疏样本；短于采样间隔的瞬时事件可能完全漏过。相似度只用于排序，不能当作事件成立的概率；即便查询对象不存在，也可能返回相对最接近的图片。固定的 CLIP 图片预处理会缩放并中心裁成 224×224，细小文字、细小目标和画面边缘可能丢失。

英文查询由调用方传入；中文查询需先经本地翻译并保留实际英文输入，便于核查。模块拒绝未翻译的中文以及超过 CLIP 77 tokens 的英文输入，避免静默丢掉末尾目标。离线测试用的注入编码器只检验程序逻辑，不能代表模型效果。

## 固定模型、缓存与资源

模型下载到 `%LOCALAPPDATA%/VisionForge/models`，不进入 Git。首次准备需要访问公开模型仓库；推理在本机 CPU 完成，图片不会发给模型仓库。FastEmbed 使用匹配的固定版本快照：

| 编码器 | 固定 Hugging Face revision | model.onnx 字节数 | SHA-256 |
|---|---|---:|---|
| text | `48ca1db27cb4063eb311ec2aa7f087a808112876` | 254102519 | `4dbe762b11e36488304471e439cde89da053ad7acaddbf9e096745d142ec8d8b` |
| vision | `e0c24ed0fa57fa3e4f97f30de74c51d944036ace` | 351686194 | `c68d3d9a200ddd2a8c8a5510b576d4c94d1ae383bf8b36dd8c084f94e1fb4d63` |

来源：[text 固定快照](https://huggingface.co/Qdrant/clip-ViT-B-32-text/tree/48ca1db27cb4063eb311ec2aa7f087a808112876)、[vision 固定快照](https://huggingface.co/Qdrant/clip-ViT-B-32-vision/tree/e0c24ed0fa57fa3e4f97f30de74c51d944036ace)。权重合计约 606 MB，运行内存另计。每批最多四帧，CPU provider、两线程，不启用多进程复制模型。

索引身份包含预览内容哈希、视频参数、采样设置、索引代码哈希、两侧模型版本及模型／配置／分词文件哈希、相关依赖版本。图像与文本显式 L2 归一化，矩阵以 float32 保存到压缩 NPZ，读取时禁用 pickle。NPZ 中还保存索引身份、帧号和每行图片哈希，防止向量行与证据错位。

复用和查询均核查预览、每张证据图及 NPZ 校验值；模型或输入改变时重建，损坏索引拒绝检索。生成数据写入独立 generation，成功后才原子更新 `index.json`；失败不会把半成品标为成功。旧 generation 保留在本机索引目录，当前实现不自动清理。每个索引有进程文件锁。

默认编码器在构建和多条查询间复用。调用方应把构建、查询与模型释放放在应用共用的 `inference.lock` 内，在 `finally` 调用 `video_index.release_models()`，再进入 Ollama 复核，避免两类模型同时占用内存。释放函数清空默认编码器缓存并触发垃圾回收；不能在其他索引操作仍在使用编码器时调用。调用方注入的编码器由调用方自行管理。

两个函数额外接受仅限关键字的 `embedder=`，可注入实现 `identity()`、`embed_images(paths)`、`embed_text(text)` 的对象做离线测试。生产默认始终使用上述真实 CLIP；下载失败会明确报错，不生成替代向量。

## 验证记录

2026-10-13，17 项离线测试通过，覆盖缓存身份、输入变更、图片／NPZ 损坏、向量行对应、时间戳及路径约束、归一化排序、时间去重、77-token 输入上限、失败后保留上次完整索引及编码器释放。测试使用注入编码器；下面的实测使用通过上述官方 SHA-256 校验的真实 ONNX 权重。

实测输入为 [Street traffic 的 35 秒规范化预览](../artifacts/video-evidence/0354ceb6ccaf4e3eb7857a7368ff0b67/video.mp4)，来源与 CC BY 3.0 署名见 [视频复用说明](VIDEO_REUSE.md)。预览 SHA-256 为 `35af362b6c2e58fa6d3d8ae1483815805fea0ec61e3296f85248c94702ab8045`；960×540、12 fps、420 帧。默认设置得到 19 张 PNG：0、2、4、…、34、34.916667 秒，最大间隔 2 秒；加上尾帧后的平均间隔为 1.939815 秒。

机器为 i5-1130G7、16 GB RAM，CLIP 固定两线程 CPU。第一次并行尝试因整机换页压力主动终止，未产生成功索引，也不计入成绩；下表来自随后取得应用共用推理锁的一次完整运行。其他应用仍在共享机器资源，因此这些数字是该次观测，不是隔离环境的吞吐承诺。公开权重下载已预先完成。

| 操作 | 本次耗时 |
|---|---:|
| 首次完整索引，含编码器准备、内容校验、解码与落盘 | 124.658 秒 |
| 其中 19 帧视觉编码累计 | 19.741 秒 |
| 第一条英文查询，含首次文本编码器加载 | 8.856 秒 |
| 后续四条查询，各含索引与模型校验 | 5.849／3.983／3.471／3.476 秒 |
| 再次构建：核验后复用，无重新编码 | 16.945 秒，`cache_hit=true` |

每条查询取 Top 3，候选时间至少相隔 4 秒。表中的括号数值是原始余弦相似度，保留四位小数；目视备注由开发时查看对应原始 PNG 得出，不是用户标注，也不是独立完整评测集。

| 实际英文输入 | 排名顺序：实际秒数（score） | 原始帧目视核查 |
|---|---|---|
| `a delivery truck with a Red Bull logo` | 32（0.2086）、6（0.2063）、18（0.2043） | 32 秒可见 Red Bull 标识货车；6 秒为普通货车，18 秒主要为公交，后两帧不支持该目标。 |
| `a person wearing a red helmet` | 24（0.2425）、28（0.2341）、34（0.2091） | 24／28 秒清晰可见红头盔人物；34 秒未见该目标。 |
| `a pedestrian crossing the street in front of a bus` | 28（0.2846）、32（0.2842）、18（0.2690） | 28／32 秒可见道路中的行人及公交；18 秒未见对应横穿行人。静态帧不能单独证明完整穿越动作或路径。 |
| `a silver bus filling the left side of the image` | 8（0.2824）、14（0.2790）、4（0.2765） | 三帧均有左侧或左中部公交；14 秒左侧车身占幅最明显。分数不能精确证明“占满左侧”的空间限定。 |
| `an empty countryside road without any vehicles or people` | 34（0.2035）、26（0.1973）、30（0.1967） | 三帧都是有车辆及行人的城市道路，全部不支持查询。这是必须拒绝的负例，检索排名本身不会拒绝不存在的对象。 |

公开的 [真实检索记录与候选图片](../artifacts/visual-retrieval/validation.json) 保留原始分数、帧号、时间、图片 SHA-256、模型身份和耗时，附这五次查询的 11 张唯一候选 PNG。复制后逐一核验 SHA-256 未变；图片路径均相对该 JSON 所在目录。记录完整嵌入样片的来源、作者和许可，并将逐帧备注标为助手目视核查。

本机原始运行记录位于 `VisionForge/video-data/search-validation-20261013/validation.json`，完整索引矩阵及 PNG 保留在该目录；路径相对系统本地应用数据目录。公开附件只含实际候选图片，不发布整个索引或机器路径。进程内存计数未成功取得，因此不报告内存数值。运行结束已释放 CLIP 编码器。

本次结果支持“从原始画面召回候选，再核查证据”的实现路线。它同时暴露普通货车与品牌货车混淆、时间去重后的低相关候选、否定条件失效等问题；不据这五条查询声称整体准确率或完整事件检测能力。
