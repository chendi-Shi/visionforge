# VisionForge · 连续机器人视频的失败检测与时间定位

用一台 Windows / 16 GB / CPU 电脑和本地 Ollama，研究一个具体问题：**同一模型、同一相机、同样12帧预算下，按机器人运动状态抽帧，是否比等间隔抽帧更容易找到失败发生的位置？**

数据复用作者公开的 [FailTime](https://huggingface.co/datasets/ChangUoA/FailTime/tree/07aea9142616cf1f8047512bd53a4d9fcd5e7cee)：连续 ALOHA / SO-101 视频、同步机器人状态和作者人工失败起点。已核验19条原文件，3条用于开发；固定测试为8个任务的16条记录，每个任务1成功、1失败。开发任务与测试任务分离，标签不进入模型提示词或抽帧器。

这是**完整录像上的离线诊断**。模型看到按时间排列的12帧，输出成功、拒答或一张证据帧；程序将帧ID映射回实测时间。它不等同于实时预警、因果诊断或机器人控制。

[实验协议](docs/ROBOT_TEMPORAL.md) · [双模型对照](artifacts/robot-temporal-study/report.html) · [Qwen 视频证据](artifacts/robot-temporal/report.html) · [MiniCPM 视频证据](artifacts/robot-temporal-minicpm/report.html)

HTML 报告需克隆后在本地打开。每条记录保留原始响应、实际输入拼图、作者标签和可按帧定位的完整复核视频。

## 固定比较与结果

两模型为 `minicpm-v4.6:1b` 与 `qwen3.5:4b`，各运行16条×2种采样器，64次预定测试调用已全部完成。模型串行，温度0、seed 0、context 8192、输出上限128 tokens、4个CPU线程。测试前固定样本、提示词、采样算法、模型身份和评价源码；不依据测试成绩修改协议。型号不代表能力，也不宣称最新或最强。

Qwen 的[原始模型清单与来源记录](artifacts/model-provenance/qwen3.5-4b/README.md)另行公开，区分 API 模型摘要和 Ollama 本地包装文件；没有以同名的其他量化权重替代固定模型。

跨模型结果仅比较这两个冻结的本机配置。参数规模、视觉编码器和模板不同，不能据此隔离架构因素或推断整个模型系列的优劣；两份运行元数据均标记为 `Q4_K_M`。

| 方法 | 固定操作 |
|---|---|
| `uniform` | 在实测首末时间之间等间隔选择12个最近帧 |
| `state_event` | 6个等间隔锚点，加最多6个运动状态变化峰；不足时按最大时间间隙补齐 |
| `clock_midpoint` | 不读取标签，全部判失败，位置取整段实测时间中点 |
| `state_peak` | 不读取标签，全部判失败，位置取最大的运动变化分数 |

状态分数由每只手臂自身坐标内的位置加速度、夹爪变化和固定平滑构成；它是选帧依据，不是失败概率。两种模型收到相同采样条件下完全相同的拼图字节。

| 模型 / 方法 | 正确数 | 失败召回 | 成功误报 | **Hit@2s** | 有效定位覆盖 | 条件 MAE |
|---|---:|---:|---:|---:|---:|---:|
| MiniCPM · uniform | 8/16 | 0/8 | 0/8 | **0/8** | 0/8 | 无定义 |
| MiniCPM · state_event | 8/16 | 0/8 | 0/8 | **0/8** | 0/8 | 无定义 |
| Qwen · uniform | 8/16 | 8/8 | 8/8 | **0/8** | 8/8 | 13.60秒 |
| Qwen · state_event | 9/16 | 8/8 | 7/8 | **1/8** | 8/8 | 11.79秒 |
| 无模型 · clock_midpoint | 8/16 | 8/8 | 8/8 | **1/8** | 8/8 | 13.56秒 |
| 无模型 · state_peak | 8/16 | 8/8 | 8/8 | **1/8** | 8/8 | 19.53秒 |

**本轮没有建立可靠的失败定位能力。** MiniCPM 的32条输出全部判成功；Qwen 等间隔采样全部判失败，状态抽帧也有7/8成功误报。后者的 Hit@2s 命中数为1/8，与两个简单控制相同，不代表命中了同一轨迹。有效定位覆盖只表示真实失败被检出且提供了合法时间，不能解释为定位正确率。

Qwen 内部的状态抽帧相对等间隔抽帧，Hit@2s 差值为 +12.5 个百分点，95%任务簇区间为 [0, +37.5]，包含零；MiniCPM 两方法都为零。这个小样本结果不足以建立稳定的采样收益。

64条响应全部可解析，`done_reason=stop`，没有达到128-token输出上限。MiniCPM 累计模型调用耗时24.04分钟，Qwen 67.18分钟；总计91.22分钟，不包含下载、视频解码、抽帧和导出。单次 Qwen 调用76.66–146.76秒，不能作为实时方案。

主要指标 **Hit@2s = 检出失败且有效定位误差≤2秒的条数 / 全部8条真实失败**。漏检、拒答、调用报错和格式无效都留在分母。失败召回与成功误报同时报告；条件 MAE 只对检出且有效定位的失败计算，必须结合定位覆盖率阅读。采样器最近帧覆盖仅衡量时间分辨率，不代表那一帧一定能看见失败。

这是从官方 test 来源固定选取的小规模研究子集，不是完整官方榜单。成功/失败分层抽样改变了自然比例；来源调研曾接触部分作者原因文本，不能称严格双盲。作者定义的失败起点可能早于明显视觉后果。按8个任务簇配对 bootstrap 的区间只作描述，不作显著性或部署保证。具体定义与选样规则见[协议](docs/ROBOT_TEMPORAL.md)。

开发阶段的错误也保留：旧输出格式在6次调用中产生类别与帧ID矛盾；测试前改为联合枚举后，同3条开发轨迹×2个采样器的6次调用均有效，却仍全部判断成功，每个采样器失败召回均为0/2。格式修复没有建立失败检测能力；[两版开发证据与原始响应](artifacts/robot-temporal-development/README.md)独立保存，不混入测试分母。

## 先审阅，再运行

克隆后可直接打开上述 HTML，不需要模型或联网。以下命令只使用 Python 标准库，验证完整文件清单、冻结输入身份、原始响应解析并重算全分母指标：

```powershell
git clone https://github.com/chendi-Shi/visionforge.git
cd visionforge
python -I -S -B artifacts/robot-temporal-minicpm/verify.py
python -I -S -B artifacts/robot-temporal/verify.py
python -I -S -B artifacts/robot-temporal-study/verify.py
```

校验器从已保存的抽帧、状态分数和响应重算指标，不重新运行模型或 Gaussian/峰检测。原始推理源码与数据身份另行冻结，支持带依赖的重新运行。SHA256 检查内部一致性，不是作者签名或泛化能力证明。

首次运行使用已测试的 Python 3.12，先安装并启动本地 Ollama。以下命令建立包含界面和视频依赖的项目环境；模型和数据首次下载需要网络，推理时图片不发送给在线模型。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-robot-lock.txt -r requirements-temporal.txt
ollama pull minicpm-v4.6:1b
ollama pull qwen3.5:4b

# 固定的19条选择；源 HDF5 合计约1.19 GB，另需解码后的数据空间
.\.venv\Scripts\python.exe -m robotbench.temporal_data --plan artifacts/robot-temporal-minicpm/dataset_snapshot/plan.json --dest data/temporalbench

# 每个模型使用新目录。顺序执行，不并发占用本机模型服务。
.\.venv\Scripts\python.exe -m robotbench.temporal freeze --data data/temporalbench --out data/my-temporal-minicpm --model minicpm-v4.6:1b --split test
.\.venv\Scripts\python.exe -m robotbench.temporal run --data data/temporalbench --out data/my-temporal-minicpm --split test
.\.venv\Scripts\python.exe -m robotbench.temporal_report report --data data/temporalbench --run data/my-temporal-minicpm

.\.venv\Scripts\python.exe -m robotbench.temporal freeze --data data/temporalbench --out data/my-temporal-qwen --model qwen3.5:4b --split test
.\.venv\Scripts\python.exe -m robotbench.temporal run --data data/temporalbench --out data/my-temporal-qwen --split test
.\.venv\Scripts\python.exe -m robotbench.temporal_report report --data data/temporalbench --run data/my-temporal-qwen
```

打开各运行目录中的 `report.html`。中断后重复同一 `run` 命令可继续未完成项；已有失败输出保留，不自动重试覆盖。改变源码、数据、依赖、服务器或模型 digest 需要新运行目录。未来同名模型 tag 可能变化，新运行会记录新摘要，不能自动称为已发布固定权重的精确重现。首次安装命令按声明的依赖编写；本机已有环境的测试通过不等于全新环境安装已验证。

本项目使用 i5-1130G7 / 16 GB / Iris Xe 的 CPU 环境，不训练基础模型或控制策略，也不承诺实时。数据和运行缓存位于已忽略的 `data/`；模型权重、用户上传和原始大型 HDF5 不随 Git 发布。

## 前序实验与保留的负结果

之前的 Guardian UR5-Fail 实验只包含动作前后状态，每条各3个视角；140条标注复用部分观测，不是140次独立 rollout。它没有连续轨迹或失败起点，不能与本轮秒级定位成绩混为一谈。

| 已完成实验 | 保留的结论与证据 |
|---|---|
| 140条、16基础任务组留一验证 | 动作后线性方法63.57%，前后变化67.86%；差值+4.29个百分点，任务簇区间[-1.32,+9.38]跨零。后者仍漏掉31/71个真实失败。[主报告](artifacts/robot-benchmark/index.html) |
| 四粗组后验压力测试 | 动作后与前后变化都为58.57%，原收益不稳定；没有用这个后验结果替换主表。[协议与指标](docs/ROBOT_BENCHMARK.md#独立后验压力测试四个粗分组) |
| 同观测、相反指令的17对诊断 | 动作后与前后变化都只有3/17对同时答对；配对复用样本，不能称17次独立实验。[错误分析](docs/ROBOT_BENCHMARK.md) |
| 指令条件关系核 | 嵌套训练折选参后，16组关系核55.71%，低于动作后核60.00%；粗组关系核失败召回81.69%，但成功召回仅30.43%。全部预设方法保留。[完整负结果](docs/ROBOT_RELATIONAL.md) |
| 原 Qwen3-VL 2B 输入消融 | 16条×3条件的48次输出全部判成功；每条件准确率50%、失败召回0/8。交换顺序没有改变预测，原标签只用于输入损坏诊断。[原始48次响应](artifacts/robot-benchmark/predictions) |
| 四例协议诊断 | 反转枚举未改变全成功输出；去掉前缀的4次调用只有 thinking、无最终答案，均按失败调用计为弃权；MiniCPM也全判成功。[独立诊断与审计](docs/ROBOT_VLM_AUDIT.md) |

旧 UR5 的数据分组、特征交互和后验方法设计仍有限制；嵌套 CV 不会把已看过的140条变成独立确认集。[完整实验说明](docs/ROBOT_BENCHMARK.md)继续提供所有指标、区间、复算命令和错误例。历史原判不因后续实验而被改写。

## 视频证据库与界面

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

打开 <http://127.0.0.1:7860/>，查看只读连续轨迹实验、作者时间标签、输入拼图和模型原判。尚未导出的实验明确显示为空。界面也保留视频库：批量导入、按名称/标签筛选、限定范围搜索、回到视频证据、追加人工复核及离线导出，见[视频库说明](docs/LIBRARY_WORKFLOW.md)。

更新服务版本后请刷新已打开的页面。后台轮询已修复旧选项反复报错和覆盖新勾选的问题；相关真实会话回归与浏览器验收见[修复记录](docs/LIBRARY_VALIDATION.md#服务更新后的会话恢复修复)。

视频库用固定 CLIP 检索，可选择本地已安装的视觉模型核验；历史默认模型为 `qwen3-vl:2b`，它的旧实验不会被改写成新模型结果。首次视觉索引会下载约606 MB CLIP 权重。全批默认最多核验48帧、每视频最多12帧；目前各视频独立排序，没有全库统一 Top K。相似度不是概率，稀疏帧“未见”不能解释为整段“没有”，音频不参与判断。服务只监听本机回环地址。

历史街景实测中的32秒漏检与人工复核记录仍公开；复核修正证据记录，不会训练模型或证明准确率提高。[视频库验收](docs/LIBRARY_VALIDATION.md) · [旧实测与已知错误](docs/SEARCH_VALIDATION.md)。旧完整界面 `legacy_app.py` 按需单独启动，不能与主界面同时占用7860端口。

## 复用与实现入口

| 入口 | 职责与复用 |
|---|---|
| `robotbench/temporal_data.py` | 固定 FailTime 来源、校验 HDF5、分离模型输入与作者标签；复用 h5py / NumPy / OpenCV |
| `robotbench/temporal.py` | 固定抽帧、实测时间映射、原始拼图与 Ollama 调用；复用 SciPy 平滑/峰检测、Pillow |
| `robotbench/temporal_evaluation.py` | 全分母检测/定位、标签无关控制、配对任务簇区间；标准库实现 |
| `robotbench/temporal_report.py` / `temporal_comparison.py` | 完整视频证据、双模型比较、源码快照、离线复算；复核视频使用 FFmpeg |
| `robotbench/probe.py` / `relational.py` | 前序固定 CLIP 特征的线性与关系核实验；复用 scikit-learn |
| `app.py` / `robotbench/temporal_ui.py` | 只读实验审阅与证据时间定位 |
| `video_index.py` / `library_jobs.py` / `search_reviews.py` | FastEmbed / ONNX Runtime 索引、SQLite 持久任务、追加式复核；无需向量数据库服务 |

本轮问题受 [ProTracer](https://github.com/Chang-AIML/ProTracer) 的状态辅助失败诊断启发，使用其公开数据；没有复制其无明确开源许可的实现，也不声称重现论文的完整系统。具体复用边界见[时序协议](docs/ROBOT_TEMPORAL.md)和[检索架构](docs/SEARCH_REUSE.md)。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

测试检查实现和完整性，模型能力以实际保留的预测与指标为准。项目新增代码为 [MIT](LICENSE)；FailTime 为 CC BY 4.0、Guardian 数据为 Apache-2.0，模型和依赖保留各自许可。完整作者归属、转码说明及 FFmpeg 的 GPL 边界见 [THIRD_PARTY.md](THIRD_PARTY.md)、各证据包的归属文件和 [PUBLICATION.md](PUBLICATION.md)。
