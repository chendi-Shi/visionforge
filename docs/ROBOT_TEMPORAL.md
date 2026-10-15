# 连续机器人轨迹：失败检测与起点定位

研究问题：**相同模型、相同相机、相同12帧预算下，使用机器人运动状态选择观察时刻，是否比等间隔抽帧更容易找到失败发生的位置？**

这是完整录像上的离线诊断。系统可以看到整段视频的状态序列和后续画面，不能据此宣称在线预警、实时控制或因果识别。模型输出必须落在实际观察过的帧上；没有让语言模型任意编一个秒数。

## 数据、独立标注与选择

复用 [FailTime 数据集](https://huggingface.co/datasets/ChangUoA/FailTime/tree/07aea9142616cf1f8047512bd53a4d9fcd5e7cee)，固定 revision `07aea9142616cf1f8047512bd53a4d9fcd5e7cee`，许可 CC BY 4.0。它由 [ProTracer](https://github.com/Chang-AIML/ProTracer) 工作发布。FailTime-Long 官方提供100条真实 ALOHA / SO-101 连续轨迹，57失败、43成功、13个任务；视频和机器人状态包含在每条独立 HDF5 中。

失败起点来自作者的人工标注，不是本项目根据 reward、模型响应或视频最后一帧生成的标签。[论文](https://arxiv.org/html/2609.21369v1)将其定义为事后判断的、最终导致失败且未获恢复的最早偏离；暂时发生而后恢复的错误不算这个起点。错误顺序、错误对象的起点可能早于明显的视觉后果。成功样本没有失败时间。只用一个头部视角的模型，未必能看到作者同步审阅多视角与状态时使用的全部证据。

FailTime-Long 只有官方 test split，没有 Long train/calibration。这里明确是从官方 test 来源构建的**本地小规模固定研究子集**，不是官方完整榜单，也不将另一个 FailTime-Short 的 calibration 冒充 Long 的训练集。

- 开发：3条已经查看过的媒体，来自 `block_assembly` 与 `uncap_marker_draw_recap` 两个任务。
- 测试：排除这两个开发任务的所有轨迹，在余下任务中按固定 seed 的 SHA256 排序选8个任务；每个任务按独立样本哈希选1成功、1失败。共16条、8成功/8失败，10条 ALOHA、6条 SO-101。
- 选择不使用失败时间、类型、困难程度或模型成绩。来源调研中读到过部分作者原因文本，因此不声称严格双盲。
- 分层抽样改变了自然成功/失败比例；不能把此处准确率或误报率直接当作部署环境的性能估计。

选择计划在首次测试推理前冻结，规范化 SHA256 为 `0dcc9022ae06cf6958859b46014c94528c9138ffa1bf6ba845aa52d0f37f7c23`。已下载并适配全部19条：测试原文件1,109,569,728字节，开发81,996,969字节；每条文件核验作者 HF LFS SHA256。标签与任务 XML 分别对照固定版本的上游文件。适配后的预测输入和标签存放在不同文件，`load_inputs` 不打开标签或选择计划。

每个相机的帧数、帧率和独立时间序列必须与机器人状态对齐。开发数据的9路视频已全帧解码；测试视频先机械检查首、中、末帧，推理抽帧再按帧顺序解码，避免随机 seek 静默偏移。计分采用记录中的实测时间戳。SO-101 两只手臂的位姿分别属于各自基座坐标系，不计算跨臂世界距离；夹爪数值保持原始关节单位，不称作厘米。

## 方法与计算预算

两种方法都只使用原始头部/总览相机，每条输入为按时间排列的12帧拼图。每格320×240，另有帧ID与实测时间标头；JPEG quality 92、无色度子采样。输出位置引用拼图帧ID，再由程序映射回原记录的时间戳。腕部相机仍在本地数据中，本次固定实验没有使用它们。

| 方法 | 固定操作 |
|---|---|
| `uniform` | 在实测首末时间之间等间隔取12个最近帧 |
| `state_event` | 6个等间隔锚点，加最多6个状态变化峰；不足时按最大时间间隙补齐 |
| `clock_midpoint` | 不看标签，所有轨迹都判失败，时间取实测首末中点 |
| `state_peak` | 不看标签，所有轨迹都判失败，时间取最大的运动变化分数，平分取最早 |

状态变化分数由每只手臂自身坐标内的位置差分加速度和夹爪差分速度构成，各通道按整条记录的95百分位归一化；稀疏变化的分母为零时退回最大值，静止通道保持零。先做0.12秒尺度的平滑，再取通道最大值。候选为局部峰，与已选帧至少间隔0.75秒。它是**运动事件采样分数，不是学习得到的失败概率**。整段平滑、归一化和峰选择都使用未来信息，所以明确属于离线方法。

固定比较两个本地模型：`minicpm-v4.6:1b` 与 `qwen3.5:4b`。每个模型16条×2采样器，两模型合计64次测试调用；模型串行运行。温度0、seed 0、context 8192、输出上限128 tokens、4个CPU线程。实际模型 digest、服务器版本、依赖版本和源码 SHA 保存在各运行计划中。型号不是能力结论，也不宣称绝对最新或最强。

跨模型结论仅针对这两个冻结的本机配置。参数规模、视觉编码器和模板不同，不能隔离架构因素或推断整个模型系列优劣。两份运行元数据均为 `Q4_K_M`；Ollama 的参数量字段也不能自动视作统一口径的完整视觉语言模型参数总数。

这是受 ProTracer 的状态辅助诊断问题启发的独立轻量适配，没有复制其无明确开源许可的实现，也没有重现其完整多轮诊断、经验学习或原论文模型配置。解码复用 OpenCV/FFmpeg，峰检测与平滑复用 SciPy，本地模型执行复用 Ollama。

## 开发阶段协议修复

原开发协议独立生成 `prediction` 和 `frame_id`，要求非失败时 frame=-1。MiniCPM 在3条开发轨迹的6次调用中返回了诸如 success+frame11、abstain+frame0 的矛盾组合。严格解析将其保留为无效响应，没有静默改成正确输出。

测试前将协议改为一个联合枚举：`success`、`abstain` 或 `failure_at_frame_00` 到 `failure_at_frame_11`。程序从它派生类别、帧ID和时间。两种模型与两种采样器使用同一冻结格式。它有12个失败位置与各1个成功/拒答选项，不同类别的生成路径并不对称，因此不能把它与旧格式的类别分布差异归因于视觉理解提升。

完整[开发证据包](../artifacts/robot-temporal-development/README.md)保留两版各6条原始响应、原字节拼图、计划和源码快照，并保留一份因适配源码变化而被身份检查阻止、0次调用的中间计划。新格式6次调用均有效，却全部预测成功：每个采样器在3条开发轨迹上正确1条，失败召回0/2、Hit@2s 0/2。格式修复没有建立失败检测能力；这里的3条轨迹也不是6个独立样本。

开发结果不混入测试分母；测试结果不能用于修改当前研究的提示词、阈值、采样器或模型选择。跨模型研究计划另行绑定采样源码、评价源码、固定成员和预定义比较。开发包中的 `evaluation_sources` 是事后核验所用的评价实现，原推理源码单独保存在各版的 `prediction_sources` 中，未改写历史响应。

## 如何读指标

主要定位指标为 **Hit@2s：判对失败且有效时间误差不超过2秒的条数 / 全部8条真实失败**。同时报告失败召回、成功误报、拒答和 ±1秒/±5秒命中率。缺失、报错、格式无效、漏检都保留在固定分母里。

条件 MAE 只在“真实失败、模型检出、时间有效”的子集上统计，必须与有效定位覆盖率一起看。一个只答出一条的模型不能凭较小 MAE 胜过覆盖更多失败的模型。配对比较另给共同有效定位子集的 MAE 差值。

`sampler_oracle_at_*` 是用作者起点计算最近已选帧的时间覆盖率，只衡量采样时间分辨率。它不表示该帧一定能看见失败，也不是模型成绩。

区间使用固定预测上的任务簇配对 bootstrap：seed 1337、2000次，8个任务为重采样单位。数据量小、任务簇少，区间仅作描述，不作显著性或部署保证。模型间比较在相同采样器、相同16条输入和相同作者标签上进行。

## MiniCPM 已完成的测试结果

2026-10-15，固定 `minicpm-v4.6:1b` 的 **32/32次测试调用全部完成**。两种采样条件均有16/16条有效输出，却都把全部16条记录判断为成功：各答对8条成功，漏掉全部8条失败。没有拒答或格式失败，但这不等于完成了失败检测。以下结果来自[公开评价文件](../artifacts/robot-temporal-minicpm/evaluation.json)，运行身份见[冻结计划](../artifacts/robot-temporal-minicpm/plan.json)。

| 方法 | 正确 / 全部 | 失败召回 | 成功误报 | Hit@1s | Hit@2s | Hit@5s | 有效定位覆盖 | 条件 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM `uniform` | 8/16 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 未定义 |
| MiniCPM `state_event` | 8/16 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 未定义 |
| `clock_midpoint` 控制 | 8/16 | 8/8 | 8/8 | 1/8 | 1/8 | 1/8 | 8/8 | 13.5601秒 |
| `state_peak` 控制 | 8/16 | 8/8 | 8/8 | 1/8 | 1/8 | 2/8 | 8/8 | 19.5276秒 |

两个控制不调用 VLM，始终报告失败；它们的满失败召回同时带来满成功误报。四行准确率均为50%，却有完全不同的错误分布，因此不能只凭准确率或失败召回解释能力。MiniCPM没有一条有效失败定位，条件 MAE 应为未定义（JSON 为 `null`），不能写成0秒。两采样器 Hit@2s 的差值和固定输出 bootstrap 区间均为0；这是双方全零命中的退化情况，不是采样方法等价的证据。

### 抽帧覆盖的事后解释

两采样器到作者失败起点的最近帧覆盖均为 **4/8（±1秒）、6/8（±2秒）、8/8（±5秒）**，但覆盖对象不同。下表列出全部4条在±2秒阈值下改变覆盖状态的失败：状态事件抽帧获得2条覆盖，也丢失2条。样本链接指向公开[作者标签与样本索引](../artifacts/robot-temporal-minicpm/samples.json)，帧时间链接指向各条件的原始记录；时间和误差单位均为秒。

| `state_event` 相对 `uniform` | 样本与任务 | 作者起点 | `uniform` 最近帧；绝对误差 | `state_event` 最近帧；绝对误差 |
|---|---|---:|---|---|
| 获得覆盖 | [`ft-6ccabedb3368519904df`](../artifacts/robot-temporal-minicpm/samples.json) · `balls_to_rack` | 29.206246 | [帧04 · 26.233333588](../artifacts/robot-temporal-minicpm/predictions/ft-6ccabedb3368519904df--uniform.json)；2.972912412 | [帧05 · 28.866666794](../artifacts/robot-temporal-minicpm/predictions/ft-6ccabedb3368519904df--state_event.json)；0.339579206 |
| 获得覆盖 | [`ft-30ed92dcf91893b28406`](../artifacts/robot-temporal-minicpm/samples.json) · `empty_bowl_insert_block` | 54.996332 | [帧09 · 58.000000000](../artifacts/robot-temporal-minicpm/predictions/ft-30ed92dcf91893b28406--uniform.json)；3.003668000 | [帧08 · 56.720001221](../artifacts/robot-temporal-minicpm/predictions/ft-30ed92dcf91893b28406--state_event.json)；1.723669221 |
| 丢失覆盖 | [`ft-4edb1758fef95be05a90`](../artifacts/robot-temporal-minicpm/samples.json) · `stack_cup_pyramid` | 49.520000 | [帧08 · 48.479999542](../artifacts/robot-temporal-minicpm/predictions/ft-4edb1758fef95be05a90--uniform.json)；1.040000458 | [帧08 · 53.360000610](../artifacts/robot-temporal-minicpm/predictions/ft-4edb1758fef95be05a90--state_event.json)；3.840000610 |
| 丢失覆盖 | [`ft-b29a409043e540ac0617`](../artifacts/robot-temporal-minicpm/samples.json) · `cube_stack` | 35.055738 | [帧10 · 35.066665649](../artifacts/robot-temporal-minicpm/predictions/ft-b29a409043e540ac0617--uniform.json)；0.010927649 | [帧10 · 37.066665649](../artifacts/robot-temporal-minicpm/predictions/ft-b29a409043e540ac0617--state_event.json)；2.010927649 |

`cube_stack` 的状态事件最近帧误差为 **2.0109276494140644 > 2秒**，严格不计入±2秒覆盖；不能先把时间四舍五入成2秒再计分。表中所有案例的 MiniCPM 两种条件仍都预测成功，**获得采样覆盖不等于修正了模型判断**。时间相近也不保证单一视角能看见作者定义的不可恢复偏离。这是对固定结果的事后解释，没有据此重调采样算法、阈值或提示词。

发布前独立核对了16条数值输入身份，并重新计算全部32条抽帧索引、实测时间和完整状态分数；另从16段原始头部 MP4 顺序解码，用独立拼接代码重建32张模型输入 JPEG，逐字节与保存面板一致。公开包的114个载荷文件通过标准库验证，主要指标及两个控制也与独立标量计算一致。这些检查支持输入与计分的一致性，不证明模型使用了相关视觉细节。

## Qwen 与固定双模型对照的最终结果

2026-10-15，固定 `qwen3.5:4b` 的32/32次测试调用完成，两模型合计64/64次。Qwen 的两种条件都返回16/16条有效二元判断，没有拒答。其[冻结计划](../artifacts/robot-temporal/plan.json)、[完整评价](../artifacts/robot-temporal/evaluation.json)及[双模型配对比较](../artifacts/robot-temporal-study/comparison.json)均保留原始记录，开发调用不进入下表。

| 方法 | 正确 / 全部 | 失败召回 | 成功误报 | Hit@1s | Hit@2s | Hit@5s | 有效定位覆盖 | 条件 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen `uniform` | 8/16 | 8/8 | 8/8 | 0/8 | 0/8 | 2/8 | 8/8 | 13.601126秒 |
| Qwen `state_event` | 9/16 | 8/8 | 7/8 | 0/8 | 1/8 | 2/8 | 8/8 | 11.789565秒 |

Qwen 的 `uniform` 把全部16条轨迹判为失败；`state_event` 判15条失败，仅正确识别出1条成功轨迹。因此，8/8失败召回同时伴随8/8或7/8成功误报，不能解释为可靠的失败识别。有效定位覆盖8/8只表示真实失败得到合法时间，不表示时间正确。状态采样的 Hit@2s 命中数与两个无标签控制同为1/8，尚未在这个主指标上超过它们。

Qwen 内部 `state_event − uniform` 的 Hit@2s 差为 **+12.5个百分点，95%任务簇配对区间为[0，+37.5]个百分点**。两条件共同有效定位的8条失败上，MAE差为−1.811561秒，区间为[−13.858770，+10.208109]秒。这些小样本、固定预测区间没有建立稳定的采样收益。唯一的状态采样 Hit@2s 来自 `balls_to_egg_carton` 失败样本：作者起点47.590000秒，模型选择48.720001秒，绝对误差1.130001秒；[原始响应与选帧](../artifacts/robot-temporal/predictions/ft-82f1c41e671d5091c6d5--state_event.json)可直接核查。

跨模型的 `Qwen − MiniCPM` Hit@2s 差在 `uniform` 下为0，区间[0，0]；在 `state_event` 下为+12.5个百分点，区间[0，+37.5]个百分点。前者是双方全零命中的退化结果，不是等效性证据。MiniCPM 没有有效失败定位，因此跨模型共同有效定位数为0，MAE差未定义，不能把它的 `null` 当作0秒参与比较。上述比较仅适用于这两个固定本机配置；两者都记录为 `Q4_K_M`，但参数规模、视觉编码器及模板不同，不能隔离架构因素或推广为整个模型系列的排名。

### 三个事后错误例

以下例子在全部结果完成后按错误类型选取，**不是盲选或独立评估集**，也未用于修改提示词、采样器或阈值。作者标签见[公开样本索引](../artifacts/robot-temporal/samples.json)，模型响应的理由不作为新的人工真值。表内时间单位均为秒。

| 样本与作者标签 | `uniform` 原判 | `state_event` 原判 | 可核查的错误 |
|---|---|---|---|
| `ft-4edb1758fef95be05a90` · `stack_cup_pyramid`；失败起点49.520000 | [失败@36.360001](../artifacts/robot-temporal/predictions/ft-4edb1758fef95be05a90--uniform.json)，误差13.159999 | [失败@19.480000](../artifacts/robot-temporal/predictions/ft-4edb1758fef95be05a90--state_event.json)，误差30.040000 | 两条件均判对失败类别，但都比作者起点早；状态采样在此例的定位误差更大。 |
| `ft-6ccabedb3368519904df` · `balls_to_rack`；失败起点29.206246 | [失败@26.233334](../artifacts/robot-temporal/predictions/ft-6ccabedb3368519904df--uniform.json)，误差2.972912 | [失败@57.700001](../artifacts/robot-temporal/predictions/ft-6ccabedb3368519904df--state_event.json)，误差28.493755 | 状态采样已有距作者起点0.339579秒的帧，模型仍选择了很晚的另一帧；获得采样覆盖没有保证正确选帧。 |
| `ft-a653361a22949043025c` · `balls_to_egg_carton`；成功、无失败起点 | [误报失败@57.119999](../artifacts/robot-temporal/predictions/ft-a653361a22949043025c--uniform.json) | [误报失败@47.119999](../artifacts/robot-temporal/predictions/ft-a653361a22949043025c--state_event.json) | 两条件的具体失败叙述都与作者成功标签冲突。没有真实失败起点，时间误差不定义，不能填0。 |

这些记录说明本轮存在成功误报和时间选择错误，不能据此确定错误来自枚举先验、视觉编码或某个具体推理环节；本实验没有隔离这些原因。

### 停止状态、成本与独立核对

64条原始响应均可解析、`status=ok` 且 `done_reason=stop`。MiniCPM 输出20–25 tokens，Qwen 输出35–48 tokens，达到128-token上限的调用均为0；原始 thinking 字段均为空。格式有效和正常停止并不等于判断正确。

| 模型 | 调用数 | 累计客户端调用耗时 | 单调用范围 | 单调用中位数 | API `total_duration` 合计 |
|---|---:|---:|---:|---:|---:|
| MiniCPM | 32 | 1442.141295秒（24.04分钟） | 31.709004–54.169909秒 | 45.935332秒 | 1433.942932秒 |
| Qwen | 32 | 4030.883501秒（67.18分钟） | 76.661678–146.761002秒 | 132.699738秒 | 4019.622412秒 |

累计模型调用耗时共5473.024796秒，即91.22分钟。它包含模型调用本身的加载、提示处理和生成，不包含下载、视频解码、抽帧与导出，不是整个工作流的端到端吞吐率，也不支持实时声明。

最终审计使用独立的标准库标量代码，直接读取冻结作者标签并解析64条原始 JSON，重算四个条件的准确率、回答覆盖、失败召回、成功误报、Hit@1s/2s/5s、有效定位覆盖和条件 MAE，逐项与两个公开 `evaluation.json` 相同。Qwen 的主 Hit@2s 配对区间也用独立的8任务、2000次、seed 1337重采样核对一致。公开计划、标签、响应和面板与本地原记录逐字节相同；两模型对应32个条件的选帧、状态分数、任务提示及拼图字节全部相同。这些检查支持输入与计分可追溯，不证明模型关注了正确的视觉细节。

## 复现入口

完整离线包分别为 [Qwen](../artifacts/robot-temporal/report.html)、[MiniCPM](../artifacts/robot-temporal-minicpm/report.html)，包含原始响应、原字节输入拼图、作者标签快照、数值状态证据、压缩复核视频和标准库校验器。点击报告中的拼图小格可回到对应视频时刻。压缩视频仅供复核，推理使用的拼图另存，未用转码视频替换模型输入。

```powershell
# 已保存实验的离线复算：无模型、无第三方Python包、无需网络
python -I -S -B artifacts/robot-temporal/verify.py
python -I -S -B artifacts/robot-temporal-minicpm/verify.py
python -B tools/export_temporal_development.py --verify --out artifacts/robot-temporal-development

# 在现有项目环境增加HDF5读取依赖
.\.venv\Scripts\python.exe -m pip install -r requirements-temporal.txt
.\.venv\Scripts\python.exe -m robotbench.temporal_data --plan artifacts/robot-temporal/dataset_snapshot/plan.json --dest data/temporalbench

# 先安装本地模型；每次新实验用新目录，实际digest必须入计划
ollama pull qwen3.5:4b
.\.venv\Scripts\python.exe -m robotbench.temporal freeze --data data/temporalbench --out data/my-temporal-run --model qwen3.5:4b --split test
.\.venv\Scripts\python.exe -m robotbench.temporal run --data data/temporalbench --out data/my-temporal-run --split test
.\.venv\Scripts\python.exe -m robotbench.temporal_report report --data data/temporalbench --run data/my-temporal-run
```

未来同名模型 tag 可能改变；新运行会记录当时 digest，但不能自动称为本次固定权重的重现。继续同一运行前会检查源码、依赖、数据、模型摘要、服务器版本和缓存记录；已有失败调用不会自动重试并覆盖。推理使用与视频工作流共用的进程锁。

校验器重算已保存预测的指标与原文解析，不重新推理，也不是作者数字签名。没有实体机器人、没有策略训练、没有成功部署证明；本项目可以支持审阅的内容是固定任务、真实输入、明确对照、可追溯错误及其测量结果。
