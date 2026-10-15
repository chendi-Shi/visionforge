# 机器人子任务失败理解：数据与实验协议

本轮研究问题是：**给定机器人要执行的子任务及动作前后画面，加入时序变化信息，是否比只看动作后的画面更可靠地判断成功/失败？** 实验在本机 CPU 上提取固定 CLIP 特征，并通过本地 Ollama 比较视觉语言模型的不同输入方式。

实验单位是一个真实机器人子任务的动作前后观测。当前公开数据没有秒级失败发生时间、完整连续视频、关节轨迹或逐步控制动作，因此本轮只能评价**子任务状态转换验证**，不能称为已完成连续视频的秒级失败定位、策略学习或机器人控制。已有视频证据库用于回查和整理；算法比较、样本划分、错误分析和可复算结果是本轮重点。

本轮真实结果是：固定2B VLM的48次调用全部给出成功，三个条件的失败召回均为0/8；线性分类器在140条主实验中的小幅增益区间跨零，在独立四粗组压力测试中准确率增益消失。下文保留协议、负结果和错误记录，不把结构校验通过写成任务正确。可直接查看[主报告](../artifacts/robot-benchmark/index.html)、[完整指标](../artifacts/robot-benchmark/metrics.json)和[压力测试指标](../artifacts/robot-sensitivity/evaluation.json)。

## 数据来源与可取得的内容

采用作者发布的 [Guardian-FailCoT-OOD-datasets / UR5-Fail](https://huggingface.co/datasets/paulpacaud/Guardian-FailCoT-OOD-datasets/tree/bd2b24224e4da833a8d42ca694d839622e54bb48/ur5fail_test)，固定 revision：

```text
bd2b24224e4da833a8d42ca694d839622e54bb48
```

上游[数据卡](https://huggingface.co/datasets/paulpacaud/Guardian-FailCoT-OOD-datasets/blob/bd2b24224e4da833a8d42ca694d839622e54bb48/README.md)说明，UR5-Fail 由作者在真实 UR5 机器人上采集，执行子任务的成功/失败由人工标注。当前选用的是执行验证部分，不使用自动扰动产生的规划验证样本。

已经核查的上游文件如下；这些下载与解析事实不等于模型评测结果：

| 文件 | 字节数 | 内容 |
|---|---:|---|
| `ur5fail_test/metadata_execution.jsonl` | 140,467 | 任务、子任务、图像关联、成功标签与失败原因等元数据 |
| `ur5fail_test/internVL_dataset_execution_vanilla.jsonl` | 350,449 | 上游格式化输入及答案，用于核对记录关联与标签 |
| `ur5fail_test/records.tar.gz` | 81,685,957 | 本 split 的 PNG 资产，约77.9 MiB |

图像包 SHA-256：

```text
4d3ff623db50221281a053343f8aec82e438cd5a0358998743e158e564f228fa
```

已完成本机下载和哈希核验：140条记录，其中71条失败、69条成功；有23个 `taskvar`、16个基础任务组、56个 episode 分组，以及510个不同 SHA-256 的 PNG 资产。**140条记录不是140次独立机器人 rollout**：数据包含复用观测的任务重标注与状态对调整。同一任务和同一图片之间的相关性必须进入划分与不确定性估计。

每条包含6张图，发布列表顺序为动作前3个视角、动作后3个视角。**以发布顺序为准，不能按数字文件名重新排序**：文件名数字不总与语义时序一致。103条记录的视角文件名使用 `0/1/2`，另外37条使用 `left/right/wrist`，但上游通用对话模板统一写作 front/left/right。因此本项目采用中性的 **view slot 0/1/2**，不宣称统一相机标定或绝对方位。

适配器以 `execution_reward` 的0/1为二元真值，并检查 `reward` 一致、`planning_reward == 1` 及上游 vanilla 对话答案 True/False 一致。失败类别来自上游 vanilla 的 `<category>`，映射为 `no_gripper_close`、`imprecise_grasping_pushing`、`wrong_state_or_placement`、`wrong_object`、`no_progress`；成功样本为 `success`。元数据的 `failure_mode` 存在不同别名，仅保留作来源记录，不混作评价类别。

适配器生成 `inputs.jsonl`、独立的 `labels.jsonl`、来源记录 `private-provenance.jsonl` 和校验清单 `manifest.json`。推理输入只含任务/子任务文本及被选定的图片；成功标签、失败类型、失败原因和答案仅在评价阶段使用。图片使用内容 SHA-256 文件名，避免源路径成为标签线索。当前 `inputs.jsonl` SHA-256 为 `a9aca272141a9e0e4e26b0bd42db5ff9437c03ae09b49357d5a7694113b7f664`，`labels.jsonl` 为 `59baa374c9d6894d50cfdd9792b3539f3e3efe46415d084c182a3f204254398c`。

许可证以该固定版本数据卡声明的 Apache-2.0 为依据，使用时保留作者、论文、版本和来源记录。本轮只使用该作者自采的 UR5-Fail split；同一仓库再分发的 RoboFail、RoboVQA 各有原始来源条款，不能把本轮选择泛化为对所有 split 的许可判断。归属见[第三方说明](../THIRD_PARTY.md)。

## 固定特征与基础任务组留一验证

CLIP 使用项目已有的固定 FastEmbed / ONNX 编码器，权重和预处理身份见[模型复用说明](SEARCH_REUSE.md)。冻结图像/文本编码器，训练的仅是轻量线性分类器；没有微调 VLM、CLIP 或机器人策略。

采用**按基础任务组留一验证（leave-one-group-out）**。组 ID 直接取 `taskvar.split('+')[0]`，每次留出一个完整组，使用其余组拟合分类器，再预测被留出的所有记录。若训练记录与当前测试组共享任一图像内容哈希，进一步从该折的训练集合移除相关记录，以减少重复画面泄漏。

这个分组只去掉发布方的 variation 后缀，没有合并语义近似任务。例如 `real_stack_cup`、`real_stack_cups`、`real_stack_cup_spatial` 分属不同组，`real_put_food_in_box` 与 `real_put_fruit_in_box` 也可能相近。**无精确图像 SHA 或 episode 重叠，不等于语义完全隔离**；本实验不能被称作严格的全新任务家族泛化。主协议和16组结果保持原样；发现语义重叠后另行冻结并执行了下文的唯一4粗组后验压力测试，分开报告。

线性分类器固定为 `LogisticRegression(C=1, class_weight="balanced")`；不根据测试分数搜索超参数。比较：

| 方法 | 输入与作用 |
|---|---|
| `train_majority` | 当前折训练集多数类；检查类别分布即可达到的水平 |
| `text_only` | 子任务文本的 CLIP 特征；检查只读指令的偏差 |
| `after` | 动作后图像与目标文本的特征及交互 |
| `transition` | 动作前、动作后、变化特征及文本交互 |

每个视角先做单位长度归一化，同一阶段的3个视角取均值后再次归一化，得到512维 `b`（before）和 `a`（after）；子任务文本单位向量为 `t`。`text_only = t`；`after = [a, t, a*t]`；`transition = [b, a, a-b, t, b*t, a*t, (a-b)*t]`，其中 `*` 为逐元素乘法。方法分别为512、1536、3584维。`StandardScaler` 只在每折训练数据上拟合；分类器使用 `lbfgs`、`max_iter=2000`、`random_state=0`，记录收敛警告。

方法之间保持相同样本、同一折划分和同一重复资产剔除规则。若剔除后训练集仅剩一类，明确记录多数类回退；若没有训练记录，运行报错，不删除测试样本以伪造完整成绩。分类器输出的分数未校准，不能解释为真实成功概率。

`transition` 相比 `after` 同时改变了输入维度（3584 对1536）、特征组合和交互项；在固定 `C` 下也改变了正则化作用。对于这里的线性分类器，`a-b` 已在 `b,a` 的线性张成空间中，`(a-b)*t` 也等于 `a*t-b*t`：重复拼接没有增加对应的线性表达空间，却会与标准化、L2惩罚一起改变有效正则化。因此它比较的是两套预先固定的低成本基线，**不能将成绩差异单独归因为模型理解了时序，也不构成已验证的新时序架构**。隔离这种因果解释还需要控制特征容量和正则化的进一步实验；本轮不会根据测试分数再改表示。

**这是本项目在小规模数据上进行的探索性交叉验证，不是 Guardian 官方零样本结果。** 该公开测试集被用于本项目的交叉验证拟合，成绩不能与直接零样本评测等同，也不能声称已对所有未来机器人任务泛化。

## 2026-10-15：已完成的全量 CV 实测

在本机完成140条样本、16折基础任务组留一验证。四种方法均返回140条预测，没有弃权、缺失记录、单类别回退或收敛警告；当前数据不存在跨基础任务组的完全相同图片 SHA，因此本次共享图片检查移除0条训练记录。这不排除近似画面和共同数据来源带来的偏差。

| 方法 | 正确 / 全部 | Accuracy | Balanced accuracy | 失败召回 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| 训练折多数类 | 60/140 | 42.86% | 42.54% | 46/71 · 64.79% | 39.71% |
| 仅子任务文本 | 75/140 | 53.57% | 53.63% | 35/71 · 49.30% | 53.51% |
| 仅动作后状态 | 89/140 | 63.57% | 63.70% | 39/71 · 54.93% | 63.35% |
| 动作前后变化 | 95/140 | 67.86% | 68.02% | 40/71 · 56.34% | 67.48% |

前后变化比动作后状态多答对 **6条**：修正13条，同时退步7条；Accuracy 差值 **+4.29个百分点**，按基础任务组配对 bootstrap 的95%区间为 **[-1.32, +9.38]个百分点**，跨越零，不能证明稳定提升。 主比较使用完全一致的140个样本 ID、固定折外预测、2000次基础任务组配对重采样，种子为0。16个基础任务组仍是小规模探索，不能据此宣布对新任务稳定泛化或复现官方零样本成绩。

提升主要来自成功样本：成功召回由50/69（72.46%）变为55/69（79.71%）；失败召回只由39/71（54.93%）变为40/71（56.34%）。前后变化方法仍把31条真实失败判断为成功，并把14条真实成功判断为失败。净增加6个正确中，5个来自成功类、1个来自失败类，不应描述成失败检测的大幅提升。

按作者失败类型分解，变化方向也不一致：

| 作者失败类型 | 仅动作后识别为失败 | 前后变化识别为失败 |
|---|---:|---:|
| `imprecise_grasping_pushing` | 10/15 · 66.67% | 8/15 · 53.33% |
| `no_gripper_close` | 11/13 · 84.62% | 12/13 · 92.31% |
| `no_progress` | 0/1 · 0.00% | 0/1 · 0.00% |
| `wrong_object` | 11/20 · 55.00% | 10/20 · 50.00% |
| `wrong_state_or_placement` | 7/22 · 31.82% | 10/22 · 45.45% |

未关闭夹爪与状态/放置错误的召回有所增加，但不精确抓取/推动和操作错误对象的召回下降；唯一的 `no_progress` 样本两种方法都漏判，1例不能用于估计该类别的通用性能。这里评分的是“是否识别为失败”，没有把作者失败类别作为模型已正确解释的原因。

本机统计来源为 `data/robot-probe/evaluation.json`，逐折预测为同目录 `cv.json`；已导出[公开指标](../artifacts/robot-benchmark/metrics.json)和[逐折预测](../artifacts/robot-benchmark/cv.json)。本次 `cv.json` SHA-256 为 `ac5b24a961d136c6fff30b8770d96f015e0795ab39b95c8dadf995b265cdb629`。提取510个图像向量与66个子任务文本向量耗时452.43秒，CV耗时66.59秒；当时与本地 Ollama 并发争用CPU，不能作为隔离环境下的速度基准。

### 相同观测、不同子任务的补充诊断

复用已经生成的折外预测，检查“完全相同的有序6图、不同子任务文本、相反成功标签”记录对，不增加推理，也不改变训练、参数或主指标分母。本次有10组观测、27条记录组成17个符合条件的配对；这些配对可以复用样本，不能当成17次独立试验，也不据此声称统计显著。未发现任务文本、子任务文本及有序观测全部相同却标签相反的精确输入冲突。

| 方法 | 两条均正确 / 全部配对 | 两条给出相同二元判断 / 全部配对 | 配对回答覆盖率 |
|---|---:|---:|---:|
| 训练折多数类 | 0/17 | 17/17 | 100% |
| 仅子任务文本 | 3/17 | 10/17 | 100% |
| 仅动作后状态 | 3/17 | 13/17 | 100% |
| 动作前后变化 | 3/17 | 14/17 | 100% |

主表总分较高的前后变化方法，在这个指令条件诊断上没有比动作后方法多答对配对，且多数配对仍忽略应随子任务改变的标签。它尚不能作为可靠的指令条件动作效果验证器。结果已保存在[配对诊断](../artifacts/robot-benchmark/diagnostics.json)；原因解释仅限这些可观测输出，不推断分类器具有或缺少某种人类心理能力。


## 独立后验压力测试：四个粗分组

发现基础任务标识之间存在语义近似及相同总体任务文本之后，另做一次**后验敏感性分析**。它在看过主实验结果后提出，不是主实验的事前协议，也不替换上面的16组结果。粗分组首次拟合前保存 `plan.json`，冻结唯一映射、分组理由、来源与代码 SHA、运行源码快照。只使用这一份冻结方案，不根据其成绩继续选择划分。

| 粗分组 | 记录数 |
|---|---:|
| `cup_stacking` | 12 |
| `drawer_storage` | 25 |
| `open_receptacle_placement` | 92 |
| `tabletop_pick_tool` | 11 |

仍使用全部140条样本及同一份冻结 CLIP 特征，每次留出一个完整粗组；四种方法、折内标准化、固定 `C=1` 与共享图片检查不变。组大小12/25/92/11极不平衡，最大的开放容器放置组占92/140；不同粗组仍共用抓取等动作，不能称为严格的语义 OOD 或跨机器人验证。

| 方法 | 正确 / 全部 | Accuracy | Balanced accuracy | Macro F1 | 失败召回 | 成功召回 |
|---|---:|---:|---:|---:|---:|---:|
| 训练折多数类 | 67/140 | 47.86% | 48.08% | 46.66% | 23/71 · 32.39% | 44/69 · 63.77% |
| 仅子任务文本 | 78/140 | 55.71% | 56.09% | 52.58% | 21/71 · 29.58% | 57/69 · 82.61% |
| 仅动作后状态 | 82/140 | 58.57% | 58.69% | 58.36% | 36/71 · 50.70% | 46/69 · 66.67% |
| 动作前后变化 | 82/140 | 58.57% | 58.46% | 58.26% | 47/71 · 66.20% | 35/69 · 50.72% |

四种方法覆盖率均为100%。动作后与前后变化方法都为82/140，Accuracy 差值为0.00个百分点；17条修正与17条退步相抵。以4个粗组对固定折外预测进行2000次配对重采样，Accuracy 差值的95% percentile 区间为[-3.65, +13.04]个百分点。**只有4个 cluster，区间非常探索性**，不包含重新训练的方差，不能用来作可靠的显著性或部署结论。

前后变化的失败召回从36/71（50.70%）提高到47/71（66.20%），同时成功召回从46/69（66.67%）下降到35/69（50.72%）；多识别11个失败的同时，多误判11个成功。其 balanced accuracy 为58.46%，略低于动作后方法的58.69%；macro F1也从58.36%变为58.26%。这是不同错误类型之间的取舍，没有形成整体提升。

原主协议的前后变化方法为95/140（67.86%），在这个粗分组下为82/140（58.57%）。这个下降说明本轮表现对分组方案敏感，不支持稳定时序收益。它也不将后验粗分组变成新的主结果：两套划分都保留，明确分开报告。

已单独导出[冻结计划](../artifacts/robot-sensitivity/plan.json)、[逐折预测](../artifacts/robot-sensitivity/cv.json)和[全部指标](../artifacts/robot-sensitivity/evaluation.json)，与主报告目录分开。复算已有主流程结果时，应显式指定同一份特征和主 CV；先生成计划，再在独立命令中拟合：

```powershell
.\.venv\Scripts\python.exe -m robotbench.sensitivity --data data/robotbench --features data/robot-run/features.json --reference-cv data/robot-run/cv.json --output data/robot-sensitivity plan
.\.venv\Scripts\python.exe -m robotbench.sensitivity --data data/robotbench --features data/robot-run/features.json --reference-cv data/robot-run/cv.json --output data/robot-sensitivity run
```

本轮实跑的特征和主 CV 位于 `data/robot-probe`，与上述按首页命令顺序复算的 `data/robot-run` 仅输出位置不同；公开计划保存了实际输入校验值。


## 本地 VLM 输入消融

为控制 CPU 成本，每个基础任务组按固定样本哈希选取一条记录，构成固定诊断子集。选择不依赖标签、模型预测或成绩；不能运行后替换难例。本次固定16条恰好包含8条成功、8条失败；这不是按标签平衡抽样，仍须报告类别计数、逐类召回和 balanced accuracy。

三个条件使用相同本地模型、相同问题和输出结构：

| 条件 | 输入 | 可回答的问题 |
|---|---|---|
| `after_only` | 动作后的多视角画面 | 仅凭最终状态能否判断子任务完成 |
| `ordered` | 按发布顺序提供动作前后多视角画面 | 正确时序信息是否改善状态转换验证 |
| `reversed` | 交换动作前后输入 | 模型是否对输入顺序敏感 |

默认使用全部视角，将原始256×256图片按3列拼板，每格增加24像素阶段/视角标题。`after_only` 为768×280，`ordered` / `reversed` 为768×560；阶段作为行、view slot 作为列，不插值生成中间运动。`--view 0/1/2` 可选单一视角，但属于不同实验配置，必须使用独立输出目录。默认模型 `qwen3-vl:2b`、`temperature=0`、`seed=0`、`num_ctx=4096`、4个CPU线程、输出上限160 tokens。

`ordered` 比 `after_only` 多输入3张观测、拼板总像素翻倍，因此观测信息与计算预算同时改变。这项比较不能单独隔离时序顺序的作用；同样6张图的 `ordered` / `reversed` 才保持图片集合一致，但后者仍只是输入损坏诊断。

另有5/140条记录的动作前后三个视角逐一完全相同。固定16条 VLM 子集中有1条：`ur5-8e2e7ec301a2829cacbc`，作者的“grasp screwdriver”成功记录。对这条记录交换前后不会改变输入图像，因此应说明有效顺序扰动仅覆盖其余15条；不能把此样本不改变预测解释为模型忽略了时间。它仍保留在原定样本集合和分类分母中。

`reversed` 是输入损坏诊断。原始成功标签属于原始执行，**不自动成为反向任务的真值**；因此其分数只用于观察敏感性和预测变化，不能作为“反转执行”的正确率。真正时序收益的主要配对比较是 `ordered` 对 `after_only`；另报告有效配对的预测翻转数、分母与翻转率。

首个调用之前保存[运行计划](../artifacts/robot-benchmark/vlm-plan.json)，固定16个样本、3个条件、选择种子、输入/代码校验值和实际模型 digest，共48次调用。保留每次调用的实际图片哈希与组合、提示词、运行设置、原始响应、解析状态和耗时。重跑同一命令跳过身份一致的已保存结果；若模型、数据或设置发生变化，则拒绝混入原目录。诊断子集结果与全部样本的线性分类结果分别报告，不能把两者的样本量或准确率混在一起。

### 完整48次调用结果

2026-10-15，48/48次实际调用完成，全部 `status=ok`，全部预测为 `success`。下面指标来自正式导出的 `metrics.json`，每个条件均在原定相同16条上评价，覆盖率均为100%，没有丢弃失败请求或难例。固定模型 digest 为 `0635d9d857d497aeadba3d7d27485746c50554446f9f6ec01ef39788221adbe8`。

| 条件 | 正确 / 全部 | Accuracy | Balanced accuracy | 失败召回 | 成功召回 | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `after_only` | 8/16 | 50.00% | 50.00% | 0/8 · 0.00% | 8/8 · 100.00% | 33.33% |
| `ordered` | 8/16 | 50.00% | 50.00% | 0/8 · 0.00% | 8/8 · 100.00% | 33.33% |
| `reversed`（相对原标签的损坏诊断） | 8/16 | 50.00% | 50.00% | 0/8 · 0.00% | 8/8 · 100.00% | 33.33% |

主要比较 `ordered − after_only` 的准确率差值为0，逐条二元预测也完全相同。交换前后时，16个有效输出配对中有0个预测翻转；排除那条前后图像相同的样本后，仍为0/15。`reversed` 的原标签分数不能解释成反向执行正确率。三条件都漏掉全部8条失败，而不是因为运行错误、解析失败或拒答而失分。**schema 有效只说明输出符合约定，不说明图像证据或动作判断正确。**

这项负结果限定于当前 `qwen3-vl:2b`、固定提示词、拼板方式、解码设置及这16条样本。本轮没有在看到标签或分数后调提示词、挑模型或替换难例，不能据此声称所有 VLM 都不能理解机器人动作。保存的调用耗时合计4408.50秒（约73.47分钟），是本机串行调用的记录合计，不是隔离资源后的速度基准，也不代表实时可用。

### 原始错误例：事后选例

以下三例从已完成的 `ordered` 输出中事后选择，用于检查错误说明，不是另一个独立评测集。真值与失败类别来自数据作者；引文是模型实际生成的 `evidence`，不是本项目认可的视觉事实。三例在三个条件中均被模型误判为成功。

| 样本与子任务 | 作者真值 / 类别 | `ordered` 原始证据摘录 | 可观察到的问题 |
|---|---|---|---|
| [ur5-178da949ed257ef1202f](../artifacts/robot-benchmark/predictions/ur5-178da949ed257ef1202f.ordered.json) · `grasp healthiest food` | failure / `wrong_object` | “The red cylindrical can (likely the healthiest food) is now in the robot's grasp” | 把推测的对象身份作为成功依据，与作者的错误对象标签不符。 |
| [ur5-8ed8da723a89249df276](../artifacts/robot-benchmark/predictions/ur5-8ed8da723a89249df276.ordered.json) · `grasp banana` | failure / `wrong_object` | “the robot's gripper is positioned directly above the banana, indicating it has grasped the banana.” | 说明把上方位置推断为已经抓住，仍输出成功。 |
| [ur5-b2b82180904118385ce8](../artifacts/robot-benchmark/predictions/ur5-b2b82180904118385ce8.ordered.json) · `grasp scissor` | failure / `no_gripper_close` | “the robot's gripper is positioned over the scissors, indicating it has grasped the scissors.” | 说明将位于物体上方作为完成依据，未识别作者标注的夹爪未闭合失败。 |

这些输出说明当前协议会产生流畅但错误的成功理由；位置描述本身不足以确立对象正确、夹爪闭合或抓取完成。本轮只要求成功/失败/拒答及自由文本证据，没有训练或评价失败原因分类器，因此不能把这份错误分析写成“模型已完成原因识别”。完整原文及六图观测保存在[离线报告](../artifacts/robot-benchmark/index.html)和逐样本 JSON 中，未为展示修正模型文字。

### 同一16条诊断子集上的 CV 对照

使用原有16折 CV 的折外预测，按 VLM 冻结计划中的同一16个样本 ID 取子集，不重新拟合或调参；四种方法回答覆盖率均为100%。

| CV方法 | 正确 / 相同16条 | Accuracy | 失败召回 |
|---|---:|---:|---:|
| 训练折多数类 | 9/16 | 56.25% | 8/8 · 100.00% |
| 仅子任务文本 | 11/16 | 68.75% | 5/8 · 62.50% |
| 仅动作后状态 | 15/16 | 93.75% | 7/8 · 87.50% |
| 动作前后变化 | 15/16 | 93.75% | 7/8 · 87.50% |

每个基础任务组仅取一条，且这次恰好8条成功、8条失败。这个子集与140条主实验的样本加权分布明显不同，大组在子集中的权重被压低。**15/16（93.75%）只是同子集诊断成绩，不能替换140条主结果、不能作为新的总体准确率。** 与 VLM 比较时使用这里的相同16条，评估主要能力边界时仍保留140条主表和独立压力测试。


## 评分与失败处理

主标签为 `success` / `failure`。模型拒答、缺失预测、结构解析失败和运行错误都作为 `abstain` 保留在原样本集合中，不能从准确率分母里去掉。

| 指标 | 定义 |
|---|---|
| Accuracy | 正确预测数 / 全部样本数，拒答也计入分母 |
| Balanced accuracy | 成功召回和失败召回的均值；缺少某一类时不伪造有效值 |
| Macro F1 | 固定成功/失败两类取均值；拒答计入真实类别的漏判 |
| Coverage | 给出有效二元判断的样本数 / 全部样本数 |
| Selective accuracy | 仅在已回答样本上计算，作为次要指标并明确分母 |
| Task macro accuracy | 分别计算各基础任务组的 accuracy，再等权平均，避免样本多的基础任务组支配结果 |
| 按失败类型的召回 | 对真实属于该失败类型的样本，统计被识别为失败的比例；拒答仍计入分母 |

按失败类型的召回不是失败原因分类准确率。若系统没有预测原因类别，不能把识别出“失败”写成已正确解释具体原因。自由文本说明留作错误分析，不由同一个被测 VLM 自己给自己评分。

不把相关的同任务图片当成独立样本做置信区间。以整个基础任务组为 cluster，固定随机种子，默认2000次有放回重采样，计算95% percentile 区间。少于2个基础任务组不报告 cluster 区间，少于10个基础任务组明确提示估计不稳定；重采样后缺少某一类而无法定义 balanced accuracy 的次数也须记录。这里重采样的是**已经固定的折外预测**，不会在每次重采样时重新训练，因此区间不包含训练方差；各折训练集又有重叠，不能把它解释成完整学习流程的不确定性。

方法比较用完全相同的样本 ID 集合及相同 cluster 重采样，报告 `B − A` 的配对差异。不能用各方法各自成功返回的子集作主比较；共同回答子集上的比较只能作为次要结果。

协议检查应覆盖：基础任务组/episode/图像哈希泄漏、样本 ID 对齐、缺失字段、推理输入误含标签或真值键。数据适配、指标与报告的离线测试只证明程序行为，视觉性能必须来自真实图片及真实本地推理输出。

## 在这台电脑复算

已有公开结果可以先用 Python 标准库离线校验，无需安装模型、重新拟合或发出推理请求：

```powershell
python -B tools/verify_robot_release.py artifacts/robot-benchmark
python -B artifacts/robot-sensitivity/verify.py --main-bundle artifacts/robot-benchmark
```

第一条核对主报告的原始与解析输出、请求身份、模型 digest、样本标签、逐项哈希并重新计算指标；第二条用主包中的原始输入/标签和特征身份核对独立压力测试。公开 Git 包保留解压后的资产，冗余 ZIP 不要求存在。这些检查证明内部一致性，不是作者数字签名，也不能替代独立数据上的泛化验证。

按仓库首页安装基础环境后执行。数据准备、特征和交叉验证不依赖 Ollama 在线；只有 `vlm` 步骤需要已安装并运行的本地视觉模型。下载器核对固定文件大小与哈希，不自动拉取其他机器人数据集。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-robot.txt
.\.venv\Scripts\python.exe tools/fetch_robot_benchmark.py --output data/robotbench
.\.venv\Scripts\python.exe -m robotbench.run --data data/robotbench --output data/robot-run features
.\.venv\Scripts\python.exe -m robotbench.run --data data/robotbench --output data/robot-run cv
.\.venv\Scripts\python.exe -m robotbench.run --data data/robotbench --output data/robot-run vlm --model qwen3-vl:2b --view all --max-tokens 160
.\.venv\Scripts\python.exe -m robotbench.run --data data/robotbench --output data/robot-run report
```

`features.json` 记录特征、编码器身份、池化与耗时，向量缓存按模型身份和图片/文本哈希复用；`cv.json` 保存逐折训练/测试ID、共享图片剔除名单和逐样本预测；`predictions/` 保存真实 VLM 调用；`metrics.json` 和 `index.html` 为可复算指标与离线报告。停止推理后可以运行 `report` 为尚未完成的实验生成报告；推理进行中，同目录报告命令会因互斥锁拒绝执行。部分报告明确显示完成数，并将未回答样本留在分母内，不得作为完整48次调用结果。

公开包的 `protocol_sources/` 保留实际用于推理和 CV 的代码版本，`evaluation_sources/` 保留后来用于校验、评分及报告的版本。后加校验与报告代码不会改写原始预测；原始响应、解析判断、模型 digest 和输入校验值仍可逐条核对。两个目录分别记录身份，不能把后加的报告代码误称为原始推理版本。

本轮工程验证通过171项回归测试，`pip check` 通过。对公开 PyPI 执行 `--dry-run --no-deps` 检查，确认锁定包可获取；该检查没有安装依赖，不能据此宣称已验证全新环境安装。程序检查与模型内容正确率分别报告。

`data/` 为本地工作目录，不随 Git 提交原始数据。首次 CLIP 权重下载约606 MB，后续复用已有固定模型缓存；本轮机器人图像压缩包约78 MiB。CPU推理耗时依赖模型与视角数，不承诺实时。机器资源限制不能作为删掉错误或拒答记录的理由。

## 数据选择的调研记录

2026-10-15 核查了以下官方来源，明确区分数据能支持什么，以及本机可完成什么：

| 来源 | 已核查内容 | 本轮取舍 |
|---|---|---|
| [LeRobot 数据工具](https://huggingface.co/docs/lerobot/using_dataset_tools) | 官方已有在线与本地 episode 可视化，显示同步相机、状态和动作 | 不再以重建通用数据集播放器作为算法贡献 |
| [LeRobot PushT 固定 v2.1](https://huggingface.co/datasets/lerobot/pusht/tree/6e1d9c95aaf8abb19342e723935b6af7c54b5903) | 可逐 episode 下载；episode 0 的 Parquet 11,195 bytes、视频41,125 bytes；包含图像、2维状态/动作、`next.reward` / `next.success` | 适合低成本进度信号管线检查，但为二维仿真任务，不替代真实机器人失败验证；本轮未作为成绩来源 |
| [CALVIN 官方数据说明](https://github.com/mees/calvin/blob/main/dataset/README.md) | debug 包仍有1.3 GB；包含相机、动作、机器人/场景状态和语言区间 | 具备物理状态验证的研究价值，本轮没有下载或启动其完整仿真环境 |
| [RoboReward](https://huggingface.co/datasets/teetone/RoboReward) | 真实 rollout 视频与末端进度分数；官方说明测试集经过人工复核 | 末端进度分数不能直接充当秒级失败时间标签；本轮未使用 |
| [AHA 官方代码](https://github.com/NVlabs/AHA) | 官方流程涉及 RLBench / CoppeliaSim 失败生成；发布的 REFLECT 评估资产主要是组合图 | 不在这台机器上另建大型仿真与训练环境，也不把组合图当成带原始时钟的连续视频 |
| [REFLECT 官方仓库](https://github.com/real-stanford/reflect/tree/c2f9db993e372250321ca74e56ae27519716c934) | `tasks_real_world.json` 有30条任务记录及 `gt_failure_step`；值可以是一个时间点或多个时间点 | 多点不能擅自解释为起止区间；完整媒体未取得前，不能宣布已复现秒级定位 |

REFLECT 媒体还做了直接文件核查：固定仓库 README 的“Real World Dataset”链接指向 [Google Drive 的 `reflect-real-world.zip`](https://drive.google.com/file/d/1n5WLGURwL3R2kbxf-l86Gl4tV3zNR3iv/view)。服务器返回完整大小 **83,030,375 bytes**，支持 HTTP Range。读取 ZIP 中央目录得到188项，目录本身22,702 bytes；内容为旧代码快照、音效、少量示意图及 `.git`，没有任务视频，也没有代码引用的 `real_world/data/` 观测目录。没有为这次检查下载整个压缩包。该事实只描述检查时此链接的内容，不推断作者从未发布其他媒体。

以后若取得带独立时间真值的完整失败视频，应新增时间定位基准，保留原始时钟，按完整视频或基础任务组隔离划分，并报告定位误差/时间容差命中率。不能从本轮动作前后分类成绩推导出秒级定位能力。
