# 本机验证记录

日期：2026-10-13。状态随实际运行结果更新，未完成的项目不视为通过。

## 已完成

- 硬件检查：i5-1130G7，4核8线程，16GB 内存，Intel Iris Xe 核显；Ollama 0.34.4。
- 核心 Python 3.12 环境已安装，`pip check` 通过。核心依赖版本见 `requirements-lock.txt`；初次网络下载不稳定时，`pydantic-core 2.46.5` 从本机同版本 Python 的已安装分发复制，随后由匹配的 Pydantic 2.13.5 与依赖检查验证。
- 17 项离线单元测试通过，包括图片上限、EXIF、透明背景合成、缩放、模型校验、JSON 校验、截断、评分、模型 digest、恢复与 Qwen3-VL 解析兼容。证据：`artifacts/unit-tests.txt`。测试 mock 模型回复，不代表真实模型效果。
- 三张自制图片与参考答案已逐张核对：红圆4个、咖啡价格$3.50、B柱20。样例是功能检查，不是公开数据集。
- 命令行问答和评测的帮助入口通过，Python 源文件语法检查通过。
- Gradio 6.30.0 及完整依赖已安装，`pip check` 通过；版本见 `requirements-ui-lock.txt`。主页及 `/config` 曾通过 HTTP 200 检查（28组件、4张样例），记录见 `artifacts/ui-verification.json`。浏览器自动化工具超时，未宣称完成浏览器视觉检查。
- 官方模型 blob 的大小和 SHA256 校验通过，再由 `ollama pull` 完成官方校验与登记。记录见 `artifacts/model-download-verification.json`。实际模型 digest 为 `0635d9d857d497aeadba3d7d27485746c50554446f9f6ec01ef39788221adbe8`；`/api/ps` 确认 CPU 推理（`size_vram=0`），见 `artifacts/ollama-loaded-model.json`。

## 真实图片问答

使用本机 `qwen3-vl:2b`、CPU、4线程、4096上下文、最长边448像素、32输出tokens，加入兼容前缀后，3条均完整返回且归一化完全匹配参考答案。完整原始输出、输入哈希、参数和 Ollama 分项计时见 `artifacts/evaluation.json`。

|自制样例|参考答案|实际回答|原始调用耗时|
|---|---|---|---|
|4个红色圆形|4|4|105.954秒|
|英文菜单咖啡价格|$3.50|$3.50|91.264秒|
|简单柱图B柱|20|20|103.284秒|

最后一次运行复用了前两条已完成记录，仅重新调用第三条。表中为各条原始调用耗时，不将缓存读取当成模型速度。仅3个自制样例，不能解释为通用准确率100%，也不是公开benchmark。首次加载、内存压力与同时运行的其他应用均影响耗时。

## 中文收据与完整应用流程

通过运行中的 Gradio API 执行 `tools/smoke_ui.py`，真实验证上传图片、刷新模型、队列、结构化提取、JSON/TXT 导出与文件下载。使用768最长边上限（原图720×480，不放大）、384输出tokens，生成165 tokens，推理112.448秒，包含上传和导出的调用共113.812秒。

- JSON Schema 校验通过，`done_reason=stop`，未截断。
- 店名“晨光果铺”、苹果单价3.00元/数量2、香蕉单价2.00元/数量1、合计8.00元均与自制原图一致。
- 语义限制：`objects` 中将店名、商品和表头等文字条目当成对象，不能视为准确的可见物体检测。`uncertainties=[]`也不能证明内容全部正确。
- 两个导出文件成功下载；下载的JSON与返回记录完全一致，TXT包含界面结果文本。
- 证据：`artifacts/structured-receipt.json` 与 `artifacts/ui-inference.json`。界面服务最终在 `http://127.0.0.1:7860/` 运行；后台进程与日志位置保存在 `.runtime/ui-process.json`，重开电脑后按README的 `start.ps1` 启动。

本次完成的是HTTP/配置和应用API的端到端验证；浏览器自动化工具超时，页面视觉检查未完成。

## 本机 Ollama 解析器的兼容处理

本次下载的 `qwen3-vl:2b` 配置实际使用 `qwen3-vl-thinking` renderer/parser。仅传 `think=False` 时，32-token 测试的两个调用没有最终答案并被正确标记为截断，失败证据保留在 `artifacts/evaluation-32-tokens.json`；增加至256 tokens的计数调用得到正确的4，见 `artifacts/qa-256-tokens.json`。

[Ollama v0.34.4 parser](https://github.com/ollama/ollama/blob/v0.34.4/model/parsers/qwen3vl.go#L58) 没有使用 `thinkValue` 决定初始状态。因此针对已声明 thinking 的 qwen3vl family，代码在用户图片消息之后追加空且闭合的 assistant 思考前缀 `<think>\n\n</think>\n\n`，同时保留 `think=False`。[Renderer](https://github.com/ollama/ollama/blob/v0.34.4/model/renderers/qwen3vl.go#L85) 保留此未结束的 assistant 回合，parser 进入正式内容状态。Schema 也从正式回答开始约束，不再等待思考结束标记。结果记录保存 `assistant_prefill`，其他模型 family 不使用此处理。

运行期间曾出现可用内存降到约440MB、界面与评测进程意外退出的情况。没有关闭其他用户应用；恢复时可用内存约4.8GB，评测利用已保存结果续跑。这说明16GB总内存不能代替实际可用内存检查，模型/界面启动和评测应串行进行。

## 下载中断的处理记录

本次下载曾多次遇到网络超时，随后本机 Ollama 服务及终端会话停止。恢复时4份下载元数据（分片0/4/8/9）为全零，导致 `invalid character '\\x00'`，而不是模型推理错误。

依据 [Ollama v0.34.4 下载实现](https://github.com/ollama/ollama/blob/v0.34.4/server/download.go)，先将全部16份元数据备份到项目 `.runtime/ollama-metadata-backup-20261013`，再仅将损坏4片进度设为0。官方 registry 清单确认目标 blob 大小为1,889,496,384字节，16片各118,093,524字节，与已有数据文件一致。未修改已安装模型。完整下载仍必须通过 Ollama 的 SHA256 校验。

中断前界面显示的84%不是完整落盘的可恢复进度；恢复后以仍有效的元数据为准重新下载。备份、一次性恢复脚本与下载日志不提交到 Git。

官方多分片下载仍反复回退后，改用系统现成的 `curl` 从同一官方 blob 地址断点下载，完整传输约19分35秒。核对官方 SHA256 后将 blob 放入 Ollama 缓存，再执行官方 `ollama pull`，最终成功。项目目录中重复的1.9GB临时下载已删除，Ollama 已安装模型保留。
