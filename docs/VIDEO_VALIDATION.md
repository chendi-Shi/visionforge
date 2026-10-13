# 视频工作流验证 · 2026-10-13

本次真实跑通的是**可复核的视频分析工作流**。三段均返回合法结构和现存证据引用，但内容存在明确错误，不能宣称模型准确率为100%，也不能把这些结果直接当成事实。

## 输入与环境

- 公开样片 Street traffic，作者 Editor，CC BY 3.0。35秒，原始1920×1080 WebM，26,490,572字节；[来源与许可](../examples/video/source.json)。
- 源SHA256：e5facb24baf755f0c1193999d823e99f6032661ad2bdef5cc28cc1cbbfde03e2。
- i5-1130G7 / 4核8线程 / 16 GB / Iris Xe；本地Ollama 0.34.4，qwen3-vl:2b Q4_K_M；4 CPU线程、4096上下文、帧最长边448、最多512输出tokens、temperature=0。
- 模型digest：0635d9d857d497aeadba3d7d27485746c50554446f9f6ec01ef39788221adbe8。
- FFmpeg 7.1生成960×540、12 fps、420帧的H.264无声预览；预览SHA256：35af362b6c2e58fa6d3d8ae1483815805fea0ec61e3296f85248c94702ab8045。
- 本片源元数据报告1k fps，未用此名义值推导时间。证据帧时间为规范化预览的实际解码帧序号 / 12；窗口边界不是事件精确起止。

## 真实调用结果

作业ID：23958a06958844548df8a195ef49b005。三个候选窗口全部选中，但每个窗口仅采样两帧，仍不能称为连续覆盖。

|片段|采样窗口|实际证据时刻（秒）|模型耗时|调用次数|结构/引用|
|---|---|---|---|---|---|
|S0001|0–15秒|3.000 / 11.916667|267.876秒|1|通过|
|S0002|15–30秒|18.000 / 26.916667|251.857秒|1|通过|
|S0003|30–35秒|31.000 / 33.916667|310.447秒|1|通过|

总模型耗时 **830.180秒（约13分50秒）**，不包括暂停间隔和其他操作。这是一次同时进行开发验证的本机运行，不是控制变量性能基准，也不是实时视频分析。

原始输出、prompt、模型身份、图像原始/处理后哈希、耗时和人工备注都在[证据包JSON](../artifacts/video-evidence/0354ceb6ccaf4e3eb7857a7368ff0b67/evidence.json)中；[完整报告目录](../artifacts/video-evidence/0354ceb6ccaf4e3eb7857a7368ff0b67/)保留了解压后的全部资产。克隆仓库后，用本地浏览器直接打开 [report.html](../artifacts/video-evidence/0354ceb6ccaf4e3eb7857a7368ff0b67/report.html) 即可离线查看。为减少仓库体积，重复的预生成 ZIP 不入 Git；UI 仍能按作业导出完整 ZIP。

## 内容质量

[逐句内容复核](VIDEO_CONTENT_REVIEW.md)由助手目视对照实际六帧完成，尚未经用户人工确认。支持、错误、无法确认分别记录，没有修改原始模型输出。

- 可见公交车、汽车及部分大字识别有依据，例如38、6408、8165和第一帧的Red Bull字样，不能一概判为幻觉。
- 部分目的地、车号位置和朝向描述错误或无法确认。
- S0002声称公交向右移动，缺少可靠同车对应；其中一条只引用后一帧却作跨帧比较。
- S0003错误声称后一帧仍可见Red Bull货车，这是明确的跨帧对象延续错误。
- uncertainties有复述原判断或照抄提示词的情况，不能当成可靠的置信度。
- schema只能限制格式与引用存在性。模型把比较句标成“状态”时，单纯检查“变化”类别的双帧引用无法验证实际语义。

因此，这个2B模型在当前配置下适合产生**待复核草稿**；本轮没有证明可靠自动事件检测、目标跟踪、小字OCR或精确事件边界识别。观察重点、采样密度和模型能力都影响结果。

## 实测工程行为

- 准备完成停在ready，没有自动触发模型。
- S0001开始后请求暂停，当前段结束后落库并停在paused。
- 通过Gradio API继续，从S0002开始；完成后S0001的result与暂停快照完全一致，三个片段均只有一次调用。
- 浏览器选择S0001显示两张对应证据；点击第二张后播放器定位11.917秒。服务重启后，三个结果和复核备注仍存在。
- 浏览器以“开发验收”检索，仅显示含该备注的S0001；保存备注后仍为unreviewed，未冒充用户已确认。
- 界面导出API真实下载ZIP、JSON和CSV；检查规范化视频哈希、三段记录、署名与复核备注。
- 独立HTML报告通过支持Range请求的本地静态服务打开，点击末段证据后，浏览器video.currentTime为33.916667、duration为35。离线包不依赖此验证服务器。
- 证据包包含规范化MP4，而非给原始WebM改扩展名。11项完整性验证均通过，见[机器验证记录](../artifacts/video-validation.json)。

离线测试 **40/40通过**：图片处理与评测17、双帧层4、媒体层10、作业层5、导出层4。覆盖无切镜头视频、选窗预算、哈希失效、失败原文、暂停恢复、跨进程worker互斥、prepare失败重试、报告转义及预览校验。pip check通过。

## 复现

原始 WebM 按需下载；先按 README 运行 `tools/fetch_video_sample.py`，启动并导入样片，再使用自己的作业ID：

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe video_cli.py import examples/video/street-traffic.webm --max-segments 3 --max-tokens 512 --wait
.\.venv\Scripts\python.exe video_cli.py start YOUR_JOB_ID --wait
.\.venv\Scripts\python.exe tools/verify_video.py YOUR_JOB_ID --output .runtime/video-verification.json
```

verify_video.py不发模型请求，仅核对已完成作业和导出资产；对于未完成作业会明确拒绝。模型或代码身份变化时需新建作业，不混用旧结果。

