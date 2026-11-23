# 2026-11-23　正文被拆入图片的修复与实测

同一张业务原图重新上传后，原先只在图中出现的行尾可以在 PowerPoint 中单改。目标段落有两行，两行共用一个原生文本框。对这个段落单独执行了改字和移走两项检查。

## 运行与对照

| 项目 | 旧的真实上传 | 本次真实上传 |
| --- | --- | --- |
| 任务 | `152aaf9f3fda4709827b77ca798aa0c6` | `2c64a991f5a7423181d397b05582cb8b` |
| 运行源码 | `b751738a4604b1169556eb03861dbff2c7fe195b` | `88d975c2e37f0d06a0e03398fb7d20e52d7c0a87` |
| PPTX SHA256 | `024ed95a6b48ec0e65a0d4aef9df174e9eb70a67c6a840c242ccc981ceee3150` | `091fbb1c6b558bfbb7a38a245510dfe5e162b071395f9a57706a551d97aaa3d6` |
| 同口径完整原生行 / 原图 OCR 行 | 24/27 | 25/27 |
| 物理文本框 / 形状 / 独立图片素材 | 21 / 3 / 18 | 21 / 3 / 17 |
| Office 可编辑对象数 | 42 | 41 |
| 单改“实施交付” | 没有完整文本框 | 保存重开通过 |
| 事先冻结的四项其他编辑 | 4/4 | 4/4 |
| 整页 SSIM，越高越接近原图 | 0.8164 | 0.8086 |
| 整页色差 ΔE00 p95，越低越接近原图 | 35.271 | 36.931 |
| 内容由底图承载的占比 | 30.11% | 30.08% |

原始上传 PNG 的 SHA256 为 `9b59da0e4bc6cb71e26864a08d263b1aa8ae2b56e5e59abe94ba1901f7fd2460`。运行入口是 `POST /api/v1/jobs`，参数为 `orchestrator=opencode`、`baseline_only=true`、`fresh_perception=true`。本页是冻结集第 6 页。应用使用源图进行生产，测试程序单独读取冻结答案。基线标记为 `deterministic-baseline`。本次未启动规划大模型或独立 Go 复核，人工及 WPS 验收为 UNKNOWN。

旧文件已有的 24 行原生匹配全部保住。21 个文本框可承载 25 行，因为同段内的两行可以使用同一个文本框。按对象位置核对，少掉的独立图片是原先烘入“实施交付”的小图片。页首蓝色双箭头的取图范围由 40×24 变为 49×27。上排三张卡片在两份 PPTX 中均有原生底框。

## 修复与验收证据

原始 OCR 已读到完整行，置信度约 0.999。旧流程把句末顿号当作独立装饰，并将 414 像素宽的正文截短到 287 像素。`r-008` 又把末尾的一部分做成可移动图片。旧 PPTX 的副本在实际 PowerPoint 中找不到承载完整目标行的原生文本框。

现在保留原始行、标点和范围。只有图上可证实为独立项目符号或箭头时才拆符号。带编号的普通正文字段即使被规划为图片，也继续走文字修复。若另一张已提交素材已认领这些字，程序在候选副本上验证局部 OCR、来源、笔画和下层填充后再移交。跨边界或证据不足时保留原内容。多所有者移交另有聚焦测试。这次新上传在上游即避免了原先的图片误判。

Office 16.0 检查了成品文件，具有同一 SHA 的移动/删除回执共 82 条。专门的改字用例将“实施交付”换成“上线验收”，保存、关闭、重开，检查两个顿号、同段下一行、邻项和文本框位置。改字后目标区域之外的可见变化为 0 像素。移走用例另取副本，将原生段落移动 325 个原图像素并保存重开。原位置旧字的深色像素数从 2317 变为 0。

| 冻结编辑目标 | 动作 | 真实 PowerPoint 回执 |
| --- | --- | --- |
| 研发与集成保障 | move | EDIT_OBSERVED |
| 属地服务保障卡片底色 | fill | EDIT_OBSERVED |
| 第二步 选择场景，验证效果 | text | EDIT_OBSERVED |
| 算网与模型保障 | text | EDIT_OBSERVED |

整页的自动状态为 `REVIEW_REQUIRED`，自动发布为 `False`。仍无原生对应的原图 OCR 行为 中国移动、China Mobile。隐藏全部原生文字后，审计还返回 2 处原始 OCR 清单以外的读数。它们保留在 `comparison.json` 的 `audited_rows`。原图 OCR 行覆盖率可计算，独立人工清点的所有可见字召回率为 UNKNOWN。

核看 PowerPoint 的原尺寸渲染，右上卡片修好的正文比原图略窄。隐藏所有原生字后，上排卡片标题带和下排部分底色仍有浅色痕迹，目标行移开后的原位置干净。自动审计另读出的“&lt;/&gt;”来自笔记本图标里的符号，“昆 1”位于放大镜图标和分隔线的位置。这两处单列为待确认。两行品牌文字仍随右上角标志一起显示。本轮只验证这张真实上传的页面。

## 测试集证据

对四个模板家族的 10 页冻结感知记录复查了 406 行 OCR，其中 397 行符合原筛选条件。第 2、6、10 页各纠正一次正文标点误切，撤销两个错误图标候选，第 10 页实际独立的导航箭头仍保留。用新逻辑重新核对历史 PPTX，原有 324 个原生匹配保留，丢失 0 个。本段为保存证据的重放。十页重新上传后的整体覆盖率为 UNKNOWN。

## 运行可靠性

第一次真实上传 `ed06a043d1fd4337b527990771e8ef20` 已通过目标改字和移走。中间一次任务 `e83b4f64f17e49cea7750eecec27e9e4` 在一个图标的可选 RMBG 卷积中长时间停止推进，第二次 `bd56232a78d745b09deeb05748ad3cd3` 已测得 Office 连接异常，临时 PPTX 文件锁使失败回执未能送达。两次实验均取消并保留原目录。现在在独立子进程用同一模型计算可选抠图，对同一失败输入避免重复阻塞。Office 对测试副本核对完整路径，并将失败证据与仍被锁住的临时文件分开打包。本次最终任务的可选 RMBG 超时数为 0。

## 复跑

最终任务产生 11 份可核对的 Office 工作票，自动回执分布为 {'REVIEW_REQUIRED': 2, 'UNKNOWN': 4, 'PASS': 5}。此前的中间任务 `4728eb8ba0d0435d9930b29542c1b57d` 曾因 `Presentation.Close` 报错而撤下一项素材。修复后对该次工作票的副本做了 12 次实际替换、保存、关闭并重开，12 次内容均保留。最终版本接手真实 Office 后，已取消该中间任务，该任务没有计入最终成绩。中间验证期间桌面曾保留一份隐藏的空白演示文稿，以保持 Office 会话。每个实际编辑另开独立副本，并在核对路径后关闭、重开。

1. 在验证版本运行 `./image2ppt.ps1 test`。本轮 39 项通过。
2. 在项目 Python 环境运行以下十组测试。真实 PowerPoint 检查单独占用 Office 工作进程。本轮结果为 `156 passed in 193.12s (0:03:13)`。
3. 使用正常上传入口、相同原图哈希、新任务目录和上述参数重新生成。下载或从任务目录读取带哈希的确定性基线。
4. 在私有验收包内运行 `probe_edit.py <运行目录>` 复查改字和移走。在仓库以 `python services/experiments/probe_fields.py <PPTX> <新目录> --cases services/experiments/holdout/cases.json --single-page 6` 运行四项冻结编辑。

```powershell
$suites = 'native_text_ownership','background_plate_audit','region_text','host_baseline','marked_execution','batch_assets','opencode_runtime','region_crop','parallel_candidates','matting'
$files = $suites | ForEach-Object { "services/product/tests/test_$_.py" }
python -m pytest @files -q
```

4090 的任务目录是 `E:/i2p-experiments/native-text-1123/88d975c2e37f/region/jobs/2c64a991f5a7423181d397b05582cb8b`。对应的私有验收包在 `E:/i2p-experiments/native-text-1123/88d975c2e37f/acceptance-2c64a991f5a7423181d397b05582cb8b/acceptance-bundle.zip`，包含原版与改字/移走证据、逐项回执、整页、隐藏文本后的实际渲染、评分卡、测试及重放摘要。

## 改动文件

- `services/product/OPENCODE_RUNTIME.md`
- `services/product/product/matting.py`
- `services/product/product/parallel_candidate_stage.py`
- `services/product/product/parallel_edit_probe.py`
- `services/product/product/parallel_office.py`
- `services/product/product/parallel_rebuild.py`
- `services/product/product/recovery/candidate_stage.py`
- `services/product/product/recovery/region_artwork.py`
- `services/product/product/recovery/region_crop.py`
- `services/product/product/recovery/region_text.py`
- `services/product/product/runtime/batch_backbone.py`
- `services/product/product/runtime/host_baseline.py`
- `services/product/product/runtime/host_evidence.py`
- `services/product/product/runtime/marked_routes.py`
- `services/product/product/runtime/office.py`
- `services/product/product/runtime/office_bridge.py`
- `services/product/product/runtime/plate_audit.py`
- `services/product/product/runtime/service.py`
- `services/product/product/runtime/state.py`
- `services/product/tests/test_matting.py`
- `services/product/tests/test_native_text_ownership.py`
- `services/product/tests/test_opencode_runtime.py`
- `services/product/tests/test_parallel_candidates.py`
- `docs/acceptance/2026-11-23-native-text-ownership.md`
