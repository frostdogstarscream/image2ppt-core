# 独立 API 重建与结构评测

`evaluate-api` 是显式的本地实验入口。它调用配置中的模型，保存原始响应，执行固定的重建器和编译器；参考 PPTX 只进入后续评测。原有 `plan`、`reconstruct`、`convert` 保持原用途与兼容性。严格实验不要使用 `plan` 的可配置重试路径，也不要导入历史人工修订计划。

本次指定样本为 `PPT图片/02.png` 至 `15.png`。模型为 `qwen/qwen3.8-max`，地址为 `https://moma.cmecloud.cn/v1`。最多两轮，每轮每页一次，最多 28 次规划请求。固定非 thinking、8192 输出 token 上限、120 秒单请求总时限。超时、无效响应和拒绝也计入一次预算；不会换模型或自动重试。合成回环测试使用独立夹具目录，不消耗这 28 次真实模型预算。

## 配置与隔离

生成配置使用 `IndependentAPISuite`，只接受图片/OCR 的路径和 SHA-256、页序及固定模型参数。未知字段（包括 `reference_pptx`）会被拒绝。OCR 文件自身的原图哈希也必须匹配。模型的图片、精简 OCR 和统一协议提示词在发送前写入冻结请求文件，没有逐页补充提示。

参考配置使用独立的 `APIReferenceSuite`，包含参考文件哈希和图片到参考页的映射。`prepare-api-reference` 可先核对映射；`score-api` 必须等生成目录封存后才读取参考对象。参考中原生的文字、表格、形状与图片分别统计。参考内容不会反向送入生成请求。

运行时凭据只读取当前进程的 `VISION_LLM_API_KEY`，请通过本机私密环境或密钥管理工具注入，不要写入 JSON、脚本或聊天。没有凭据时写出 `NOT_RUN_CONFIGURATION`，请求数为 0，不创建生成封存文件。明确的 HTTP 401/403 会停止本次后续请求；修正凭据并显式续跑后只处理尚未请求的页，失败的认证页不重试。

冻结内容包括源图、OCR、完整请求、协议、项目代码、模型参数、字体候选及哈希、Python 包版本、绑定的 Artifact 运行时路径与关键外部文件哈希。冻结后的脚本在独立进程中执行；检查不通过就停止，不静默吸收工作区的修改。冻结编译器在自己的目录内完成校验，再将相同字节的结果复制到轮次交付目录，并重新核对 PPTX 哈希。

## PowerShell 命令

以下命令在仓库根目录执行，Python 环境需已有本服务依赖，参考评测另需 `lxml`（`pip install -e './services/vision[evaluation]'` 可安装声明的依赖；不要为已可用的运行环境强行重装）。编译使用本机已安装的 bundled Artifact runtime。

```powershell
$env:PYTHONPATH = 'services/vision/src'
$experiment = 'outputs/independent-api-20261019'

# 参考核对，输出必须是一个新目录。
python -m image2ppt_vision.cli prepare-api-reference `
  --generation-config "$experiment/generation-config.json" `
  --reference-config "$experiment/reference-config.json" `
  --compiler-root . --output "$experiment/reference-audit-new"

# 本次交付已经冻结第一轮，配置好私密环境后续跑它。
python -m image2ppt_vision.cli evaluate-api `
  --config "$experiment/generation-config.json" `
  --output "$experiment/campaign-verified" --compiler-root . --round 1 --resume

# 生成结束并存在 generation.seal.json 后，独立评分。
python -m image2ppt_vision.cli score-api `
  --run "$experiment/campaign-verified/round-1" `
  --reference-config "$experiment/reference-config.json" `
  --compiler-root . --output "$experiment/score-round-1"
```

第二轮必须等第一轮全部样本都已占用请求预算并封存后才能准备。先查看程序汇总的失败类型，只做通用修改与测试，不能改单页计划或按文件名/文字内容触发特例；随后运行以下命令冻结并执行全部页面。协议 1.1 的表格支持已实现为待验证能力，其真实收益仍须由两轮结果证明。

2026-10-19 在线首轮 14 页均收到服务端 Token 限流（HTTP 429）。据此，第二轮冻结了统一的传输优化：同尺寸 JPEG95、4:4:4，仅在平均 RGB 变化不超过 2/255 且文件更小时使用；否则保留 PNG。原图与 OCR 始终不变，实际传输图及其哈希也留档。第二轮相邻请求开始时间至少间隔 65 秒；收到 429 时停止尚未发起的页，记为 `STOPPED_RATE_LIMIT`，不自动重试。修正服务端配额后，显式续跑只会处理尚未请求的页。这个停止规则保留剩余预算，但未跑完的第二轮不能称作完整整轮结果。

```powershell
python -m image2ppt_vision.cli evaluate-api `
  --config "$experiment/generation-config.json" `
  --output "$experiment/campaign-verified" --compiler-root . --round 2

python -m image2ppt_vision.cli score-api `
  --run "$experiment/campaign-verified/round-2" `
  --reference-config "$experiment/reference-config.json" `
  --compiler-root . --output "$experiment/score-round-2"

python -m image2ppt_vision.cli compare-api-rounds `
  --first "$experiment/score-round-1/score.json" `
  --second "$experiment/score-round-2/score.json" `
  --output "$experiment/comparison.json"
```

不要通过新建 campaign 绕过预算。`budget.sqlite` 是本次两轮唯一预算账本；同一轮/页只允许一次占用，恢复已有响应不再联网。已完成的轮次会验证封存哈希后直接返回。`--prepare-only` 只冻结、不发送请求。CLI 成功返回表示报告已写出，业务成败应读取报告中的 `status`，不能仅依据退出码判断。

## 用户明确取消总量上限后的连续实验

2026-10-19 用户明确取消请求总量上限，并再次确认严格独立评测。原 `campaign-verified` 的两轮和 28 次账本保持封存。续行系列位于 `outputs/independent-api-20261019-unlimited`，`series.json` 记录授权边界及与原系列的关联；不能将其误报为此前预算内的请求。

新配置使用 `IndependentAPISuite` 的 `version: "1.1"`，显式设置 `max_requests: null`、`max_rounds: null` 和 `first_round_protocol: "1.1"`。`null` 表示没有总量上限，有限整数仍可设置预算。旧配置 1.0 仍锁定两轮、28 次以及首轮协议 1.0，其身份哈希不随新字段变化。更改已存在系列的预算仍会被拒绝。

连续实验仍须逐轮冻结：上一轮全部页面尝试完成并通过封存检查，才能准备下一轮；同一轮同一页只能占用一次请求，失败不自动重试，不加入逐页会话修正或参考 PPT 数据。用户的总量授权不改变这些独立性约束。

续行第一轮只加强通用提示词中的平面区域划分、OCR ID 绑定、非空文本覆盖、例外式栅格保留、表格区域一致性和正尺寸要求；协议校验器、重建器和编译器标准保持不变。这个版本用于检验提示词改进是否足够，不能预先宣称成功率或编辑性已改善。

可以通过 `scripts/diagnose_api_plans.py --run <sealed-or-running-round> --output <new-json>` 在冻结代码上复核保存的失败响应。它不发出模型调用、不修正计划，诊断结果必须在轮次目录之外。

## 渲染 OCR 的本地环境与分辨率诊断

已存在的 `outputs/vision-v1/models` 包含带 SHA-256 清单的 PP-OCRv4 权重。续行系列在自己的 `ocr-deps` 中安装 RapidOCR 3.10.0、ONNX Runtime 1.20.1 及缺失依赖；全局 Python 包和远端服务保持原状态。运行评测时仅为该进程设置 `PYTHONPATH` 包含此目录、`VISION_MODELS` 指向已有权重。实际版本与模型哈希见 `ocr-environment.json`。推理中的隐式模型下载仍禁用。

1280 像素回读图中的小字会降低 OCR 自身准确率。`scripts/calibrate_api_ocr.py` 以相同确认区域比较原始源图、与普通回读同尺寸的源图和生成回读图；增加 `--render-native-resolution` 时，再从实际 PPTX 导出源图宽度的 PNG 重新 OCR。诊断保留原评分，不用分辨率补偿静默改成通过，也不将 OCR 代理分数当作人工错字率。该路径仅读取和渲染已封存 PPTX，不向生成流程传入参考对象。

## 2.0 规划协议与评分修正

新增 `ReconstructionPlan` 2.0 通过独立 `plans_v2.py` 接入严格 API 实验。系列配置使用 `version: 1.1` 和 `first_round_protocol: "2.0"`；该系列后续轮次保持 2.0。既有系列身份、旧响应和旧评分不修改，新协议使用新的系列目录。

`text_decisions` 是稀疏决策，每个 OCR ID 只能选择一次 `correct` 或 `keep_pixels`；默认交给像素验证器尝试原生文字。纠错必须包含非空新文字、精确原 OCR 和理由。`semantic_groups` 只表达成员关系，边界由成员推导，允许嵌套与空间重叠；`raster_regions` 只表达不相交的照片/复杂插图裁剪，不能切过 OCR 或包住需要纠错的文字。已经物化、互不重复且 z 顺序连续的成员可以成为原生平面分组；共享残差、嵌套和跨层限制逐项记录，不能称为原生分组成功。

表格只填写一次 bbox、从 0 到 1 严格递增的行列边界，以及锚点单元格的 OCR IDs 和跨度。程序确定性地产生行列尺寸、单元格文字、覆盖空格及专属区域；仍严格核验每条 OCR 所属单元格、完整归属、源图哈希及非交叉关系。原模型提案保存在 `plan.json`，转换结果在 `lowered-plan.json`，推导记录在 `lowering.json`。这不是失败响应修复，不更改原响应，也不增加模型请求。旧服务入口仍接受 1.0/1.1；2.0 目前是独立 API 实验入口。

自定义几何评分改为比较变换到幻灯片坐标系的字面量折线/多边形、填色和线条，允许坐标单位、起始顶点、方向和冗余共线点的等价表达；XML 哈希只保留审计用途。曲线、公式导引、主题/继承颜色、复杂效果或非均匀缩放线条暂记 `NOT_RUN_UNRESOLVED_GEOMETRY`。内容 OCR 正式评分改为从实际 PPTX 直接渲染源图宽度，禁止放大旧预览冒充重新渲染。新旧轮次必须用同一新版评分器重新评分后比较，原评分文件仍保留。分数阈值未放宽。

## 表格与能力边界

`ReconstructionPlan` 1.0 继续可读。1.1 增加 `tables`：每项声明独立区域、bbox、二维单元格文本、列宽、行高、OCR ID 和零起始合并关系。列宽/行高必须与边界相符；合并不能交叉，覆盖的非锚点单元格必须为空。每条 OCR 必须完整属于一个单元格，内容不一致或跨界就拒绝，不补造文字。字体、颜色等由源图估计，仍需外观评测。

SlideIR 原生 table 新增可选 `row_heights`（目标画布像素）、`merges`、`cell_styles`，旧节点无这些字段时行为不变。行高总和必须匹配目标高度。单元格样式支持填色、文字颜色和粗体。最终交付是 OOXML 原生表格，已用合成图检查合并关系和最终 PPTX 副本的单元格编辑。该证据不代表真实图片识别准确。

严格基线不启用需要人工 `reviewed` 的图层分支，不调用 Ming，不使用生成式拆层。自定义几何仍无规划/编译支持；参考原生对象的图片替代按结构缺失计分，不能因可移动或外观近似而宣称原生成功。

## 结果与指标

每轮保存 `freeze.json`、`round.json`、`inputs/`、`attempts/*/response.body`、`transport.json`、逐页计划/重建/失败记录、实际 PPTX 与回读预览、`generation.json` 和最终 `generation.seal.json`。失败页留在分母中，成功页的部分 deck 明确列出页序。原始响应在解析前保存；若服务意外回显密钥，保存脱敏正文及原正文哈希，该页直接失败。

评测包括文本对象 P/R ≥95%、图形 P/R ≥90%、位置/尺寸容差各为画布维度的 1%、表格及合并单元格、严格自定义几何内容匹配、平均 RGB 差 ≤10/255 与 4×4 局部差异，以及最终文件副本的文字删除/模块移动/表格单元格编辑。文本合并和拆分另行列出。

内容统计只使用源图 OCR 与参考文本一致的区域，包含原生表格单元格。规范化字符准确率目标 99.5%，数字序列必须完全一致，多出的数字也算错误。未确认区域单独报告，不能把该分母解释成整页内容覆盖率。渲染内容由已验证的本地 OCR 模型检查；没有模型文件时记为 `NOT_RUN_OCR_UNAVAILABLE`，不自动下载。OCR 结果是技术代理指标，不等于人工阅读验收。

参考隔离回读会在评测目录创建只含目标 slide 的副本，保留依赖与命名空间；原参考文件哈希保持不变。参考图与源图片的差异单独报告，生成外观始终相对输入图片评分。两轮对比要求输入哈希、参考映射、参考文件和评分器代码一致，不择优拼页。

`human_acceptance`、PowerPoint/WPS 检查状态始终为 false，直到目标应用实际操作并获确认。没有可靠单价时只报 token 用量及耗时。指定 14 页已参与开发，因此不将结果推广为新页面成功率。

## 离线验证

```powershell
python -m pytest services/vision/tests -q -p no:cacheprovider --basetemp outputs/new-test-run
./image2ppt.ps1 test
./image2ppt.ps1 demo
# 对 demo 返回的 IR 路径执行：node scripts/check_demo.mjs <IR>
python scripts/verify_api_table.py --output outputs/new-table-smoke
python scripts/verify_api_pipeline.py --output outputs/new-pipeline-smoke
```

最后两个脚本分别验证合成表格的实际文件和两轮回环 HTTP 流程，使用新目录、保留所有失败诊断；它们不调用真实模型。
