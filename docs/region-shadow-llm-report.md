# LLM Region Planner 影子模式结果（2026-10-22）

## 结论

- 8 个区域全部完成模型调用并收到响应；
- 8/8 通过严格 schema + 语义校验；
- LLM 建议与人工决定一致 **7/8 = 87.5%**；
- 唯一分歧：**P8 信访示例表格**，LLM 建议 `hybrid`；已做 hybrid 单页样板，用户复核后决定维持 `retain_module`，记录为人工否决；
- P5 仪表盘和 P12 条形图两个校准样本均与人工决定一致。

这里的 **7/8 只是三分类路由标签一致率**，不是具体拆分方案一致率。人工标签分布为
6 个 `retain_module`、1 个 `hybrid`、1 个 `native`，所以“始终保留”的基线是 6/8。
P5 虽然同样标为 `hybrid`，模型计划把 KPI、图表和表格内部都原生化，人工记录是
“标题原生 + 仪表盘整图”，编辑范围明显不同。P8 也只应记录为一次带输入、执行器和
编辑目标的否决，不能推导成“所有表格区域都应保留”。

## 范围评分（任务 B）

为了避免“标签相同”掩盖拆分范围差异，人工决定增加了结构化标注：
`annotation_level`、`retain_scope`、`fidelity_target`、`required_edit_targets`、
`allowed_native_parts`、`must_retain_parts`。评分现在分开输出：

- 路由标签一致率：**7/8 = 87.5%**（始终保留基线 6/8）；
- 拆分范围一致率：**6/8 = 75%**；
- 过度拆分 2 个（P5、P8），拆分不足 0 个，保留范围冲突 1 个（P5）。

| 区域 | 路由 | 范围 | 范围问题 |
|---|---|---|---|
| P5 | 一致 | **不一致** | 模型多原生化了 panel/card/kpi/bars/table 等 12 类部件；人工要求保留 `dashboard_body` |
| P8 | 不一致 | **不一致** | 模型建议 hybrid，人工否决并保留整图；模型列出的原生部件全部属于过度拆分 |
| P12 | 一致 | 一致 | 模型原生部件都在人工允许范围内，标题/条形/标签/百分比齐全 |
| P6/P7/P9/P10/P11 | 一致 | 一致 | 人工要求整区域保留，模型原生部件为空 |

标注可信度也需要区分：P5、P12 是 `explicit`，P8 是 `trial_veto`，其余 5 个是
`proxy`（来自已验收 deck，尚未逐区域正式复核）。因此 6/8 是“当前标注下的范围一致率”，
不是独立测试集上的准确率，也不能直接外推为其他模板的还原率。

三个区域的候选试编译记录见 `region-candidates/report.md`：P5、P12 为 `ACCEPTED`，
P8 为 `REJECTED`；每条记录包含 baseline/candidate 证据哈希、对象预算、编辑覆盖和状态链。

本轮只跑影子模式，没有修改任何 IR、PPTX 或 14 页 deck。

## 模型与调用

- 模型：`kimi/kimi-k2.6`
- 端点：`https://moma.cmecloud.cn/v1`
- prompt：`region-decision/1`
- capabilities sha256：`88e09e23865ea83df7ba401021089aa4ae627e381c8af2e3250362bd87b7078e`
- 发送内容：仅区域 crop、9 个数值特征、分类、能力快照；
- 未发送：human decision/note、description、整页、OCR 文本、路径、bbox、region_id；
- 首次调用因 `max_tokens=1200` 被 reasoning 全部消耗，`finish_reason=length`、content 为空；改为 `thinking:{type:disabled}` + `max_tokens=4096` 后正常返回；
- 密钥只通过 stdin 进入 secure runner，不写入 job、settings、报告或 git；建议使用后轮换。

## 结果

| 区域 | 类型 | LLM 建议 | 人工决定 | 一致 | 置信度 |
|---|---|---|---|---:|---:|
| P5-example-058-dashboard | UI 仪表盘 | hybrid | hybrid | 是 | 0.85 |
| P6-funds-analysis-chart | 资金分析仪表盘 | retain_module | retain_module | 是 | 0.92 |
| P7-clue-network | 关系网络 | retain_module | retain_module | 是 | 0.95 |
| P8-petition-example | 表格+摘录 | **hybrid** | retain_module | **否** | 0.85 |
| P9-image-020-report-ui | 软件界面 | retain_module | retain_module | 是 | 0.92 |
| P10-vehicle-risk-example | 地图+照片 | retain_module | retain_module | 是 | 0.95 |
| P11-rural-three-assets | 照片图示 | retain_module | retain_module | 是 | 0.95 |
| P12-industry-risk-bars | 横向条形图 | native | native | 是 | 0.92 |

## P8 分歧与人工否决

LLM 认为 P8 可以做成混合方案：

- 原生部分：标题、面板、卡片、表格、行头、单元格文字、数值；
- 保留部分：图标、状态胶囊；
- 可编辑性 high、视觉风险 low、成本 medium。

随后做了 hybrid 单页样板：

- 标题、来信三行文字、5 行标签/值、风险胶囊原生；
- 6 个图标合成为一张透明条带；
- 单页导出成功，规模 164 节点、78 文本框、32 图片。

区域级 MAE：当前整图保留约 **8.6**，hybrid 样板约 **17.7**。标题、表格标签列和图标经过一轮修正后视觉接近，但表格值区仍偏平、分隔线和图标细节仍有差异。

**人工决定：维持整图保留。** 这次分歧记录为 human veto，用于校准 planner：

- LLM 正确识别了表格结构，但低估了像素保真的代价；
- 对于“表格 + 来信摘录 + 状态标签”混合区域，默认保留更稳；
- 后续如果确实需要改表格文字，再从归档样板继续。

样板归档：`deck-p8-petition-hybrid.pptx`、`p8-petition-side-by-side.png`、
`p8-petition-sample-report.md`。

## 产物

- 计划：`region-shadow-llm/plans.llm.json`
- 校验：`region-shadow-llm/validation.llm.json`
- 一致率：`region-shadow-llm/score.llm.json`
- 模型/请求/响应哈希：`region-shadow-llm/llm-provenance.json`
- 应用报告：`region-shadow-llm/apply-report.json`
- 人工决定基线：`region-shadow-llm/human-decisions.json`
- 英文报告：`region-shadow-llm/report.llm.md`

原始 `response.body`、crop 和 job 目录保留在 gitignored 的 `.build/region-shadow/llm-jobs-v2/`，不发布。
