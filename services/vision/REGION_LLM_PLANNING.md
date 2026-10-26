# Region LLM planner（影子模式）

该流程只提出“区域该原生、混合还是整图保留”的建议，不修改 IR 或 deck。人工决定仍是权威。

## 使用

```bash
# 1. 从已验收 plans 提取人工决定（P12 保持 native）
python scripts/region_shadow.py human-decisions \
  --plans .build/region-shadow/plans.json \
  --output .build/region-shadow/human-decisions.json

# 2. 离线准备 job；不联网
python scripts/region_shadow.py prepare-llm \
  --inventory .build/region-shadow/inventory.json \
  --jobs .build/region-shadow/llm-jobs-v2 \
  --model kimi/kimi-k2.6 \
  --base-url https://moma.cmecloud.cn/v1

# 3. 唯一网络入口；密钥从 stdin 读取，不落盘
python scripts/run_supervised_jobs_secure.py .build/region-shadow/llm-jobs-v2 \
  --jobs P5-example-058-dashboard P6-funds-analysis-chart ... --key-stdin

# 4. 解析保存的响应，严格校验后生成 plans.llm.json
python scripts/region_shadow.py apply-llm \
  --inventory .build/region-shadow/inventory.json \
  --jobs .build/region-shadow/llm-jobs-v2 \
  --output .build/region-shadow/plans.llm.json \
  --human-decisions .build/region-shadow/human-decisions.json

python scripts/region_shadow.py validate --plans .build/region-shadow/plans.llm.json \
  --output .build/region-shadow/validation.llm.json \
  --human-decisions .build/region-shadow/human-decisions.json
python scripts/region_shadow.py score --plans .build/region-shadow/plans.llm.json \
  --output .build/region-shadow/score.llm.json \
  --human-decisions .build/region-shadow/human-decisions.json
```

## 发送与不发送

发送：区域 crop、9 个数值特征、分类、能力快照、prompt。
不发送：人工决定/备注、description、整页、OCR 文本、路径、bbox、region_id。

响应只保留哈希和经过校验的 agent 决策；`response.body` 留在 gitignored 的 `.build`，不发布。
密钥只注入 secure runner 的子进程环境，不写入 job、settings、报告或 git。

## 人工标注与范围评分

`human-decisions.json` 的版本为 `region-human-decisions/0.2`。除 `decision/source/note`
外，可记录：

- `annotation_level`: `proxy | explicit | trial_veto | calibration`；
- `retain_scope`: `none | partial | whole_region`；
- `fidelity_target`: `pixel | balanced | editability`；
- `required_edit_targets`、`allowed_native_parts`、`must_retain_parts`。

六个字段必须同时出现；`required_edit_targets` 必须是 `allowed_native_parts` 的子集。
旧的只有三字段的人工决定仍然有效，但不参与范围评分。

`apply-llm` 与 `validate` 使用相同的人工决定规范化逻辑，比较全部范围字段。
若权威文件已包含范围而旧计划仍只有三字段，校验会要求重新 apply 已保存响应；
不应删除范围标注来绕过校验，也不需要因此重新调用模型。历史报告保持原样。

保存的计划可进入本地候选工作流，详见 `docs/candidate-workflow.md`：当前会话编写
候选 IR，工具编译并重导入最终 PPTX，再由会话按对象 ID 记录视觉问题。此流程
不新增模型 API 调用，也不把视觉复检自动转成人工验收。

`score` 现在输出两层：

- `routing`：三分类标签一致率，以及始终返回 `retain_module` 的基线；
- `scope`：拆分范围一致率，并分别统计
  `over_decomposition`、`under_decomposition` 和 `retain_violations`。

`scope.agreement` 要求标签一致且没有范围冲突。P5 是固定反例：标签同为 `hybrid`，
但模型把 KPI、图表和表格内部都列成原生，人工只允许标题和标题图标原生并要求保留
`dashboard_body`，因此路由一致、范围不一致。

## 完整性绑定

`input.json` 的绑定包含 request/crop/features/capabilities/prompt/endpoint/model 哈希。
发送前，secure runner 与 worker 都会调用 `job_binding.verify_job()`：

- 重新计算 `request.json` 和 `crop.png` 的哈希；
- 解析 request，确认 model、system prompt、classification、features、capabilities
  和 base64 图片都与绑定一致；
- 只允许 settings 中的 endpoint 生成绑定记录的那个端点。

`transport.json` 记录实际 `request_sha256`、`endpoint_sha256`、`job_binding_sha256`、
`model` 和 response 哈希。`apply-llm` 会：

- 重新核验 job 绑定；
- 要求 transport 的上述指纹与本 job 一致；
- 重新计算 `response.body` 哈希并与 `saved_response_sha256` / `response_sha256` 比对；
- 拒绝缺失哈希、跨 job 交换、凭据回显或脱敏后的响应。

因为这些字段是新增的，旧 job 的 `transport.json` 不符合新要求；需要重新 `prepare-llm`
到新的 jobs root 后再发送，不能复用旧证据。哈希只证明文件完整性及版本关联，不构成
对可同时改写证据和清单者的签名认证。
