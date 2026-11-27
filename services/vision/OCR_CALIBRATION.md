# 通过 API 校准 OCR 文字

新增 `calibrate-ocr` 显式步骤：本地 RapidOCR 初识别后，由配置的视觉模型 API 读取原图文字裁剪，再复核差异。无需在本机部署视觉语言模型，不改变现有规划模型、默认服务配置、固定 PPTX 编译器或独立单请求模型基线。

## 输入、输出及应用策略

输入为原始图片、与该图片哈希和尺寸绑定的 VisionAnalysis 或 CVElementInventory，以及不含密钥的 API 配置。输出必须使用新目录。

第一轮每批最多 16 个原图裁剪，默认 12 个。图片旁仅提供稳定的文字 ID，不提供 OCR 转录。第二轮只处理不一致或不确定项，提供原图和两个候选，由模型选择 OCR、视觉转录、无法确定或非文字；不接受模型另写第三个候选。

默认自动采用的范围很窄：第一轮明确可读，第二轮再次选择该视觉转录，且只有不超过两个汉字的局部变化、变化比例不超过 25%。数字、拉丁字符、标点及大幅改写均保留原始文字并标记 `review_required`。这只是降低误改风险的实验策略，不是准确率保证；两个阶段用同一模型，也不能当作两个独立真值来源。

- `original-evidence.json`：不可覆盖的原始 OCR 证据。
- `calibrated-evidence.json`：结构兼容原输入的校准证据；保留原文字、坐标及 `calibration_status`。可作为下一轮规划/重建的输入。
- `calibration.json`：逐项原始转录、独立转录、复核结果、最终选择、原图裁剪、失败原因及哈希。
- `review.html`：本地逐项对照表，便于按原图人工核对。
- `blind/*`、`review/*`：每次请求、原始响应、传输状态、源图和请求绑定信息。包含页面内容，仅作本地项目证据。

状态 `agreement` 只表示两次转录一致；`machine_calibrated` 只表示通过上述机器规则；均不等于人工确认正确。失败、不确定或未完成的调用保持原文。这个步骤不查找 OCR 漏检的文字，不修复文字坐标、字体或非文字图形。

## 配置

```json
{
  "base_url": "https://moma.cmecloud.cn/v1",
  "model": "qwen/qwen3.5-flash",
  "timeout_seconds": 120,
  "batch_size": 12,
  "concurrency": 3,
  "request_options": {"enable_thinking": false}
}
```

模型名是本次用户接口提供的路由名，不声称它是 3B 参数模型。切换模型只修改这份任务配置；OpenAI-compatible 请求格式是否真正支持图片和附加参数，需通过实际调用确认。

配置文件不允许写入 API key。设置 `PYTHONPATH=services/vision/src` 后，可先离线准备：

```text
python -m image2ppt_vision.cli calibrate-ocr --input <原图> --analysis <OCR证据.json> --config <配置.json> --output <新目录> --prepare-only
```

准备不会联网。确认使用的接口和数据范围已获授权后，从仓库根目录运行隐藏输入入口：

```text
python scripts/run_ocr_calibration_secure.py --runs <新目录>
```

可在 `--runs` 后指定多个已准备目录；它们必须使用同一个接口和模型。先核对 `/models` 中存在所选模型，再运行盲读和差异复核。凭据只保存在进程内存和子进程环境中，不写配置、请求文件或报告。

已有安全环境变量 `VISION_LLM_API_KEY` 时，也可省略 `--prepare-only` 直接运行 CLI。每个请求只有一次网络尝试，无自动重试、无模型降级；原始成功/失败结果保留。未完成目录可以通过隐藏输入入口续跑未发出的请求，已发出过的请求不重发。已生成最终 `calibration.json` 的目录不能重用。

## 验证边界

离线测试覆盖源图绑定、盲读不包含 OCR 答案、坐标不变、敏感字符拒绝自动替换、模型增删 ID/虚构第三候选拒绝、传输失败和证据篡改时保留原文。真实试验结果单独保存于 `outputs/ocr-api-calibration-20261019/`，不能由接口成功率推导文字准确率。

## 接回现有 PPT 生成

三方合并入口比较原始 OCR、校准结果、当前实际 PPTX 及其对应 IR。当前 PPTX 的文字和对象清单必须先与 IR 一致，否则拒绝处理，避免用过期 IR 覆盖用户在 PPT 中的修改。

```text
python scripts/text_ledger.py reconcile --ir <当前slideir.json> --calibration <calibration.json> --current-pptx <当前deck.pptx> --slide 1 --output <新的单页目录>
node scripts/compose_text_ledger.mjs <新单页目录1> <新单页目录2> <新单页目录3> --output <新导出目录> --batch-size 80
```

每段文字保留来源、候选、当前实际文字、采用结果及决定。已经与校准结果相同的文字不重复修改；仅当当前文字仍为原 OCR 且校准满足原定规则时应用变化。当前文字与两个候选冲突时保留当前版本并标记待核对。图片内保留的文字不额外生成文本框。只含空白差异的既有排版保留；标点差异不归一化消除。

`text-ledger.json` 是本轮文字选择记录，`text-ledger.html` 提供逐项原图对照。复制原图和资产到新目录，保持图形、位置、字号、颜色、层级不变；混合样式文字仅允许修改不跨越样式边界的片段，无法安全保留样式时继续待核对。

导出前检查 IR 哈希及选定文字，导出后直接读取实际 PPTX XML，核对全部原生文本和对象清单。`text-ledger-verification.json` 的 PASS 只证明选定文字正确传递到文件，不证明这些文字与原图完全一致。存在冲突的版本继续保持 `REVIEW_REQUIRED`。

本轮是使用现有对象规划的文字整合路径，没有重新调用结构规划模型，也没有启用监督自动修复。对应真实结果见 `outputs/text-ledger-integration-20261019/`。
