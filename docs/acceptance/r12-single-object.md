# R12 单对象素材恢复验收

开发基线为 `21caac6faf660b528eb647927eb32c0f4b05ca32`，分支为 `codex/r12-single-object-sam-repair`。机器可读收据见同目录 `r12-single-object.json`。

## 实际结果

第 2 页真实新上传收据中的完整层叠图标 `a-icon-layers`，由 BLOCKED 恢复为独立透明 PNG，经正式 `product.parallel_pipeline.run` 暂存、组合和最终验收。结果保留原有 7 个 TextBox，新增 1 个 PNG。原有文字定义及 selected 记录逐项相等。

素材重试仍将分组背景面板 `a-deco-row-bgs` 的完整 bbox 作为保护区，导致背景采样不足。修正后，源像素与 Surface 的底板一致率为 0.98805646。文字、不确定文字、已提交 Owner Mask 和其他前景对象继续受保护。底板关系只允许获取背景证据，擦除仍通过原有 source_owner 检查。

实际 SAM 服务通过 SSH 连接 `192.168.1.103`，沿用已部署服务与固定 checkpoint。保留完整原始分割输出。原始 SAM 与 LayerD RGB 组合的重合成 P95 为 37.30，超过既有 24 门限，被拒绝。随后使用 SAM 支持的完整源像素连通组件与 Source Unmix 软 Alpha，P95 为 4.52，通过原有门限。SAM 分割本身不授予像素擦除权。独立验证器重新推导候选，并校验计划、模型版本和原始文件哈希。

## 复现入口

在产品工作树运行以下 PowerShell 命令。真实源图、历史任务和 LayerD 证据保留在本机 `outputs` 中，Git 仓库提交代码与验收摘要。

```powershell
python services/product/run_single_asset.py `
  --job outputs/r1206-fresh-final `
  --layers outputs/r1206-fresh/fresh-cache/layers/3f1daa8db89fd43e9a595361fcc320283cf456eea867c9c2b9bcf54adc91d371-4228be1a8affda14 `
  --object-id a-icon-layers `
  --output outputs/r12-icon `
  --strategy sam
```

`--strategy source-halo` 重放原策略，`--strategy verified-backing` 单独检查背景关系纠错后的结果。输出目录可重复使用，输入、代码、图层和证据哈希相同时读取缓存。证据被修改时拒绝复用。增加 `--office` 后，仅为合法新候选执行目标对象编译、PowerPoint 编辑和删除残影检查。

正式组合使用以下命令。`--output` 必须指定新的任务目录。`--prompt-key` 从隐藏输入读取已有授权凭据。

```powershell
python services/product/run_r1204_regression.py `
  --from-job outputs/r1206-fresh-final `
  --output outputs/r12-icon-final-replay `
  --layers outputs/r1206-fresh/fresh-cache/layers/3f1daa8db89fd43e9a595361fcc320283cf456eea867c9c2b9bcf54adc91d371-4228be1a8affda14 `
  --text-region-goal 7 --reuse-verified --retry-assets `
  --asset-candidate outputs/r12-icon/runs/35f1364441cefb9a/c/candidate-1 `
  --prompt-key
```

上传主链路的区域素材恢复同样支持旧方法失败后，按保护证据变化尝试 verified-backing，再按 `sam_object_budget` 按需请求单对象 SAM。默认每页预算为 1。R12 区域路径停止重复执行旧整页素材 SAM 重建。其他旧路径及 Shadow 保留。

## 验证与耗时

| 检查 | 实际结果 |
| --- | --- |
| 产品回归 | 274 passed，36.42 秒 |
| 固定核心 | 39 passed |
| 当前单对象 L1，复用原始 SAM | 4.046 秒 |
| 相同输入与代码缓存命中 | 0.046 秒，无新增模型或 Office 调用 |
| 独立目标对象 Office L2 | PASS，13.094 秒，另有最终组合复验 |
| 正式组合 | 约 120.36 秒 |
| 最终 Office 移动、缩放、删除及保存重开 | 16 项通过，范围外变化 0 像素 |
| 最终文字短替换、等长替换及保存重开 | 14 项通过 |
| 新图标删除残影 | 确认残影 0，无法解释像素 0，检查 1,774 个 Owner 像素 |
| Go 独立视觉复核 | PASS，接受轻微字体、字距和图标线条差异 |
| 第 5 页历史交付 | 739 个提交文件哈希有效，8 个 TextBox、9 个 PNG 保留 |

本次实际 SAM 作业为 1 个，最终 Go 调用为 1 次，三分支感知模型新增调用为 0。探索期保留了不同版本的本地候选与失败结果。正式组合含目标图标 Office 批次、整页 Office 批次和文字替换批次。成本字段与实际动作数分别保存在 JSON 收据中。

## 交付范围

最终文件位于 `outputs/r12-icon-final/delivery/`，包含 `result.pptx`、`stable-fallback.pptx`、整页预览和源图对照。全部当前候选、失败、SAM 原始输出和正式验收文件保存在 `outputs/r12-icon` 与 `outputs/r12-icon-final`。

本轮使用真实新上传感知收据进行冻结重放，固定选择一个原失败完整图标。其余未通过素材继续保留栅格回退。第 5 页完成历史提交文件校验，本轮未再次调用 Office 或 Go。人工与 WPS 验收尚未执行。最终 QA 为 PASS，工作模式仍为 `parallel_review`，自动发布保持 GATED。
