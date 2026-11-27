# 视觉模型字体参数接入

本阶段在现有 API 元素重建流程上逐步改善字体，不修改 OCR 内容，不重新规划布局，不修改 skill。截图只能支持字体候选判断，不能证明原始字体元数据。

## 字体合同

新的整体元素计划使用 `ElementPlan/2`。每个原生文本要求 `font_family`（中文）、`font_latin`（西文）、`font_size_px`（原图像素 em 大小）、`bold`、`font_confidence` 和 `font_evidence`。旧 `ElementPlan/1` 可回放；没有显式字体时保留旧默认行为，不能将其报告为模型已识别字体。

还可只对已生成页面调用 `FontPlan/1`：模型返回样式表 `styles` 和文字到样式的绑定 `bindings`。视觉上相同的文字复用一个样式，页面标题、面板标题、列表正文、界面小字可以使用不同样式。每个原生文本必须恰好绑定一次；原始文本、位置、颜色、图片、图形和层级保持不变。已有混合粗细/颜色 runs 保留，样式的基础字重不覆盖它们。

当前合同覆盖字体族、字号、粗体和共享样式。行距、字距、斜体、缩进和布局约束尚未加入这一字体专项阶段，不应宣称完整排版恢复。

## 字体复用规则

- `Microsoft YaHei`、`雅黑` 规范到本机渲染链可用的 `微软雅黑`；宋体、黑体、等线等按已验证候选目录解析。
- 只发送项目已验证候选字体列表给模型。这个目录不是 Windows 全量字体清单，配置的 `VISION_FONT_MANIFEST` 可扩展候选。
- 缺少字体明确拒绝，不自动替换、不自动下载字体。部署机器和用户 Office 的字体可用性仍需分别检查。
- 字号换算：`size_pt = font_size_px × target_width/source_width × 72/96`。OCR 墨迹框高度不直接等于字号。
- 保存原模型建议、置信度、视觉依据以及实际采用的字体参数到 authoring sidecar。
- 后续本地字体拟合对这些文本只生成对照建议，保留整个原生文本节点，避免静默替换字体、字号、字重、颜色或位置。
- 文字台账接续保留模型字体记录。字体专项修订更新台账哈希，文字选择不变，继续进行导出文字检查。

## 独立字体阶段入口

```powershell
python -X utf8 scripts/font_plan.py prepare --ir <原页面/slideir.json> --output <新实验目录/job-05> --model qwen/qwen3-vl-plus --base-url https://moma.cmecloud.cn/v1
python -X utf8 scripts/run_supervised_jobs_secure.py <新实验目录>
# 隐藏输入凭据，然后输入 job-05。退出时输入 quit。
python -X utf8 scripts/font_plan.py apply --job <新实验目录/job-05> --output <新实验目录/page-05>
```

prepare 只准备本地请求。真实调用必须使用用户授权的数据和接口。每个 job 保留原始请求、响应及传输结果，失败和重试分别创建目录；禁止把失败响应当成字体结果。通过 `input.json` 检查源图、IR 和请求的绑定。输出目录必须是新的，保留历史版本。

使用现有固定编译器导出，带文字台账的页面可复用 `scripts/compose_text_ledger.mjs`。之后执行：

```powershell
python -X utf8 scripts/font_plan.py verify --ir <新实验目录/page-05/slideir.json> --pptx <实际导出的PPTX> --slide 1
```

verify 直接检查实际 PPTX 中每段文字的中文/西文字体、字号、字重和文本。PASS 说明声明的参数进入文件，不能证明原字体识别正确、运行时没有字体替换或视觉已经变好。必须再看实际文件回渲染，并将字体试验与布局修复分开评价。

## 字体相容性后处理

模型原始字体版保持不变。对出现字号与固定文本框不相容的候选，可使用独立的[字体相容性流程](FONT_COMPATIBILITY.md)，生成有记录的字号约束版，再通过实际 PPTX 回渲染选择候选或保留基准。该流程是显式后处理，不计作独立模型的原始结果，也不替代字体真值或人工验收。
