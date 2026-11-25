# image2PPT 图片转可编辑 PowerPoint

在本研究工作树中，正式入口是 OpenCode 的 `POST /api/v1/jobs`。上传幻灯片原图后，宿主识别文字和版面、重建对象并生成经过 PowerPoint 检查的基线。Agent 根据源图和疑难对象做局部修复。执行模型与独立视觉复核模型由部署配置指定。[运行和部署说明](services/product/OPENCODE_RUNTIME.md)

在含有 `services/product/opencode_client.py` 的工作树根目录运行以下命令。示例使用已配置的客户端凭据文件，将图片路径换成实际值。提交返回的任务编号可用于查询状态，任务结束后用同一编号列出可下载文件。

```powershell
$job = python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi submit C:/slides/page.png | ConvertFrom-Json
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi status $job.id
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi artifacts $job.id
```

交付按实际动作验收。单改文字或数字后要能保存重开；移动卡片时标签和图标要跟随；给底框换色后照片能单独删除，露出新底色。AI 原图上无法确认的业务文字应保留识读依据和待确认状态。复杂素材可作为单独图片，整页栅格不能算可编辑交付。用户可下载的文件以该任务的 `artifacts` 清单为准。[项目验收要求](AGENTS.md)

## 工作流与使用范围

| 目的 | 入口 |
|---|---|
| 自动上传、工程开发与真实编辑验收 | 工程助手使用当前工作树的 [Codex 工程技能](skills/image2ppt/SKILL.md)。每个 OpenCode 任务加载本工作树的 [运行技能](skills/image2ppt-runtime/SKILL.md)。 |
| 复现 r10/R12 冻结对照 | `services/product/run_local.py` 运行历史的 `parallel_review`。记录是否复用感知证据。[r10/R12 说明](services/product/PARALLEL_R10.md) |
| 显式半自动对照 | 由 Codex 看图、拆分字段并布局，单独标记为半自动结果。[历史手工流程](skills/image2ppt/references/legacy-manual-workflow.md) |
| 固定编译器与离线回放 | M1 的 `python -m image2ppt` / `image2ppt-core` 支持不可变修订、恢复与对象检查。下列命令用于编译器调试。[核心引擎说明](docs/core-m1.md) |

## 在 Claude Code 中调试编译器

Claude Code 加载 `~/.claude/skills/image2ppt/`（由维护源 `skills/image2ppt-claude/` 部署）。它的 shell 是 bash，
而 `.\image2ppt.ps1` 在 bash 下是语法错误（`syntax error near unexpected token`），因此命令请走转发脚本：

```bash
./image2ppt.sh doctor
./image2ppt.sh compile -InputPath runs/具体任务/slideir.json
./image2ppt.sh compose -InputPath 'runs/<run02>/slideir.json,runs/<run03>/slideir.json' -OutputPath outputs/deck
```

两个 skill flavor 各自有独立维护源：Codex 版在 `skills/image2ppt/`，Claude Code 版在 `skills/image2ppt-claude/`。
两者不能互相复制，也不要靠字符串替换从 Codex 版生成 Claude 版；改完维护源后重新导出（见 `exports/`）。
另外 `image2ppt.ps1` 含非 ASCII 文本，必须保留 UTF-8 BOM，否则 Windows PowerShell 5.1 直接解析失败。

## 固定编译器与历史原型命令

```powershell
.\image2ppt.ps1 doctor
.\image2ppt.ps1 demo
.\image2ppt.ps1 compile -InputPath '.\examples\synthetic\slideir.json'
.\image2ppt.ps1 prepare -InputPath 'D:\路径\幻灯片.png'
.\image2ppt.ps1 compile -InputPath '.\runs\具体任务\slideir.json'
.\image2ppt.ps1 perturb -InputPath '.\runs\具体任务\slideir.json' -NodeId title -Action replace -Value '新的标题'
.\image2ppt.ps1 compose -InputPath 'runs\任务A\slideir.json,runs\任务B\slideir.json' -OutputPath 'outputs\deck'
.\image2ppt.ps1 test
```

在 bash / Claude Code 中，把 `.\image2ppt.ps1` 换成 `./image2ppt.sh`，参数不变。

历史手工流程通过 `prepare` 建立输入和待分析结构，由 Codex 看图并填写 SlideIR。`demo` 使用固定合成样本验证编译工具链。

已经验证的合成 PPT 与实际文件预览保存在 `outputs/smoke-20261014/`；可复用的输入和结构保存在 `examples/synthetic/`。

最新的可独立复编译示例位于 `outputs/prototype-v0.1/`，包含源页面与资产副本。可运行 `compile -InputPath '.\outputs\prototype-v0.1\slideir.snapshot.json'` 生成新的输出版本。

每次准备输入都创建一个新任务；每次编译创建新的输出版本，包含 `result.pptx`、`preview.png`、`slideir.snapshot.json` 和 `quality.json`。原始输入不覆盖。详细协议见 `skills/image2ppt/references/slideir.md`。

输出版本附带源页面和图片资产，`slideir.snapshot.json` 可独立重新编译；原始未规范化输入仍保留在任务的 `source/original.*`。

## 历史原型的能力范围

- 单页图片、方向校正、原件保存、分析副本、局部裁剪。
- 原生文字、简单矩形/圆角矩形/椭圆/线条、图片模块和实际 PPT 分组。
- 文本和资产内容归属校验、文件哈希、独立输出版本、PPTX 结构检查及实际文件重新导入预览。
- 删除、移动和改写的扰动副本。测试不会覆盖原始 IR。

早期手工原型将识别、擦字与背景修复留给作者。M1 支持固定清单的逐页回放。`compose` 按传入顺序把已完成的 SlideIR 打包成多页演示文稿。当前编译器是否支持原生表格和图表以当前工作树的 `doctor` 结果为准。

## 运行环境

使用 Codex 内置 Node/Python 与 Presentations 工具，PPT 引擎为 `@oai/artifact-tool`。本机默认 Node 包目录为空，因此本次从已经下载的 Codex 运行归档恢复到项目的 `.runtime/bundle`，未修改全局依赖。`doctor` 输出实际使用的路径和版本。

项目编译器先尝试当前内置依赖，再尝试项目内恢复副本；其他机器需提供可用的同类环境。可通过 `IMAGE2PPT_NODE`、`IMAGE2PPT_PYTHON`、`IMAGE2PPT_NODE_MODULES` 覆盖运行路径，或设置不进入版本库的 `runtime.local.json`。这不是已验证的干净机器安装包。

预览由 Artifact Tool 重新导入最终 PPTX 生成。它不等于 PowerPoint/WPS 渲染；目标软件打开、修改及人工验收独立记录。质量报告始终保留 `human_acceptance: false`，脚本成功不会自动宣布通过人工验收。

