# image2PPT Codex 原型

现已增加 **image2ppt-core M1**：共享 Python 引擎、保存产物离线回放、不可变修订、阶段恢复、最终 PPTX 对象检查及直接编辑探针。入口是 `python -m image2ppt` / `image2ppt-core`。完整安装与调用说明见 [核心引擎说明](docs/core-m1.md)。SlideIR 仍为 0.1，旧 PowerShell 命令继续可用。

由 Codex 读取页面图片、理解结构并保存 SlideIR；固定工具将 SlideIR 编译为可编辑 PPTX，再重新导入最终文件生成预览。本项目的脚本不调用大模型 API。

## 在 Codex 中使用

在当前项目聊天中使用 `$image2ppt`，并提供一张清晰的幻灯片图片。如果新技能尚未出现在选择器中，可直接指定：

> 使用 `D:\Code\image2PPT\skills\image2ppt\SKILL.md`，把我提供的图片还原为可编辑 PPT，保留原文、数字、颜色和布局；生成实际 PPTX 预览，并检查移动、改字和删除后的表现。

专用技能负责忠实重建；不应用通用演示文稿的文案改写、默认字号和重新设计要求。

## 最短工具命令

```powershell
cd D:\Code\image2PPT
.\image2ppt.ps1 doctor
.\image2ppt.ps1 demo
.\image2ppt.ps1 compile -InputPath '.\examples\synthetic\slideir.json'
.\image2ppt.ps1 prepare -InputPath 'D:\路径\幻灯片.png'
.\image2ppt.ps1 compile -InputPath 'D:\Code\image2PPT\runs\具体任务\slideir.json'
.\image2ppt.ps1 perturb -InputPath 'D:\Code\image2PPT\runs\具体任务\slideir.json' -NodeId title -Action replace -Value '新的标题'
.\image2ppt.ps1 test
```

`prepare` 只建立输入和待分析结构；之后由当前 Codex 聊天查看图片并填写 SlideIR。`demo` 使用固定合成样本验证工具链，结构来自测试夹具，不代表真实图片识别效果。

已经验证的合成 PPT 与实际文件预览保存在 `outputs/smoke-20261014/`；可复用的输入和结构保存在 `examples/synthetic/`。

最新的可独立复编译示例位于 `outputs/prototype-v0.1/`，包含源页面与资产副本。可运行 `compile -InputPath '.\outputs\prototype-v0.1\slideir.snapshot.json'` 生成新的输出版本。

每次准备输入都创建一个新任务；每次编译创建新的输出版本，包含 `result.pptx`、`preview.png`、`slideir.snapshot.json` 和 `quality.json`。原始输入不覆盖。详细协议见 `skills/image2ppt/references/slideir.md`。

输出版本附带源页面和图片资产，`slideir.snapshot.json` 可独立重新编译；原始未规范化输入仍保留在任务的 `source/original.*`。

## 已支持的原型范围

- 单页图片、方向校正、原件保存、分析副本、局部裁剪。
- 原生文字、简单矩形/圆角矩形/椭圆/线条、图片模块和实际 PPT 分组。
- 文本和资产内容归属校验、文件哈希、独立输出版本、PPTX 结构检查及实际文件重新导入预览。
- 删除、移动和改写的扰动副本。测试不会覆盖原始 IR。

PDF 解析、自动 OCR、自动擦字与背景修复、原生表格/图表、多页图片自动识别、网站/API 尚未实现。M1 支持固定清单的逐页回放回归，不负责自动识别整套图片。复杂图表可以明确保留为图片；其中的数据不可编辑。复杂底图需要先提供经检查的清理资产，不能在原图文字上叠加可编辑文字冒充还原。

## 运行环境

使用 Codex 内置 Node/Python 与 Presentations 工具，PPT 引擎为 `@oai/artifact-tool`。本机默认 Node 包目录为空，因此本次从已经下载的 Codex 运行归档恢复到项目的 `.runtime/bundle`，未修改全局依赖。`doctor` 输出实际使用的路径和版本。

项目编译器先尝试当前内置依赖，再尝试项目内恢复副本；其他机器需提供可用的同类环境。可通过 `IMAGE2PPT_NODE`、`IMAGE2PPT_PYTHON`、`IMAGE2PPT_NODE_MODULES` 覆盖运行路径，或设置不进入版本库的 `runtime.local.json`。这不是已验证的干净机器安装包。

预览由 Artifact Tool 重新导入最终 PPTX 生成。它不等于 PowerPoint/WPS 渲染；目标软件打开、修改及人工验收独立记录。质量报告始终保留 `human_acceptance: false`，脚本成功不会自动宣布通过人工验收。

