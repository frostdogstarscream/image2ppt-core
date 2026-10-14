# image2ppt-core M1 验收报告

日期：2026-10-14。范围：用户确认的首个里程碑；保留现有 Artifact Tool。结论：**M1 工程实现与本机离线回放验收通过；真实页面质量仍为 REVIEW_REQUIRED。** 不将已知重建缺陷或未执行的人工检查写为通过。

## 实现交付

共享包在 `src/image2ppt/`，包含 contracts、providers、prepare、compiler、processes、quality、storage、runner、cli。公共 CLI 的 doctor、replay（IR / build）、inspect、probe、regress、resume 已实现；安装后提供 `image2ppt-core`。旧 PowerShell 命令保留，replay/regress/resume 转发共享核心。

SlideIR 未升级，仍由原 JavaScript 业务验证器校验。已增加保存输入、逐阶段缓存、失败恢复、不可变新修订、SQLite 派生索引和实际文件哈希绑定。模型调用记录缺失明确写为 not_provided。没有引入远端模型/API。

Skill 维护源与 `.agents/skills/image2ppt` 的 4 个文件逐字节一致，新增核心调用说明。官方 skill `quick_validate.py` 因现有 Python 缺少 PyYAML 未能运行；本轮未额外安装依赖，已人工检查既有 frontmatter、引用文件及副本一致性。这不影响核心运行测试。

## 验证结果

| 验收项 | 结果 | 证据与边界 |
|---|---|---|
| 原有协议测试 | 13/13 通过 | 坐标、内容归属、路径、分组、哈希、背景等原规则继续有效 |
| Python 单元/故障测试 | 24/24 通过 | 含重复 ID、缺素材/错误哈希、路径约束、原始草稿拒绝、损坏 JSON/PPTX、超时、渲染失败恢复、损坏缓存拒绝、版本/输入变化、SQLite 重建、单页失败继续、工具链目录外输出 |
| 子进程树超时 | 通过 | 合成子进程先写 started，再延迟写 survived；超时后 started 存在且 survived 不存在 |
| 14 页保存 IR | 14/14 兼容 | 原 v2 的 02—15 页；没有重新识别或改写这些 IR |
| 三页代表回放 | 02/03/05 全部完成 | 生成实际 PPTX、再导入 PNG、build/review/objects；哈希互相绑定 |
| 最终文件编辑探针 | 4/4 完成 | F01、F02、F03、分组移动；原件哈希未变 |
| 独立安装入口 | 通过 | 离线构建安装到 `.runtime/core-installed`；模块入口与控制台 doctor 输出一致 |
| 不同目录、新进程回放 | 3/3 完成 | 三个 build 快照经安装后的模块重放；控制台 resume 均复用已校验修订 |
| 输入保护 | 348 个文件哈希一致 | 14 页源图、IR、素材；原构建 PPTX 也保持原哈希 |
| 旧编译链合成 demo | 通过 | 原 demo 和 check_demo 四项扰动完成；三个原位置像素检查通过 |
| 新直接编辑合成探针 | 3/3 完成 | 删除文字、移动组、改底色；两个标注纯白 ROI 的残留检查通过 |
| 合成反例 | 按预期检出 FAIL | 保留图片文字区域却要求纯白；实际回渲染像素偏差 179.7624，大于阈值 1 |

单元测试中的 FakeBackend 仅建立编排/故障处理证据；实际编译、最终文件导入和探针另由真实回归与合成 demo 验证。反例是显式设计的负对照，不是新增产品故障，也不是语义 OCR 准确率证明。

## 已复现缺陷和待复核项

| 项目 | 自动检查结果 | 实际预览观察/后续边界 |
|---|---|---|
| F04 runs 丢样式 | FAIL | 02 的 intro-line-1，03 的 intro-line-1/2 存在 IR 与最终 PPTX 颜色/字重不一致；本轮没有修复导出算法 |
| F01 image-103 夹字 | 操作 PASS；语义 REVIEW_REQUIRED | 删除 03 的 text-072/073 后，局部图仍可见“生成”等图片内文字；这是对实际预览的观察，非自动 OCR 判定 |
| F02 image-062 夹字/遮挡 | 操作 PASS；语义 REVIEW_REQUIRED | 05 的目标原生文字删除后，局部裁片仍含边缘文字片段；报告提供相交面积、真实绘制顺序及前后对照 |
| F03 图标底色 | 操作 PASS；视觉 REVIEW_REQUIRED | 02 改深色卡片后，image-012 的浅底矩形可见；记录 alpha/边界及深浅底诊断，不因 RGB 模式自动判错 |
| card-group-015 移动 | 变换检查 PASS；视觉 REVIEW_REQUIRED | 修改真实组偏移且成员相对变换保持，原位置局部图可供复核 |

05 页没有检出的 runs 不一致，不代表该页整体视觉通过。所有正式构建均维持 `execution_status: COMPLETED` 与 `quality_status: REVIEW_REQUIRED` 的分离。

## 未执行与交付边界

- PowerPoint/WPS 打开、人工改字/移动和人工验收未执行，字段为 NOT_RUN/false。
- 未进行干净机器安装、无 Codex 依赖运行或其他操作系统验收。
- 未接远端模型；执行路径中的模型/网络 API 调用为零，没有进行网络抓包审计。
- 字体文件清单已记录，渲染器实际字体替代与原图字体识别没有被自动证实。
- 真实语义残字自动识别、主题/母版复杂样式继承仍需复核。
- 未修复 F01—F04，未扩展原生表格/图表、语义连接器、嵌套分组、多行混合 runs、自动抠图/背景修补或服务化。

初次本机执行遇到沙箱硬链接限制，保留失败运行后在允许本地硬链接的环境成功恢复。旧编译器标准输出的诊断日志与 JSON 混流已通过单独 bridge 分离；SQLite 连接关闭和 Windows 子进程所有权已通过故障测试验证。

源码包解压测试发现工具链工作区外输出会被旧引擎拒绝。适配层已增加工作区内暂编译、验证后复制完整产物的路径，未修改旧引擎校验约束。首个 ZIP 保留，最终交付使用 r2 修订；解压实测结果另存 `outputs/core-m1-validation/package-smoke-final.json`。

发布版为 `dist/image2ppt-core-m1-0.1.0-20261014-release.zip`，在 r2 相同引擎源码基础上更新本报告的最终证据索引。旧 ZIP 均保留。最终安装目录为 `.runtime/core-installed-final`。

## 可核查产物

以下为项目相对路径；通用源码包不包含真实业务源图、运行历史或专有运行时。

- 汇总机器报告：`outputs/core-m1-validation/acceptance-final.json`（含关键证据文件 SHA-256）
- Python 测试：`outputs/core-m1-validation/unit-tests-final.txt`
- 原协议测试：`outputs/core-m1-validation/protocol-tests.txt`
- 固定真实回归：`outputs/core-m1-validation/real/regression-ef429040d929/report.json`
- 独立进程回放：`outputs/core-m1-validation/independent-final/report.json`
- 14 页兼容：`outputs/core-m1-validation/independent-final/compatibility.json`
- 实际环境：`outputs/core-m1-validation/independent-final/doctor.json`
- 旧合成 demo：`runs/synthetic-2026-10-14T07-47-05-802Z-9caf5341/checks.json`
- 新合成正/负对照：`outputs/core-m1-validation/synthetic-probes.stdout.json` 与 `synthetic-negative.stdout.json`，内含完整探针报告路径

源码打包脚本 `scripts/package_core.py` 使用白名单，只含本项目源码、Skill、文档、配置与标注合成输入。每个源码文件记录哈希，打包后核对 ZIP CRC 与清单。重新导出使用新文件名，不覆盖历史包。
