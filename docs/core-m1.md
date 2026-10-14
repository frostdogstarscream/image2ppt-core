# image2ppt-core M1 使用与实现说明

M1 将当前确定性工具链封装为共享 Python 包。保存的 SlideIR、源图和素材可以在固定本机环境脱离聊天重建，生成实际 PPTX、最终文件回渲染预览、对象映射及质量证据。模型识别仍在当前对话中完成，代码没有远端模型调用。

## 环境与安装

本轮固定验证环境：Python 3.12.14、Pydantic 2.13.5、Pillow 12.3.0、Node 24.19.0、Artifact Tool 2.8.22。Python 依赖锁定清单是 `requirements-core.lock`。Artifact Tool 与 Presentations 校验辅助工具由本机 Codex 提供，不打包分发。源码包不是干净机器的一键安装包。

```powershell
$python = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
cd D:/Code/image2PPT
# 不安装也可使用共享核心
& $python -X utf8 scripts/core_cli.py doctor
# 可选：在已有依赖的环境离线安装，不下载包
& $python -m pip install --no-index --no-deps --no-build-isolation .
$env:IMAGE2PPT_PROJECT_ROOT = 'D:/Code/image2PPT'
& $python -m image2ppt doctor
image2ppt-core doctor
```

控制台命令位于安装解释器的 Scripts 目录，需要在 PATH 中。验收使用 `--target .runtime/core-installed` 独立安装及其 `bin/image2ppt-core.exe`，没有修改全局 Python。若不安装，可一直使用 `scripts/core_cli.py`。在受限运行环境中，可将 `TEMP` / `TMP` 指向项目内 `.build` 后再离线安装。

`IMAGE2PPT_PROJECT_ROOT` 指向完整工具链源码目录；`IMAGE2PPT_NODE`、`IMAGE2PPT_PYTHON`、`IMAGE2PPT_NODE_MODULES` 和 `runtime.local.json` 沿用原规则。doctor 记录解释器、Node、实际引擎与辅助工具路径/版本/哈希、已安装候选字体文件。字体清单不证明渲染器实际选择了哪款字体，后者明确未验证。环境变化会记入新构建，并使缓存失效。

既有 Artifact Tool 的正式文件提交使用硬链接。禁止硬链接的沙箱会失败并保留运行记录；需在允许该本地文件操作的环境恢复。M1 没有绕过校验或替换正式提交步骤。

## 公共命令

以下 `python` 表示上面的指定解释器。未安装时将 `-m image2ppt` 换成 `scripts/core_cli.py`。

```powershell
python -m image2ppt doctor
python -m image2ppt replay --ir examples/synthetic/slideir.json --output outputs/core
python -m image2ppt replay --build '<revision>/build.json' --output outputs/core
python -m image2ppt inspect --pptx '<revision>/result.pptx'
python -m image2ppt probe --build '<revision>/build.json' --cases configs/cases/synthetic.json
python -m image2ppt regress --suite configs/regression-m1.json --output outputs/regression
python -m image2ppt resume --run '<output>/run-ID'
python -m image2ppt --timeout 600 replay --ir '<bundle>/slideir.json' --output outputs/core
```

`--timeout`、`--project-root` 是全局选项，放在子命令之前。默认单个编译/渲染子进程超时 300 秒；resume 沿用该运行记录的配置。Windows 使用本次进程的 Job Object，其他平台使用进程组回收；不会按进程名终止其他应用。仅本机 Windows 环境已实际验证。

标准输出是一个 JSON 对象，辅助工具日志送往 stderr。成功完成流程退出 0；契约/输入/单次执行异常退出 2；多页回归或探针部分执行失败退出 3。**退出 0 不意味着质量通过**，必须查看 `quality_status` 和各检查项。

保留 PowerShell 的 doctor/demo/prepare/compile/validate/inspect/perturb/test 和原参数；新增：

```powershell
./image2ppt.ps1 replay -InputPath examples/synthetic/slideir.json -OutputPath outputs/core
./image2ppt.ps1 regress -InputPath configs/regression-m1.json -OutputPath outputs/regression
./image2ppt.ps1 resume -InputPath '<output>/run-ID'
```

PowerShell replay 默认导入 IR；构建快照、完整 inspect/probe 参数用 Python 公共入口。旧 `perturb` 修改 IR 后重新编译；新 `probe` 直接编辑最终 PPTX 副本，两者证据含义不同。

## 模块及数据边界

| 模块 | 责任 |
|---|---|
| contracts | Pydantic 输入包、Provider、运行/阶段/构建记录、检查和回归清单 |
| providers | ManualDraftProvider 接收保存的合法 IR；ReplayProvider 验证构建清单并读取独立快照 |
| prepare | 源图和素材路径约束、哈希校验、独立复制、规范化相对路径 |
| compiler / processes | 参数数组调用固定 JS 验证器/编译器/渲染器、工具链指纹、超时进程所有权 |
| quality | 实际 ZIP/OOXML 对象、分组、文字和 runs 检查；受限副本编辑及局部证据 |
| storage | 原子 JSON、文件清单、运行锁、哈希与 SQLite 派生索引 |
| runner / cli | 固定阶段、失效缓存拒绝、失败恢复、修订提交、串行回归 |

SlideIR 保持 0.1，唯一正式业务验证入口仍为 `scripts/ir.mjs`。新增构建/探针协议使用独立 `format_version: "1"`，不增加旧 IR 隐含语义。原始模型回答、任意模型代码及未经适配的草稿不被接受。历史模型调用记录标记 `not_provided`。

## 产物、缓存和恢复

每个 replay 创建新 `run-ID`，保存导入输入、run.json、阶段工作目录和 revisions。正式修订含 `source/`、`assets/`、`slideir.json`、`result.pptx`、`preview.png`、`build.json`、`review.json`、`objects.json`、`config.json` 及旧编译校验记录。复制单个修订目录即可重放；无需原输入路径或旧聊天。Manifest 内历史阶段位置只用于溯源，重放不依赖其继续存在。

执行链：导入与正式验证 → compile → 实际 PPTX 再导入 render → check → 暂存目录原子改名提交。编译器内部兼容预览仍保留，新引擎再进行独立 render 阶段。每个文件绑定 SHA-256 和大小，review/object map 绑定最终 PPTX 哈希。build.json 的哈希保存在 run.json 及 SQLite 索引；manifest 不自我哈希，也不提供数字签名。

输出目录在工具链工作区外时，适配层先在工具链 `.build/core-input-*` 编译，经过旧引擎正式验证后将完整结果复制到运行暂存目录。该编译工作目录保留供排查，保证旧引擎要求的工作区约束与 CLI 自由选择输出目录兼容。

缓存作用域是单个运行的阶段记录，非跨项目全局缓存。键覆盖输入内容、运行配置、Python/Node 实现指纹与引擎/辅助工具/字体环境。仅已完成且文件哈希一致的阶段复用；损坏阶段重算，损坏正式修订保留并生成新修订。导入输入被改动时拒绝恢复，修改作者 IR 应重新 replay。恢复不依赖原始导入路径继续存在。

SQLite 是可重建的运行、阶段与修订索引，不是唯一数据源。运行锁避免同一运行并发写入；回归按页串行，单页失败保存错误并继续后续页。崩溃遗留工作/暂存目录保留供排查，不自动删除历史产物。

## 检查与已知边界

`PASS / FAIL / REVIEW_REQUIRED / NOT_RUN` 均带对象、方法和证据。存在已知重建缺陷时构建仍可 COMPLETED，质量为 REVIEW_REQUIRED；契约、哈希、渲染失败不会提交为已验证结果。

- F01：03 页删除 text-072/073，保留 image-103，输出局部前后/差异图。
- F02：05 页删除 text-054/055/058/059，记录 image-062 与原生对象的实际相交面积、层级，输出前后对照。几何相交只提示风险。
- F03：02 页修改 shape-011 深底，保存 image-012 的 alpha/边界信息及白底、深底、棋盘格诊断；无 alpha 不自动判错。
- F04：逐字符对照 IR runs 与最终文件显式/段落默认颜色及字重。无法解析的主题/母版继承标记待复核，不猜测。当前蓝色加粗 runs 被统一覆盖可确定性检出。
- 分组移动：直接修改 card-group-015 的组偏移，检查成员相对变换不变，保留原位置局部图。

探针仅支持删除原生文字对象、移动真实分组、改变原生形状底色；原始 PPTX 必须保持原哈希。所有操作重新检查 ZIP 关系/对象结构并通过同一 Artifact Tool 回渲染。探针 ROI/delta 使用目标预览像素（96 dpi），不是源图像素。配置示例见 skill 的 `references/core.md`。

真实样本语义残字不做未经验证的自动 OCR 判定。仅显式标注的合成统一底色 ROI 使用固定像素阈值检查。PowerPoint/WPS 人工编辑、人工视觉验收均保留 NOT_RUN/false。M1 不修复 F01—F04，不实现远端模型、抠图、背景修补、原生表格/图表、语义连接器、嵌套分组、多行混合 runs 或服务化。

## 复核与测试

```powershell
./image2ppt.ps1 test
python -m unittest discover -s tests -p test_core.py -v
./image2ppt.ps1 demo
# 使用上一步生成的 IR 路径
node scripts/check_demo.mjs '<synthetic-run>/slideir.json'
```

`tests/test_core.py` 的 FakeBackend 只验证阶段编排和故障处理；真实导出、重新导入及探针由固定回归和合成 demo 另行验证。默认 `configs/regression-m1.json` 引用本地保存的 v2 02/03/05，固定 IR、源图、素材哈希及页序。这些业务源文件不放进通用源码压缩包；复制项目原有输出或提供自己的 suite 后才能运行该真实回归。

`scripts/verify_m1.py` 支持核验 14 页 IR、使用安装目录从不同 cwd/新进程重放回归结果的 build 快照，并证明输入和原始结果哈希不变。验收报告分别列出实现测试、已复现缺陷、待复核及未执行项。

本次完整结果见 [M1 验收报告](m1-acceptance.md)。额外负对照配置 `configs/cases/synthetic-negative.json` 故意保留复杂视觉图片并要求该区域纯白，预期 `labelled_residue: FAIL`，用于证明实际回渲染的检查不会对所有样本一律通过。

## 源码导出

```powershell
python scripts/package_core.py --output dist/image2ppt-core-m1-new-version.zip
```

导出使用白名单和独占新建模式，不覆盖旧包，完成后验证 CRC 和逐文件哈希。解压到新目录后，将 `IMAGE2PPT_PROJECT_ROOT` 设为解压的源码根；本机专有依赖位于旧项目 `.runtime/bundle` 时，还需将 `IMAGE2PPT_NODE_MODULES` 指向那个现有 node_modules 目录。用 doctor 核验实际环境后再 replay。ZIP 不包含 Codex 引擎、业务图片或真实回归运行产物；默认真实 suite 需另行提供其引用输入。
