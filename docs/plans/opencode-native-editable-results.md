# OpenCode 原生编辑主线实验，第 08 页

2026-11-19。最新真实上传的指定编辑通过 **1/3**。能独立删除建筑照片，改中间卡片底色和单改「会议材料整理」仍失败。交付状态为 `REVIEW_REQUIRED`。评测使用正常上传、实际 PowerPoint 及保存重开。人工和 WPS 验收均为 `UNKNOWN`。

## 源码和输入

| 用途 | Git SHA 或 SHA-256 |
| --- | --- |
| Carbon 原始 HEAD | `304f7eb` |
| 含原有未提交实现的冻结快照 | `076034c22b1e72b475f710f9b5e3048864fcfa39` |
| 三路形状与修补消融 | `bd64b285fd09f9dfb65ebdf17dada292946809dd` |
| 最新完整真实上传 | `e0ce488a843efd29101f669ff8d4f2db9bd4daa4` |
| workbench-08 上传原图 | `3ed7ccbde6a908d2200290556b116005459f1b0324ca4df7dcb9cbb31382d48e` |
| 最新交付与同任务确定性基线 | `80c94cf3d220158cb259b6d21c9fffd387f75247ce9ed0761ae50fdf9222d892` |
| 最新 Agent 回退前候选 | `c4dcc991b1d075de2bc5d980f3c05faa5bc7cb2643afa06371536b581170cc7c` |

每次测试从原图重新上传。V3、OCR 和 LayerD 都有本任务证据。最新完整上传另外保留了 4090 同机 OmniParser 的 29 条原始检测。Go 执行模型为 `hy4-flash-free`，独立审查模型为 `hy4`。单页各组采用 180 分钟、128 步。冻结接口对单页拒绝 192 步，HTTP 422 的请求没有创建任务。双页默认预算为 180 分钟、192 步，本次未运行双页全量验收。

## 实际编辑与局部视觉一致性

| 指定编辑 | 确定性基线 | Agent 候选 | 最终交付 | 最终版局部 SSIM / ΔE00 p95 |
| --- | --- | --- | --- | --- |
| 中间大卡片换色 | 缺少目标原生形状 | 同左 | 同左 | 0.9411 / 1.911 |
| 单改「会议材料整理」 | 缺少内容匹配的文本框 | 同左 | 同左 | 0.2319 / 42.675 |
| 独立移走左上建筑照片 | 删除后保存重开通过 | 同左 | 同左 | 1.0000 / 0.000 |

照片操作采用运行前冻结的删除用例。PowerPoint 重开后对象 `r-001` 消失，变化 31,540 像素，检查区域外变化为 0。编辑后预览可见蓝色标题条、其文字和右侧照片保留。单独拖放的通用 Office 回执与此指定删除用例分别保存。

中间卡片的事后诊断找到了原图坐标 `[573,168,534,436]` 的外框。对真实 Office 输入进行局部诊断时，该区域有 4,455 个受保护像素，并包含尚未独立提交的「公文写作辅助」候选 `t-012`。现有容器分支跳过它。此诊断没有改变任何交付文件。

标题的 OCR 读为「会议材科理」。Agent 请求了源图局部复读，新的读法没有通过字形核对。另一处「政策解读与瓷询」在 Agent 候选中改为「政策解读与咨询」。校对保留了原始读法、局部图哈希和两次读法的字形拟合值。独立审查没有给出有效回执，选回了同任务基线。表格里的指定编辑通过数和这一处额外校对分别统计。

## 同一提交的三组消融

下表前三行均取 `bd64b28`，同原图、相同步数上限。最后一行还包含执行器和检测配置等变化，不能把与前三行之间的差异单独归因于 OmniParser。

| 4090 任务 | 方式 | 指定编辑 | 原生文本 / 与 OCR 完整行匹配 | 经 Office 验证的可编辑对象 | 原生形状 | 底图承载内容 | 整页 SSIM / ΔE00 p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0ebb29dcf9d44843bccf2cf7c9e9afce` | 仅形状拟合 | 1/3 | 28 / 23 | 42 | 4 | 50.33% | 0.9316 / 2.416 |
| `94e5688615e1417c8296228dd34ff18c` | 仅联合修补 | 1/3 | 27 / 22 | 30 | 0 | 58.45% | 0.9343 / 2.107 |
| `4b273fbc5f82468ab81f83c56ee05cbd` | 组合 | 1/3 | 29 / 24 | 44 | 5 | 49.70% | 0.9300 / 2.503 |
| `a19c007b39f94a978b06d09e17596431` | 最新完整上传 | 1/3 | 26 / 22 | 41 | 7 | 54.97% | 0.9364 / 2.013 |

最新交付中，原生可编辑内容占 32.60%，经验证可移动栅格素材占 12.43%，重复图片区占 0%。底图可见面积为 86.18%。这几个指标使用不同分母。该页只有一项冻结的特定文字编辑，内容匹配为 0/1。全页人工字段分母缺失，全页文字召回、全量对象形态正确率为 `UNKNOWN`。历史第 08 页底图承载内容占比 33.51%，历史值只用于诊断。项目平均 ≤15% 目标未验收。

## Agent、Office 与独立审查

| 实验 | 真实模型回复数 | 输入 / 输出 / 推理 token | 实际结果 |
| --- | --- | --- | --- |
| 冻结快照 `076034c` | 68 | 711009 / 9298 / 65791 | 任务 `7ef5d1d805d147989f19824cf50b424b` 超出 180 分钟，终态 `REVIEW_REQUIRED`，交付清单为空。没有可比较的最终 PPTX。 |
| `bd64b28` 组合 | 22 | 61268 / 5797 / 18028 | 独立复核未通过，交付已验证基线 |
| `e0ce488` 组合 | 19 | 61699 / 5599 / 22540 | 基线、回退前候选、最终文件的三项指定编辑均为 1/3。候选和基线的通用 Office 检查均为 PASS。 |

模型回复数来自 OpenCode transcript。回复数包含工具往返，不等同于增量修复轮数。单独在 4090 运行完整版 `hy4` 的合成旧数字残影、单位错误、卡片缺陷和正常对照测试，收到 HTTP 429，视觉能力验证为 `UNKNOWN`。真实任务的独立审查记录为不可用。两者分开留存。合成数据图表的 PowerPoint 实测将 B2 从 50 改为 60，保存重开后工作簿、系列取值和渲染均符合预期。真实业务数值图表编辑、合并单元格表格及照片压字的新增留出页验收为 `UNKNOWN`。

两次中断上传也保留了记录。`a1bf2c221b5c44869d4182499ad66145` 触发裁切参数不匹配。`9f9dac4cfd544768ac567c07170b6030` 成功执行 OmniParser，随后因标记卡片调用旧的背景拟合接口中断。两次均无交付文件。修复这两个断点后才有上表中的完整结果。

## 复跑与产物

在 PowerPoint 主机打开已检出的研究源码，在 SSH 隧道和 4090 隔离 runtime 就绪后执行。环境变量指向既有本机凭据配置、输入和新的输出目录。

```powershell
python services/experiments/run_native_benchmark.py --pages workbench-08 --input-root $env:I2P_INPUTS --profile $env:I2P_PROFILE --url $env:I2P_TUNNEL --output $env:I2P_OUTPUT --steps 128 --seconds 10800 --probe-office
python services/experiments/native_results.py $env:I2P_OUTPUT "$env:I2P_OUTPUT/page-comparison.csv"
```

形状和联合修补的单独对照分别使用 `--baseline-only --variant shape-only` 与 `--baseline-only --variant joint-only`，选独立输出目录。冻结集的 10 页、38 个用例已重新核对文件哈希，其中第 9、10 页与 workbench-01、03 完全重复。workbench-08 属于开发页。按先达到三个里程碑再扩量的顺序，全量业务页重建未启动。

4090 研究根目录为 `F:/i2oc/experiments/native-editable-e0ce488`。其中 `jobs/a19c007b39f94a978b06d09e17596431` 保存宿主基线和真实 Agent 尝试，`assessment/p08-e0ce488/workbench-08` 保存 Office 编辑副本、编辑前后图与逐项结果，`reviewer-control/result.json` 保存 hy4 合成能力探测。`F:/i2oc/experiments/native-editable-bd64b28/assessment` 保存三组消融和合成图表。原始业务图片、PPTX、编辑图、模型和凭据均位于实验环境，GitHub 只保存代码与脱敏结果。

涉及实现包括 `runtime/host_baseline.py`、`host_evidence.py`、`batch_backbone.py`、`marked_routes.py`、`text_reading.py`、`independent_review.py`、`service.py`，恢复层的 `container_stage.py`、`marked_surfaces.py`、`chart_stage.py`、`native_shapes.py`，以及 `services/experiments` 中的上传、真实编辑和评分脚本。所有路径均以研究仓库为根，运行时代码位于 `services/product/product/runtime`。固定编译器与公开 SlideIR 沿用原协议。

聚焦测试覆盖首次推理 429、成品退回基线、标记照片和卡片的真实接口、错误形态、编号校对授权、原生图表以及实际 Office 探针。协议使用 `./image2ppt.ps1 test`。最后一次实际重建以后新增的候选下载名、CSV 汇总以及审查 HTTP 状态字段通过聚焦测试或现有产物复算，其业务页结果引用上述实测提交。



命令从研究源码根目录运行。Python 测试路径为 `services/product;services/planner/src;services/vision/src;ppt-rebuild/src`。

| 检查 | 结果 |
| --- | --- |
| 合并时的产品回归，覆盖宿主、OpenCode、标记执行、裁切、原生主线 | 84 passed |
| 最新真实上传之前的运行时与原生主线测试 | 72 passed |
| 4090 上 OmniParser 触发的标记卡片适配修复 | 18 passed |
| 候选下载、宿主回退、原生主线、实验适配、裁切、编号执行和评分卡的最终聚焦回归 | 42 passed |
| 基准 CSV 哈希绑定、区域测量测试 | 2 passed |
| 固定协议 `./image2ppt.ps1 test` | 39 passed |
| 冻结输入与用例 | 10 页、38 个用例通过哈希检查 |
| 技能源同步及打包 | 同步 PASS，源包 CRC 和清单通过，技能入口归档 verified |

测试集之间有重叠，各行计数按实际命令独立报告。

```powershell
$env:PYTHONPATH='services/product;services/planner/src;services/vision/src;ppt-rebuild/src'
python -m pytest -q services/product/tests/test_host_baseline.py services/product/tests/test_native_first_mainline.py services/product/tests/test_experiment.py services/product/tests/test_region_crop.py services/product/tests/test_marked_execution.py services/product/tests/test_scorecard.py --basetemp=.build-pytest-final
python -m pytest -q services/product/tests/test_native_benchmark_results.py --basetemp=.build-pytest-reports
./image2ppt.ps1 test
python exports/sync_codex_skill.py --check
python exports/build_source_package.py
python exports/package_skill_entries.py
python services/experiments/holdout/check_inputs.py --project-root D:/Code/image2PPT --workbench-inputs D:/Code/image2ppt-Carbon/outputs/benchmark-ai-workbench-8p/inputs
```
