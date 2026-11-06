# Image2PPT r7 修改计划与本轮落实记录

日期：2026-11-06。本文结合用户提供的 r7–r9 设计文档、代码审阅意见及本轮真实运行记录，说明已经落实的范围和仍需完成的验收。本文不是全项目正式验收证明。

## 1. 本轮目标与边界

本轮选择完成 r7 的工程闭环：对象证据适配、限定范围的归属仲裁、候选与底板的事务提交、失败恢复、实际 PPTX 检查，并生成完整 12 页审阅版。

- 以 r6 的 `TypographyBatch/1`、其对应 SlideIR、字体清单、源图、素材及文字掩码为冻结输入。源图用于保真判断，r6 局部场景用于保留现有编辑能力和回退；r4 仅为历史参考。
- 复用固定编译器及现有 Windows 4090 运行环境。本轮入口为确定性工程命令，不加载 Skill、不调用模型 API，也不重新运行全部 LayerD。
- 本轮实际重建只覆盖已经明确复核的第 2 页标签与第 8 页箭头。其余 10 页保留 r6 场景并参加整稿测试；“保留基线”不等于该页所有问题已经修复。
- 原文、数字、段落、runs、现有字体改善和非目标对象保持冻结。r8 的字体精修、r9 的复杂图标路径和自然段落重排继续留在后续迭代。
- 历史文件和失败尝试保留。业务图像、生成 PPTX、凭据和模型权重不进入 Git。

## 2. 计划及已实现的工程变化

| 子系统 | 本轮落实 | 明确边界 |
|---|---|---|
| 对象注册表 | 增加 `ObjectRegistry/2`；对旧 `/1` 显式执行 xywh→xyxy 转换，保存证据与输出归属 | 未取得像素证据的掩码保持 `UNKNOWN`，不能因 bbox 存在就授权擦除 |
| 掩码与仲裁 | 修复对象区分可见贡献、修复授权、保护和验收掩码；校验尺寸、来源哈希及冲突 | shadow 模式中的基线保留不是全页像素归属验证；未知归属继续待复核 |
| 事务 | 完整保存父场景和候选场景，所有必需 gate 明确 `PASS`、所有权及父版本校验通过后才原子发布 | 失败恢复完整父场景，不以源图裁片或 r3 卡片覆盖 r6 已接受对象；已发布版本不原地改写 |
| 修复候选 | 第 2 页恢复“交流主线”标签；第 8 页恢复三处流程箭头和底部双箭头，撤销对应错误短横线 | 先尝试基础原生候选；拒绝后最多尝试一次源形状透明 PNG。PNG 回退不宣称内部路径可编辑 |
| 关系检查 | 对上述复核元素检查轮廓、颜色、对比度、箭头尖端及方向 | 覆盖范围为 `reviewed_key_elements_only`，不是通用关系图全覆盖；其他页 `NOT_APPLICABLE` 不等于关系通过 |
| 编辑与去字检查 | 实际编译、渲染删除/移动/替换/改色副本，分别检查导出一致性、源笔画残留、允许变化范围和保护区域 | 检查通过只对应已执行的探针；不能据此宣称任意长文本都可自动重排 |
| 断点恢复及独立 QA | 冻结代码、输入、字体和运行时；常规恢复拒绝变更。独立 QA 允许新校验器在新目录检查原成品，保留父子哈希链 | 独立 QA 不重新生成候选，不覆盖父版本；只允许代码指纹变化，输入和产物必须继续匹配 |

事务依赖、对象包含关系及 PowerPoint 实际分组分别处理。现有独立图标和段落不因同属卡片而强制合并为一个不可独立编辑的对象。

标签 PNG 回退从蓝色区域独立拟合颜色并填补文字孔洞，不把原图白字嵌入新标签素材。未修改素材保持字节一致。

## 3. 实施与验收顺序

1. 冻结实际 r6 输入、12 页顺序、源图/IR/素材/字体/代码哈希，导入注册表；影子运行只输出归属诊断。
2. 用现有第 2 页白字浅底、第 8 页箭头缺失作为负例，确保新关系门槛能拒绝已知错误。
3. 按第 2、8 页验证候选与回退，随后验证第 3 页图标和段落，扩展到第 4、10、11 页及其余页。
4. 合成完整 12 页，渲染正常版和删除全部原生文字版，执行代表性编辑探针，输出逐页原图/r6/r7/去字对照及失败记录。
5. PowerPoint 应用检查、WPS 检查和人工视觉审阅分别记账；任何未完成项保持相应未完成状态。

本轮保留既有局部门槛：规则背景拟合 MAE≤2、P95≤5；轮廓偏移 P95≤1.5 px；修补边界新增跳变 P95≤3；源笔画残留比例≤1%；授权范围外底板改动为零。整稿与单页删字渲染 MAE≤0.1 仅为导出一致性门槛。

关系门槛另记录在源图绑定的 `RelationSpec/1` 中。缺少独立证据、关键元素缺失、保护区域改变和未执行的必需检查不能按“无报错”判通过。整页 SSIM/MAE 不单独决定交付。

故障测试包括：未知掩码拒绝写入、旧协议坐标转换、重复 owner、未完成 gate、父版本变化、并发写入、候选失败、提交中断及重试幂等性。新增测试位于 `ppt-rebuild/tests` 的 resolver、transaction、relation 和 r7 测试文件中；测试数量以对应运行日志为准，本文不另行估计。

## 4. 12 页本轮实测快照

以下记录对应 4090 的 `20261106-pilot01` 成品及 `20261106-qa02` 独立 QA。QA 没有重新生成 PPTX。

| 项目 | 已记录结果 |
|---|---|
| 页面数量和顺序 | 完整 12 页 |
| 实际更新页面 | 第 2 页、第 8 页 |
| 其余页面 | 保留 r6 基线 |
| 正常/去字导出一致性 | 12 页报告为 `PASS` |
| 冻结源笔画与独立 OCR 去字检查 | 12 页报告为 `PASS`；只证明对应检查范围 |
| 关键关系检查 | 第 2、8 页 `PASS`；其余页为 `NOT_APPLICABLE` |
| 当前自动硬失败 | 第 1 页封面 `paragraph-l003` 的 `replace_short` 探针改变保护区域 |
| 模型 API 调用 | 0 |
| 总体交付状态 | `REVIEW_REQUIRED` |

封面短文本替换报告记录新增越界像素 215、允许区域之外变化像素 142。进一步诊断确认新字形上下各超出原文本框约 8 px，未与其他登记对象相交，距下方蓝线最近 19 px。它是紧凑标题容器下的真实编辑越界，保留硬失败，不放宽检查或改变 r6 字体版式。报告也记录封面基线自身存在 bbox 外可见贡献，基线溢出和本次编辑新增越界分别分析。长文本压力探针的溢出也保留诊断；它不代表当前流程支持自然重排。

本轮共有 38 项编辑探针，16 项明确 PASS、22 项 REVIEW_REQUIRED；其中 4 项长文本压力探针识别到溢出，封面短文本替换是当前唯一硬失败，其余待复核项包括独立底板证据不足。工程回归实际执行为 141 项 pytest 通过、39 项协议测试通过。

本轮 PPTX SHA-256：

```text
dbe74cd0ac5f27116d2ebcd081c4d027c505957a085c0ce715733afb515fd273
```

后续在本机 Microsoft PowerPoint 16.0 完成应用自动检查：打开 12 页、修改第 3 页段落、移动独立图标、修改卡片填充并保存副本，报告为 `PASS_APP_AUTOMATED`；原文件哈希未变化。该报告并未覆盖封面短文本探针，不能取消上述硬失败，也不能代替全页人工视觉验收。

证据位置：

- 4090 父运行：`F:\code\image2ppt\outputs\reconstruction-r7\20261106-pilot01`。
- 4090 独立 QA：`F:\code\image2ppt\outputs\reconstruction-r7\20261106-qa02`。
- 本机 QA 副本：`D:\Code\image2PPT\outputs\reconstruction-r7-transport\qa.json`。
- PowerPoint 报告：`D:\Code\image2PPT\outputs\reconstruction-r7-20261106\powerpoint-checks\20261106-171309-853fac54\report.json`。

4090 交付清单在该应用检查之前生成，因此其中 PowerPoint 仍为 `NOT_RUN`；后补的应用检查以独立报告及同一 PPTX 哈希关联。WPS 未测试，全文视觉人工未验收。

## 5. 复现与继续处理

以下命令从真实 CLI 与现有运行路径核对。工作目录为 4090 的 `F:\code\image2ppt`。`replay01`、`qa-replay01` 是示例新目录，首次执行必须不存在；不要覆盖已完成的 `pilot01`、`qa02`。

```bat
image2ppt-local.cmd rebuild reconstruct-r7 --input outputs\review-4090-20261106-r6\batch.json --output outputs\reconstruction-r7\replay01 --mode shadow
image2ppt-local.cmd rebuild reconstruct-r7 --input outputs\review-4090-20261106-r6\batch.json --output outputs\reconstruction-r7\replay01 --mode gated --pages 02 08 03
image2ppt-local.cmd rebuild reconstruct-r7 --input outputs\review-4090-20261106-r6\batch.json --output outputs\reconstruction-r7\replay01 --mode gated --resume outputs\reconstruction-r7\replay01
```

代码与冻结输入均未改变时，可在同一运行中重新检查：

```bat
image2ppt-local.cmd rebuild qa-r7 --run outputs\reconstruction-r7\replay01
```

校验器代码更新后，使用显式新目录复核原成品，保留原记录：

```bat
image2ppt-local.cmd rebuild qa-r7 --run outputs\reconstruction-r7\20261106-pilot01 --output outputs\reconstruction-r7\qa-replay01
```

工程检查命令：

```bat
.runtime\core-venv\Scripts\python.exe -m pytest ppt-rebuild\tests -q -p no:cacheprovider
image2ppt-local.cmd test
```

如后续修改固定编译器或 OOXML 适配，还需运行合成示例、`scripts/check_demo.mjs` 和实际文件预览；协议变化按项目规定运行 `image2ppt.ps1 test`，并指向项目私有 Node/Python 环境。

继续工作的优先级：针对已定位的封面短文本编辑越界设计容器与排版修复，保持真实失败记录；扩大独立关系清单与未知归属的复核覆盖；之后推进 r8 字体候选诊断、局部匹配与样式修正。r9 的复杂矢量路径、自然换行和增删段落能力单独立项。

本轮已经完成一轮 12 页测试与审阅文件生成，但自动硬失败尚未全部清零，WPS 和全文人工视觉验收尚未完成，因此不能称为 r7 全部验收通过。
