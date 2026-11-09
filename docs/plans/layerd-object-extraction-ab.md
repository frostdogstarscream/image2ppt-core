# LayerD 完整对象提取与 SAM 对照

当前仍为实验实现，默认对象路线关闭。第 5 页已确认的未解决问题：完整云图替换失败后保留旧左右半图；背景像素归属检查会拒绝部分透明候选；文本框矩形的一像素相交会使整对象回退。当前上传保留这些问题的真实状态，运行或编辑检查通过不代表完整透明对象验收通过。

实施分支：`codex/layerd-object-extraction-ab`，基线 `9a95d3b`。
独立 worktree 保留根工作区及其他实验工作区的所有修改。引入的本地模块逐文件来源记录位于 `outputs/object-extraction-ab/import-provenance.json`。

## 已确定的实验

- 固定现有 12 页源图、RGBA 图层，以及已交付 Terra 版本文字和版式；该基线不是人工验收结果。
- MoMA 与 GPT-6.1 Terra 分别输出 `LogicalObjects/1`，同一个对象计划冻结后分别进入 Alpha、Alpha + SAM 3、Alpha + SAM 3.1。
- Alpha 优先提取；SAM 只处理 Alpha 歧义对象。定位框是提示，不是硬裁切边界。
- 保留原 RGBA 部分透明、孔洞及可见贡献；排除独立文字。添加对象和删除原有图像归属作为一个事务提交，失败整对象回退。
- 输出透明 PNG，可独立移动、缩放、删除；SVG 内部路径编辑不在本轮范围。
- 同一 SAM 后端用于两版最终 PPTX。SAM 3.1 必须通过单帧接口与权重检查，再做完整性、细线、文字夹带和错误合并比较，不能根据视频提速推断静态图更好。

## 运行入口

在本 worktree 根目录运行，输出必须新建：

```powershell
python services/product/object_experiment.py prepare --root outputs/object-extraction-ab
python services/product/object_experiment.py plan --root outputs/object-extraction-ab --planner moma
python services/product/object_build.py --root outputs/object-extraction-ab --planner terra --route alpha
```

MoMA 凭据可通过进程环境注入；不要写进命令参数或文件。`object_experiment_server.py --root outputs/object-extraction-ab` 提供仅监听本机、仅内存保存密钥的密码输入页。Terra 计划由明确委派的 Codex 子代理生成，记录其上下文方式，不宣称是独立部署的 Terra API。

SAM 在 4090 上使用独立环境，不升级当前 LayerD venv。`services/segmentation/single_frame.py` 对两种版本调用同一 `build_sam3_predictor` / 单帧会话协议，要求明确指定源码 revision、权重 SHA256、本地权重和源图绑定；不自动下载或切换模型。最终 owning model 的 checkpoint 缺失或额外键会使任务失败。

## 后台接口

`POST /api/jobs?object_extraction=alpha|sam3|sam3.1|off|default`，旧调用兼容。
`default` 读取 `IMAGE2PPT_OBJECT_EXTRACTION`，未设置为 `off`；真实图验证通过前保持关闭。
启用路线需配置 `LAYERD_TOKEN_FILE` 和 `LAYERD_URL`，上传后自动请求 LayerD，保存源图绑定的图层包。已有文字流程不变，对象提取在固定编译前执行，OpenCode Go 仍复核最终文件。

SAM 可选配置路径由 `IMAGE2PPT_SAM_CONFIG` 指定。配置包含独立环境 `python`、上游源码 `source`、`code_revision`，以及 `models.sam3` / `models["sam3.1"]` 的 `checkpoint` 和 `sha256`。不存在有效 SAM 时明确回退，不能记为 SAM 成功。
对象规划最多初始一次加两轮修正，原流程的其他模型调用政策不变。对照实验不会根据复核悄悄改写冻结计划。

## 验证与交付

- 组件测试：完整碎片、透明边缘、遮挡、源图绑定、文字保护、唯一归属、整个对象回退及重复执行稳定性。
- 执行 `./image2ppt.ps1 test` 和 `services/product/tests`。编译器和 OOXML 未修改；批量导出适配器复用既有固定编译器的 `exportBatchSize` 参数。
- 12 页逐页比较最终 PPTX 回渲染，并在 PowerPoint 副本中进行对象移动/删除/保存再打开验证。图形完整性不能仅由字段或像素 MAE 判定。
- 交付 MoMA 和 Terra 两版 12 页 PPTX、整页预览及简短差异说明；未经实际视觉或人工检查的状态如实保留。
- GPU 无法连接、凭据不可用、无有效 SAM 推理等情况只阻塞对应阶段，不用占位结果冒充完成。

官方接口依据：https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py

## 2026-11-09 执行状态

- 4090 已通过 `192.168.1.138` 恢复 SSH，主机名和原主机密钥匹配；独立实验目录为 `F:\code\image2ppt\outputs\sam-object-ab-20261109`。未修改原 LayerD 环境、网络配置或防火墙。
- 官方源码已固定至 `f1647e8d0c1d280991f4d1b7d974c2704537fe66`，通过 Git bundle 传到 `sam3-official`。Python 3.12 独立 venv 已创建，依赖安装尚未完成。
- 用户指定 ModelScope 下载权重。使用 `facebook/sam3/sam3.pt`（SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`）和 `facebook/sam3.1/sam3.1_multiplex.pt`（SHA256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`）。支持有界分段续传和完整文件校验，未校验完成不得加载。下载遇到超时，SAM 推理仍未运行。
- MoMA 183 对象，首次 Alpha 87 个；按 Go 首次复核整对象回退 20 个，修正版 67 个。Terra 176 对象，首次 Alpha 105 个；回退 22 个，修正版 83 个。对象数量及粒度不同，这些计数不是同分母准确率。
- Terra 由一个 GPT-6.1 Terra 子代理连续查看 12 页源图和图层 checker，未读取 MoMA 输出。源图哈希、尺寸、图层引用和 raw/plan 一致性已验证，但不是逐页独立上下文或独立部署 API。
- 两版修正版的 67／83 个对象均通过实际 PowerPoint 移动、缩放、删除、保存重开检查，非目标变化像素为 0。MoMA 第 12 页首次出现正文渲染差异，只读重开原文件也出现同样变化，文字属性一致；首次失败材料保留，完整复测通过。
- 修正版最终 Go PowerPoint 视觉复核因连接错误未完成；首次 Go 复核的源图差异不能视为修正版验收。后台对象路线仍默认关闭。
- 产品测试 30 项、项目协议测试 39 项通过。Alpha 阶段人工复核稿位于主仓库 `outputs/deliveries/layerd-object-ab-20261109-alpha-r3`，包含两版 PPTX、每版 12 页 PowerPoint 预览、源图对照页面和简短说明。SAM 完整对照和最终验收尚未完成。

- 后续修复了局部变量覆盖复核回退清单的问题，增加后续成员回退测试。最新计数为全量应用首次 Go 图形意见后的 67／83 个；历史 r2 保留。Terra 修复版采用既有逐页分批导出通过实际文件回读，不修改固定编译器。
- 独立环境改为复用本机已安装的 PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128。重打包前验证各文件原安装 RECORD，局域网传输后验证新包 SHA256；不是重新下载的官方 wheel。Triton 使用与 PyTorch 2.8 对应的 3.4 系列。

## 同日真实运行结果

ModelScope 两份完整权重均通过 SHA256；4090 CUDA 和依赖导入通过。SAM 3 首轮全部 23 个真实页/规划任务、96 个困难对象完成；SAM 3.1 经有界关闭参数兼容桥后同输入全部完成，首次失败保留。受控三图形中细线召回均为 1，但 3.1 原始 mask 多余像素更多，不能当作真实图准确率。两版真实 mask 不完全相同。

应用同样硬约束和首次 Go 整对象回退后，Alpha 为 67／83，Alpha+SAM3 为 79／87，Alpha+SAM3.1 为 78／87。3.1 在 MoMA 第 4 页发生一次归属重叠回退。本组选择 SAM3；两份最终候选各 12 页，实际 PowerPoint 全部提取对象移动、缩放、删除、保存重开验证通过，非目标区域变化为 0。交付目录是主仓库 `outputs/deliveries/layerd-object-ab-20261109-sam3-r1`。

第 6 页 MoMA 主要图标选 layer-01，Terra 选 layer-02；冻结计划不做事后人工修正。该页差异主要由选层和归属驱动，SAM 未解决 MoMA 选错图层的问题。Terra 的上下文方式也与 MoMA 不同，不能将整组差异归因于模型本身。最终候选仍待 Go 最后一轮和人工复核，新路线默认关闭。

最终候选 Go PowerPoint 全 12 页复核完成：MoMA 79 条、Terra 95 条意见，均 FINDINGS_REMAIN，人工验收未完成。MoMA 最后一页首次回复有不存在的 extracted- 对象引用，原始回复保留为 FAILED_UNKNOWN_NODE；使用用户明确提供的密钥标准认证，让模型只修正引用，验证其他所有复核字段和顺序完全不变后记录新有效回复。已移除内部实验助手对 CC Switch 数据库凭据的读取。最终交付维持 REVIEW_REQUIRED，未启用默认新对象路线。
