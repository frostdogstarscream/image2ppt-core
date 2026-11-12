# r10 独立三分支

## OpenCode 自主工具运行路径

`orchestrator=opencode` 与 `visual_review_backend=opencode` 启用独立 Runtime。业务入口为 `/api/v1/jobs`，任务由 OpenCode Session 选择基础 MCP 工具、高级区域工具或流水线快捷工具。OCR、LayerD、SAM、图像观察、背景、Mask 和场景工具可以独立组合。场景归属由已有融合与像素检查推导，候选提交绑定父场景、代码、文件及实际 Office 收据。

Runtime Skill 的维护源为 `skills/image2ppt-runtime/SKILL.md`。启动器将其安装到每个任务的 OpenCode 工作目录。`tool_profile=basic` 是自主组合验收目录，省略四个高级快捷工具；默认目录保留全部工具。两种目录均由模型自主选择目标和调用顺序。

最终视觉结论由同一 OpenCode Session 读取原图、Office 渲染及编辑图后记录。提供图像的收据只记录图片已传给模型。Go 在该路径中没有必需收据或调用。内容、Owner、邻居、残影、编辑检查及最终文件哈希继续独立校验。尚未通过真实闭环的部署应以实际验收记录为准。

## R12 区域候选入口

`product.parallel_pipeline.run` 支持 `candidate_engine=r12-regional`，本地入口可用 `--candidate-engine r12-regional`，上传 worker 可通过 `IMAGE2PPT_CANDIDATE_ENGINE` 选择。默认 `off` 继续使用稳定路径。`r11-shadow` 保留，`r12-shadow` 仅收集本地区域候选，采用原有重建作为交付基础。

R12 在三分支融合之后运行文字 Mask 补全、背景比较、字体候选比较、逐区域实际 Office 验证及组合验证。局部渐变或可信 LayerD 候选通过时跳过 LaMa。每个区域保存依赖、选择与失败原因，已验证收据绑定实际文件哈希，重启可按相同父场景恢复。显式不可分关系和依赖区域共同回退，共享字体不建立像素回退依赖。

Go 或 Office 基础设施故障会保留独立区域候选并记录 UNKNOWN。实际发现的对象错误按依赖闭包回退，组合改变后重新编译及验证。交付保留 `stable-fallback.pptx`，所有 R12 结果仍为 `REVIEW_REQUIRED`，`parallel_auto` 继续受原发布门禁约束。真实 Office 删除残影、短文本／等长替换、整页独立视觉复核分别记录。

`run_r12_product_regression.py` 调用同一正式入口，可通过源图／三分支／OCR 哈希回放已有感知证据。其记录明确标注 `hash-bound-recorded-branches`，不计作一次新的上传感知验收。故障注入参数只存在于此测试驱动中。

## R12-06 三分支归属与失败路由

融合层按同一 OCR 行的水平阅读顺序核对序号与标题。背景素材可通过完整 parts 分别与多个 Surface 对应，记录逐底板关系。背景面板关系由源图连续背景样本继续验证，前景图标与不确定复合区域保留独立归属。Surface 本身继续作为原生背景候选接受现有像素门禁。

`peer_evidence.py` 提供只读采样保护和实际清除保护。前景优先使用绑定源图、父场景和 Owner 文件哈希的完整 Mask。具备背景关系及源像素连续性证据的面板允许参与独立局部背景拟合，目标文字仍由既有 TextMaskCompletion 证明完整笔画范围。未知对比像素继续受保护。每个区域保存 `peer-protection.json`、NPZ 与局部预览，记录对应对象、角色、Mask、采样和冲突像素。

`run_r1206_ab.py` 复用冻结证据执行 A 文字／Surface、B 旧三分支保护、C 修正三分支保护对照。此入口只做 L1，模型和 Office 调用为零。实际交付继续使用 `product.parallel_pipeline.run`，经原固定编译器及真实 Office 验证。

正式区域入口启用 `regional_agent` 与 `agent_skill` 时，可按 OWNER、Mask、背景或素材失败归因接入现有受控 Agent。先判断是否存在支持新策略的实际收据，基础设施 UNKNOWN 和已经尝试过的同证据素材策略保留原结果。每页默认最多一个文字修复 Agent 会话，两次决策、一次工具执行，模型调用继续使用全任务预算。新增 `recheck_text_ownership` 只重融合现有独立 OCR 和三分支证据，全部候选仍须经过原有暂存、Office 及最终 Go。失败工具保留当前区域回退，其他成功区域继续组合。

已验证种子可同步不影响已选 Owner 闭包的归属关系。源图、文字定义、字体或已选闭包不兼容时，记录 `verified-seed-reanalysis.json` 并继续受控区域重建，历史交付继续保留。第 2 页新上传采用本轮独立感知，分组底板修正后复用同次原始收据继续执行。实际结果与未解决对象见 `docs/acceptance/r12-06.json`，复核入口为 `summarize_r1206.py`。

## R12-05 局部扩展与工具目录

冻结扩展可用 `run_r1204_regression.py --from-job <verified-job> --reuse-verified --retry-assets --text-region-goal 8`。复用前核对不可变 generation、源图、文字定义、Owner、依赖关系、已用字体文件与原 PNG 字节。旧文字候选直接载入，最终组合继续执行实际 Office 和 Go。`--retry-assets` 顺序使用最新父场景重提取完整独立 icon，通过源组件、受限 halo 与 Owner 复算后暂存，再批量检查新增图片。缺少或 UNKNOWN 的编辑／残影收据保留暂存状态，明确失败按依赖闭包局部回退。

`run_local.py` 同时支持 `--retry-assets`、`--reuse-perception-from <job>`、`--reuse-verified-from <job>`。新上传先使用 `--fresh-perception`。同次上传发生工具故障时，可保留原始感知证据并仅补跑失败分支。复用文字与版面证据的续跑须在验收记录中写明来源，完整感知是否完成以三条分支的实际收据为准。

`run_r1205_agent.py --job <job> --layers <layers> --skill <installed-image2ppt-auto-agent> --kind text|asset --output <new-output> --prompt-key` 执行已有 RegionalRepairAgent。文字模式需要真实 `--failure` 收据。素材模式可从原 source-owner 闭包失败事务自动发现目标。工具目录提供 `retry_text_mask_completion`、`switch_background_candidate`、`retry_asset_extraction`，宿主仅注册真实可执行回调。背景切换额外需要 `--rejected-candidate` 和 `--rejection-report`。每次最多两次 VLM 决策、一次工具执行，候选就绪后继续由实际 Office 和 Go 决定最终质量。

`run_text_coverage_scan.py` 对全部尚未验证的文字执行 L1 扫描并记录逐对象阻塞。独立图片须完整通过源图、RGBA、保护区域、Owner 及重合成验证。素材批次的实际编译和 Office 包次数同时进入现有 adaptive_usage 计数器。background_candidates 统计文字背景，PNG 候选另记，比较成本时需注明两种候选的范围。相关真实记录见 `docs/acceptance/r12-05.json`。

## R12-03 自适应执行

`r12-regional` 默认采用 `regional_strategy=adaptive`。可信 LayerD 下层、连续背景拟合、Big LaMa 按需推进，首个通过源像素、邻居保护、实际 Office 编辑与残影门禁的区域候选即停止搜索。页内文字优先使用三分支已经选择并统一校准的已安装字体，不再为每段正文运行多字体 Office 排名。最终组合仍须独立验收。`regional_strategy=exhaustive` 保留上一版完整比较路径，`r12-shadow` 继续保留原交付基础。

轻微字体、行距、颜色与位置差异由独立 Go 记录到 `minor_deviations`，相关区域可获得 `ACCEPT_WITH_MINOR_DEVIATION`。内容、邻居、源归属、残影、裁切和实际编辑检查继续是硬约束。外部超时和预算耗尽记录 `UNKNOWN`。`workflow_states` 分别记录候选生成、内部区域提交、最终 QA 和正式发布资格，内部 generation 提交不授予自动发布资格。

`adaptive_budget` 支持 `region_candidates`、`complex_region_candidates`、`repair_rounds`、`model_calls`、`office_validations`、`max_seconds`、`cost_units`。默认分别为 2、4、1、12、60、2400、120。复杂区域增加额度须附实际语义笔画补全证据。成本单位为记录在 `cost-report.json` 中的操作权重，不代表金额。模型调用在实际发送前持久预留，缓存响应不再次计费；LaMa 请求共用此预算。相同代码、父场景、字体和输入的成功及失败收据均可恢复，不盲目重复失败。输入或策略改变会使缓存失效；有意恢复基础设施检查可更改请求中的 `retry_revision`。

新图验收使用 `run_local.py --image <source.png> --output <new-output> --candidate-engine r12-regional --fresh-perception --prompt-key`，该选项使用独立空缓存，三条分支重新调用 MoMA，本地 OCR 重新执行。正文被感知为多个连续行时，产品依据相同文字角色、独立 OCR 行和同一小卡片证据合并为逻辑段落，保存原始行、成员和新对象映射，随后才进行 SceneGraph 归属仲裁。强调 run 不参与这种合并。中断后的修复可以复用本轮原始模型收据，继续经正式入口融合与重建。

`run_r1203_visual_control.py` 使用实际 Office 删除预览构造缺字负对照，检验 Go 能否拒绝严重缺陷。`summarize_r1203.py` 校验 generation 哈希并统计候选、编译包、Office 包、替换包、耗时和实际可编辑对象；区域耗时与新感知耗时分别列出。独立空缓存的新图与带七个稳定素材锚点的回放具有不同素材编辑数量，必须明确报告。现有配置、GPU 部署、固定编译器和公共 SlideIR 协议沿用既有路径。

## AI 开发规范与实现状态

R12-04 在融合前对全分支一致的素材坐标方言进行显式适配。至少两个对象按 xywh 越界且全部对象、parts、exclusions 均符合合法 xyxy 时，生成带输入和代码哈希的坐标适配收据，再经原有源像素和 Owner 门禁生成候选。`run_r1204_assets_local.py` 与 `recovery.text_probe.probe_one` 支持真实冻结证据 L1 检查及候选哈希复用。

本地入口可用 `--text-region-goal 4` 在每页达到四个实际 Office 验证通过的文字区域后停止继续尝试。`--regional-agent --agent-skill <已安装的 image2ppt-auto-agent 目录>` 注册受控的坐标修复与文字重试工具，VLM 读取真实失败、调用工具、检查结果并决定接受候选或停止。Agent 接受表示候选就绪，实际 Office、最终 Go 与发布资格继续分别判断。`run_r1204_agent.py` 可仅执行区域决策，`run_r1204_regression.py` 可复用三分支结果完成集成验收；`summarize_r1204.py` 验证最终哈希、区域工具产物与实际编辑结果。默认稳定路径及 Shadow 入口沿用现有开关，回退文件使用不可变源图表示。

主规范为 `skills/image2ppt/SKILL.md`，细则见其 `references/parallel-product.md`；根目录 `AGENTS.md` 采用同一自动产品入口。Codex 负责开发、调试和验证，不得手写 SlideIR、替换后台规划结果或手工拼装交付来冒充自动生成。旧手工流程保存在 `skills/image2ppt/references/legacy-manual-workflow.md`，固定编译与历史代码继续保留。

局部对象事务是本轮整改要求：共享字体不建立像素提交依赖，独立区域失败不应扩散到全页。文字内容、位置和清除证据分别判断；source-plate 保留未知内容，禁止叠字或按 bbox 清空。当前 r10 基线仍含整页 Surface 门禁、共享样式联动和页级 QA 回退；规范更新不代表这些修复已接入或通过回归。

验收必须分开记录视觉正确性和编辑完成度。`reconstructed_pages` 现有含义只是含保留重建节点的页数，不是完整重建页数；新增覆盖指标应采用独立参考清单和真实归属像素，缺少分母时 recall 为 UNKNOWN。Office 事务外像素稳定不等于目标内部无残影，Go PASS 也不等于所有文字/图形已编辑化。保留 `REVIEW_REQUIRED`；`parallel_auto` 的发布资格未因本次规范更新而改变。

入口为 `product.parallel_pipeline`。网页和 `run_local.py` 默认 `parallel_review`；`serial` 不受支持，`parallel_auto` 尚未取得发布资格。原串行模块保留为历史对照，仅借用 Provider、固定编译、有效字体核对等工具函数，不执行其 `run()` 或 VisualPage 全页规划。

安装 `requirements.txt` 后，显式核验现有本地 OCR：

```powershell
python services/product/configure_parallel_ocr.py D:/Code/image2PPT/ppt-rebuild/models .runtime/ocr-models.json
$env:IMAGE2PPT_OCR_MANIFEST="$PWD/.runtime/ocr-models.json"
python services/product/launch.py --port 8770
```

可用网页现有密码框配置 MoMA 和 Go。密钥仅驻留进程内存。新服务使用新端口，旧服务不必停止。缺少 OCR、LayerD 或 SAM 时后台记录相应未知/失败证据，保留完整源图表示，不能标为透明成功。

`IMAGE2PPT_PARALLEL_CACHE` 指定本项目共享缓存，默认当前工作区 `.runtime/parallel-cache`。源图/OCR/图层和模型原始 JSON 均绑定实际输入、版本及提示。外部请求先持久预留；响应未知不自动重发同一请求。`state/parallel.sqlite` 保存分支状态、依赖及任务唯一 generation 提交；下载只读取该提交中的交付结果。

GPU 部署先运行 `python services/prepare_gpu_service.py outputs/r10-gpu-stage`。这会为 LayerD Docker 上下文及 SAM 服务树放入同一份 `services/resource_lease.py`。同一物理 GPU 的两个服务必须配置相同的 `IMAGE2PPT_RESOURCE_ROOT`（容器需绑定同一个宿主目录）及 `IMAGE2PPT_GPU_ID`（物理 GPU UUID），否则不能宣称跨服务互斥生效。锁覆盖模型加载和推理；LayerD 每任务结束释放本项目模型，不结束其他产品进程。

每个最终对象在 PowerPoint 副本中单独移动/缩放、删除、保存重开，检查事务外像素和兄弟对象结构。最终 Go 视觉检查失败或证据缺失会保留原图页，不调用旧串行链路。详细记录位于任务内部；交付仍仅 PPTX、整页预览及简短说明。`REVIEW_REQUIRED` 不代表人工或 WPS 验收。

验收计划与阶段完成记录见 `docs/plans/parallel-three-branches-r10.md`。测试成功只能证明相应实现；12 页真实重建、117 对象复用、GPU 实测和人工验收分别报告，不以合成夹具代替。

## 私有服务连接和本机环境

可在本工作区 `.runtime/parallel-services.json` 配置 `layerd.url/token_file/model_binding`、`sam.url/token_file`、`ocr_manifest` 和可选 `ocr_python_path`。只存服务地址及本机路径，MoMA/Go 密钥仍仅在进程内存。OCR 独立依赖路径必须位于当前工作区 `.runtime`，避免污染机器上的其他项目。三个 ONNX 每次使用前都核验 SHA；任务中不下载模型。

GPU 服务监听远端 loopback 时，应先按本机既有 SSH 访问方式建立端口转发；转发通道是部署连接，不是逐页传文件或手工启动推理。上传后 LayerD/SAM 均由客户端自动提交、查询和下载。远端 LayerD 的模型内容哈希参与任务身份；同源图、同设置、同模型的再次提交返回同一任务，客户端 `.remote.json` 可恢复中断查询。不要把服务令牌或 API 密钥提交到 Git。

`deploy_r10_layerd.py` 和 `deploy_r10_sam.py` 的 `--port`、`--task` 用于创建不同端口、不同名称的新实例；部署拒绝覆盖已有任务。SAM 状态原子替换期间的短暂 Windows 共享冲突会有限重读，持续不可读仍失败，不将其解释为成功。

所有已完成对象先暂存再实际编辑验证。文字框相交本身不构成冲突，稳定图标按实际 RGBA 保护；原图出现不能安全解释的文字边缘、邻居或背景时保留原图，不把未知像素删掉。旧稳定 PNG 若无法通过当前场景的独立源像素检查，保留历史字节并允许生成等价候选，不将旧自动结论当作正确答案。

最终 Office 检查缺失、最终预览与已有 Go 绑定不一致，均不能沿用旧 QA 提交编辑对象；整套安全回退仍是 `REVIEW_REQUIRED`。generation 清单同时绑定全部页、SceneGraph、归属图、事务、资产、QA、输入及交付文件。稳定 Catalog 只是可恢复索引，不是第二个发布指针。
