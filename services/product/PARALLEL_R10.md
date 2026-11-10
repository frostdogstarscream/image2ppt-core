# r10 独立三分支

## AI 开发规范与实现状态

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
