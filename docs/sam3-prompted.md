# Qwen 描述与框驱动 SAM 3（实验入口）

此入口保留 SAM2 默认流程。SAM3 在独立开发镜像、GPU 1、无网络的临时容器中运行；不改动生产容器。模型权重来自 ModelScope `facebook/sam3`，使用明确的提交和 SHA-256 清单。官方源码固定提交，构建日志与实际运行环境保存在实验输出中。

## 输入与调用边界

`scripts/plan_segmentation.py run --target sam3` 生成 `SegmentationPlan/2`，每个叶子 `keep_pixels` 区域增加 `sam_prompt_en`（短英文视觉名词）与 `segmentation_mode`（object/composite）。旧 `/1` 契约继续支持原入口。原图、请求、响应与计划相互绑定；不接受后改框冒充模型原始输出。

`scripts/plan_sam3_batch.py` 是显式批次入口：最多十张 PNG、20 次请求；每页首次调用加一次格式校验失败重试。请求前写预算预留记录。网络、配额、认证或服务商错误立即停止，不切换模型、不自动重复付费请求。重新运行必须新建输出目录；跨批次总预算由操作者核对，不得用新目录绕过授权额度。导入模块和默认命令不会发起调用。

## 分割与对象归属

SAM3 先在原图缓存图像特征，逐对象清空提示，然后同时使用英文描述和正框。0–1000 xyxy 框转换成 0–1 cxcywh。框是识别提示，不是裁剪边界，因此还需排除异位置的同类图标。单对象选一个合适候选，复合插画合并位于目标区域的候选。

原始候选掩膜、分数与框保存在 NPZ；使用时核对原图、计划、尺寸与掩膜哈希。检查图形边缘覆盖及透明度归属；空结果、异位置、线条缺失或归属冲突均记录明确的矩形回退。回退不算 SAM3 成功。掩膜应用到原 OCR/LaMa 流程的去字像素，文字另外重建。SAM3 路线不会静默调用 SAM2；未被计划覆盖的内容沿用图像测量与残留恢复。

T4 使用 FP16 autocast，检测非有限输出后最多一次 FP32 重试并记录。固定 1008 模型输入分辨率，不静默降分辨率，不启用编译或多卡。模型/显存错误直接失败。合成冒烟只验证推理机制，不等于真实业务图准确。

## 执行

可选 `--sam3-box-policy crop-hard`：逐个按原始 Qwen 框裁图，在局部图上使用原英文描述与覆盖局部图的正框，掩膜再映射回整页；原框之外必须为零。模型输入仍为 1008，不增加规划调用。后处理不扩大框，跨界 OCR 进入明确回退；矩形回退也限于原框。记录 `box_policy` 与 `crop_bbox_px`，应用时核验越界。默认 `prompt` 保留旧的整页提示行为。硬框保证空间范围，不保证框内完整覆盖；原框偏小会截断图形，照片也可能仍只选中主体。

先构建 `deploy/sam3-worker/Dockerfile` 与运行 `sam3_worker --smoke`。只有完成 GPU 冒烟后才调用付费规划。工作器配置只含 `ssh_host` 和 `remote_root`，密钥留在原服务器。

```powershell
python scripts/plan_sam3_batch.py --inputs <十张源图> --output <新规划目录> --model-config <客户端配置> --model qwen/qwen3.8-max --max-calls 20
python scripts/layered.py --inputs <源图> --output <新目录> --worker-config <原开发工作器配置> --quality-policy quality-loop --segmentation-plans <规划目录/plans.json> --segmentation-backend sam3-prompted --sam3-worker-config <独立SAM3配置> --max-extra-vision-calls 0 --max-repair-rounds 1
python scripts/check_automatic_delivery.py --job <任务目录>
```

三路对照使用相同原图：SAM2 自动分割、同一新 Qwen 计划加矩形保留、同一计划加 SAM3。记录所有失败页与回退，不以对象减少代替还原质量。实际 PPTX 回渲染、编辑探针和源码绑定检查通过后，仍最多 `REVIEW_REQUIRED`，不代表 PowerPoint/WPS 人工验收。
