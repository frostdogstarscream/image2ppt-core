# 单帧 SAM 3 / 3.1 实验适配器

这是独立实验适配器，不包含模型权重。2026-11-09 已在 RTX 4090、Python 3.12、PyTorch 2.8.0+cu128、Triton Windows 3.4 环境完成两种版本的受控单帧和真实 12 页困难对象运行；每种版本处理相同的 96 个困难对象。上游源码固定为 `f1647e8d0c1d280991f4d1b7d974c2704537fe66`。

```text
python single_frame.py --version sam3.1 --source source.png --plan objects.json --output new-output --sam-source /isolated/sam3 --code-revision <exact-commit> --checkpoint /models/sam3.1_multiplex.pt --checkpoint-sha256 <sha256>
```

两种版本使用同一 full-page 单帧 PNG、相同规范化 xywh 框与文本提示，逐对象 reset，不强行裁到定位框。保存所有原始候选而不是仅保存选中的 mask。已有 `load_sam` 校验源图、计划、版本、尺寸、权重及源码来源，再进行空间归属过滤。

先用受控图形验证单帧接口及坐标，再运行实际 12 页。空 mask、非有限值、模型维度不符、权重不匹配和显存不足均为失败；不自动退成矩形或另一模型。接口成功与语义分割完整性分开记录。

模型来源与授权由已有模型访问权限提供；程序默认离线，不获取或记录访问令牌。

权重可显式运行 `fetch_weights.py <directory> --only facebook/sam3` 或 `--only facebook/sam3.1`，从 ModelScope 获取冻结文件，分段续传并校验完整 SHA256。推理器本身仍为离线运行。`verify_environment.py` 只有在真实 CUDA 计算和 SAM 导入通过后才写就绪标记；`gpu_batch.py` 顺序运行冻结任务，依赖未就绪则等待。

该上游版本的 3.1 统一入口会给 `init_state` 传递它不支持的 `offload_state_to_cpu=False`。实验适配器按函数签名仅省略该关闭参数；若请求开启则明确失败。未修改上游文件，首次失败记录保留。这个桥接有专门回归测试。

两种版本均通过最终 owning model 的权重完整加载检查。真实图形质量仍需源图对照与人工复核，接口运行成功不能等同语义分割正确。本组交付选择 Alpha + SAM 3，后台新对象路线尚未默认启用。
