# LLM先规划分割（实验）

此入口直接向已有视觉模型发送原始截图，先产出模块、文字、图形及包含关系的规划，再把模型边界画回原图。它不先运行OCR、SAM或去字，也不生成PPTX；现有 layered 默认链路不变。

```powershell
python scripts/plan_segmentation.py run --input <source.png> --output <new-job-directory> --model-config <existing-client-config.json> --model deepseek-flash
```

每个新任务最多一次模型请求，错误也占用该次请求；目录已存在时拒绝覆盖。凭据继续留在既有服务器。原图、请求/响应、规划、模型用量及代码/配置摘要均绑定保存。导入脚本与协议测试不会调用模型。

产物：`modules.png` 为红色大模块边界；`objects.png` 为细分对象边界；`all.png` 为叠加；`PLAN.md` 列出编号、父模块、类型、后续处理建议与风险。绿为文字，紫为完整图形，橙为形状，青为装饰，蓝为表格，玫红为图表。边界是模型提出的归一化 xyxy 坐标，只按原图尺寸映射，不进行人工修边或规则吸附。

`SegmentationPlan/1` 最多12个模块、60个非模块对象、3层层级。验证器拒绝非法坐标、重复ID、父节点循环等；子对象超出父模块时明确警告，并保留模型原始坐标。

所有规划为 `REVIEW_REQUIRED`。矩形框不等于像素级分割掩膜，也不证明裁切后内容完整；重叠模块是逻辑容器，不应同时导出为重复裁图。先查看规划边界，再决定是否接入OCR/SAM/去字及对象重建。
