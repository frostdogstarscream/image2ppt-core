# 0.3.0 服务器元素重建

本版本把已经存在的元素重建能力接入服务任务队列和 HTTP 产物协议，并修复文字测量结果的应用方式。服务输入为原图、服务内分析任务 ID 和声明式 ElementPlan；输出为 SlideIR、局部素材、原始提案和诊断。模型规划仍由调用方显式完成，PPTX 仍由宿主机固定 Artifact 编译器生成。

## 调用流程

1. 上传原图到 `POST /v1/assets`，保存返回的 `id` 和 `original_sha256`。
2. 提交分析：

```json
{
  "operation": "analyze",
  "asset_id": "<上传的32位ID>",
  "element_inventory": true
}
```

等到任务 `SUCCEEDED`，通过原有产物接口下载 `elements/inventory.json`、`elements/candidate-contact.jpg` 和原图。清单包含这次 OCR 的 ID、文字、坐标以及 OpenCV 图形候选。失败或部分完成的分析不能用于新入口。

3. 调用方根据这些证据生成 `ElementPlan/1` 或 `ElementPlan/2`。计划中的 `source_sha256` 使用原始上传文件哈希，`image_size` 使用清单尺寸，每个 OCR ID 必须恰好有一个处理决定。模型没有验收权。
4. 提交重建：

```json
{
  "operation": "reconstruct_elements",
  "asset_id": "<上传的32位ID>",
  "analysis_job_id": "<步骤2的任务ID>",
  "element_plan": {"schema": "ElementPlan/2", "...": "完整计划"},
  "renderer_version": "2.8.99"
}
```

上面 `element_plan` 是结构示意，不是可直接运行的完整计划。完整字段见 `supervised_elements.py` 和 `element_service.py`；未知字段、路径/代码字段、超限数据、跨源图计划、缺失 OCR 所有权和重复原图文字会拒绝入队。计划最大 1 MB，原生图形/栅格对象不超过 400 个，总对象不超过 500 个；单个局部栅格框最多占源图面积的 18%。原生文字字体族可用性仍需通过服务字体目录核验。

`renderer_version` 应由调用方从下游实际编译器读取。省略或不匹配已验证曲线时，不套用旧版本垂直补偿，也不声明位置已验证。当前附带曲线仅验证过特定字体字节、画布和 Artifact Tool 2.8.94；2.8.99 尚无等价曲线。

这次接口对带非默认 EXIF 方向的原图明确拒绝元素流程。先在调用方保存方向归一后的新图片，再从上传开始，不复用归一前的哈希。

## CLI

将 `{"element_inventory":true}` 保存到 `elements-options.json`：

```text
image2ppt-vision --url http://127.0.0.1:18761 analyze --input page.png --options elements-options.json
image2ppt-vision --url http://127.0.0.1:18761 reconstruct_elements --input page.png --analysis-job-id <分析任务ID> --plan element-plan.json --renderer-version 2.8.99 --output new-element-job
```

第二条命令可以重新上传同一字节的图片，服务按原图哈希核验关联。旧输出目录不会覆盖。省略 `--url` 可使用本地同一服务队列，此时分析任务必须位于同一 `--data` 目录。

HTTP 下载包中的主要产物：

| 路径 | 内容 |
|---|---|
| `reconstruction/submitted-plan.json` | 提交的方案，保留用于审计 |
| `reconstruction/proposal/` | 应用源图测量前的 SlideIR 和素材 |
| `reconstruction/calibrated/slideir.json` | 交给固定编译器的结果 |
| `reconstruction/calibrated/calibration.json` | 每个文字的前后参数、源图测量、拒绝原因、对比度诊断 |
| `reconstruction/quality.json` | 待复核项、渲染器版本与验收边界 |

本地 CLI 的 `--output` 直接复制 `reconstruction/` 的内容，`calibrated/slideir.json` 位于输出根目录下。

## 文字修正规则

- 模型提供了字体参数也必须接受源图测量；原参数保存在独立提案及 authoring 的 `original_model_typography` 中。
- 字体完整替换需满足已有像素测量条件，并额外要求字形重合度至少 0.58、尺寸误差不超过 0.12。不满足时保留提案并说明原因。
- 字体拟合中的搜索位移不再作为文本布局偏移。源图可见字形左上边界扣除字体字形 bearing 后作为原点；输出采用左对齐单行文字。是否与最终渲染器完全一致仍需要实际 PPT 检查。
- 搜索窗口内只排除完全处于原 OCR 框外的邻接连通块；穿越 OCR 框的可疑图形仍可触发拒绝。
- 对平滑背景上已隔离的单色文字，即使字体拟合失败，也可单独应用测得的文字颜色。字体、字号、位置仍保留原提案并列为待复核；多色、混合字重不强行改成单一风格。
- 文字与源图背景的低对比度作为风险报告，不能替代实际 PPT 的可见性检查。
- 新入口不自动改写图形填充，不抹平渐变。形状、图标、纹理及模型识别出的文字语义仍需复核。

`SUCCEEDED` 只表示任务和产物生成成功。`quality.status` 始终为 `REVIEW_REQUIRED`，`human_acceptance=false`，`actual_render_verified=false`。同一个对象可能有多个待复核原因，因此问题数不等于错误对象数。

## 验证

```text
python -m pytest tests -q
python tools/check_element_pipeline.py --input page.png --plan element-plan.json --output new-check --renderer-version 2.8.99
```

第二条会调用本机 ASGI HTTP 应用，使用真实本地 OCR 和字体资源，下载并核验全部重建产物。它不替换 OCR、不调用大模型、不手写 SlideIR。需要提前配置 `VISION_MODELS`，Linux 上还需配置已核验的 `VISION_FONT_MANIFEST`。这是本机服务集成检查，不是远程服务器验收；PPT 编译和实际预览检查单独执行。

参考 PPT 评测中的一项测试需要可选依赖 `image2ppt-vision[evaluation]` 的 `lxml`；缺少时仅跳过该测试，服务重建测试照常执行。完整开发验证应安装该可选依赖。

## 升级交付

源码包不含私有图片、模型权重、字体文件、API 凭据、CUDA 依赖或 Node/Artifact 运行时。已有服务器可复用原依赖镜像和已核验的模型/字体目录。新版本单独解压到 `/data/image2ppt-vision/releases/v0.3.0`，保留旧版本与持久化数据。

```text
docker build --network=none --build-arg PREVIOUS_IMAGE=image2ppt-vision:0.2.1 -f Dockerfile.upgrade -t image2ppt-vision:0.3.0 .
python3 tools/upgrade.py --release /data/image2ppt-vision/releases/v0.3.0 --previous-release /data/image2ppt-vision/releases/v0.2.1 --image image2ppt-vision:0.3.0 --expected-version 0.3.0 --font-manifest /fonts/manifest.json
```

执行前先核实服务器当前版本和目录，命令中的 0.2.1 是示例基线。`upgrade.py` 仅按已有升级规则切换本服务的 api/ingress，并保存升级证据；首次部署的 `deploy.py` 带业务容器停机副作用，不用于已有服务升级。

上述命令是部署说明，不表示已经执行远程构建或切换。Linux 镜像构建、现场模型/字体、远程 HTTP、最终 PPTX 的 PowerPoint/WPS 编辑以及用户视觉验收，均需在对应环境确认。
