# LayerD 三分支重建

本目录对保存的 LayerD 图层包执行本地 OCR、字体拟合、视觉对象提取和基础几何识别。候选对象经过归属检查、实际 PPTX 渲染和区域质量门禁后提交。输出经过 `SlideIR 0.1` 和项目固定 Artifact Tool 编译器，`run`、`all`、`export` 使用同一条链路。

## 运行

在当前目录的 `auto` 入口显式传入 `--native-variant native_synthesis`，可先观察整页，再按模块规划，从空白背景绘制 PowerPoint。例子如下。

带图标的列表按行保留编辑位置。编号流程的每个步骤在选择窗格里可以按标题单选。连续段落允许换行，短标题先利用标题条的空余宽度。复杂回路线从原图中提取可验证的笔画轮廓，写成可编辑的 Office 路径。局部原图复读结果在 `icon-reading-decisions.json`。安装 PowerPoint 的 Windows 主机还会记录实际字框边界，并对溢出字段最多重编译一次，结果见 `office-text-fit.json`。视觉环境缺少 COM 库时，可以用 `IMAGE2PPT_OFFICE_PYTHON` 指向已安装 pywin32 的解释器。

```powershell
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py auto `
  --native-variant native_synthesis --inputs page.png --output runs/native
```

视觉规划沿用 `PLANNER_BASE_URL`、`PLANNER_MODEL` 和现有凭据配置。每页独立保存原始 OCR、整页观察、组件与文字归属、对象级图片清单、PPTX 和成品预览。核对已知招牌或徽章小字时查看 `object-report.json` 的 `embedded_text`。未确认转录保留在同一报告里。正文中的自然折行可以共用一个可编辑框。

现有 `--pack` 接受与本次输入像素一致的 LayerD 图层包。没有图层包时，难以分离的图标按对象请求 `--layerd-url`。可用 `IMAGE2PPT_NATIVE_SAM_URL` 和 `IMAGE2PPT_NATIVE_SAM_TOKEN_FILE` 配置分割服务，或用 `IMAGE2PPT_NATIVE_RMBG_HOST`、`IMAGE2PPT_NATIVE_RMBG_WORKER`、`IMAGE2PPT_NATIVE_RMBG_MODEL` 指定远端抠图环境。SSH 解释器和身份文件使用现有的 `IMAGE2PPT_SSH_WIN_PYTHON`、`IMAGE2PPT_SSH_IDENTITY`。透明度采用 0 到 1，形状纯色使用 `fill_opacity`，渐变按每个色标的 `opacity` 控制。

在项目根目录使用已有视觉服务 Python 环境，该环境包含 RapidOCR、ONNX Runtime、OpenCV、NumPy、Pillow、Pydantic 和 pytest。OCR 读取 `ppt-rebuild/models` 中的 PP-OCRv4 权重，禁用运行时下载。字体读取本机 Windows Fonts。

业务图层包、逐页复核 JSON 和生成产物采用本地保存方式。运行前准备自己的 `configs/reviews/NN.json`、`configs/text-repair-reviews/NN.json` 或 `configs/surface-reviews/NN.json`，并按各阶段命令传入源批次和复核目录。下文的页号表示本地样例，复核记录需要与实际源图哈希、尺寸和对象证据对应。

```powershell
# 全自动：截图 -> LayerD（默认 ssh://CMCC@192.168.1.103:18840）-> 本地测量 -> PPTX
# 令牌只从文件读取；BigLaMa 默认 ssh://CMCC@192.168.1.103:18841/inpaint。Cursor 不参与。
$env:PYTHONPATH=".runtime/core-installed;src;ppt-rebuild/src"
$env:IMAGE2PPT_SSH_HOST="CMCC@192.168.1.103"
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py auto `
  --inputs PPT图片/13.png --output ppt-rebuild/outputs/auto `
  --token-file .runtime/layerd/client-token

# 已有 LayerD 包时跳过远程拆层
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py auto `
  --inputs <source.png> --pack <layerd-pack-dir> --output ppt-rebuild/outputs/auto --inpaint-url none

# 读取全部图层包并核对来源、图层与复合结果
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py baseline

# 第 3 页，默认读取 configs/reviews/03.json
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py run 03

# 第 10、4 页回归，按给定顺序组合
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py batch --pages 10 04

# 12 页批处理
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py batch

# 从既有批次的完成记录继续。每次生成新的批次目录
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py batch --resume <batch.json或progress文件>

# 对既有交付补做图标组删除和原生候选文字替换，均直接编辑 PPTX 副本
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py check-edits <batch.json或delivery.json>

# 文字优先修复。读取既有批次、保存的源图复核和原生候选逐行样式
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py repair-text <batch.json> --pages 03 04 10 --inpaint-url ssh://<server>:18841/inpaint

# 检查实际合成 PPTX 的逐段样式，并直接删除、移动和替换诊断副本
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/src/text_repair_acceptance.py <修复批次batch.json>

# 使用新的工作目录执行回归测试
& services/vision/.venv/Scripts/python.exe -X utf8 -m pytest ppt-rebuild/tests -q --basetemp ppt-rebuild/workdir/tests-new
& ./image2ppt.ps1 test
```

`--pages-root` 放在子命令之前。`run` 接受 `--review`、`--ocr-cache`、`--output`、`--timeout`。`batch` 接受 `--reviews`、`--pages`、`--resume` 和 `--output`。`--skip-probes` 适用于调试，交付时执行直接编辑验证。旧的 `layers`、`ocr`、`fonts`、`mask`、`scene` 命令保留为模块诊断入口。

## 来源与人工决策

`configs/layerd-baseline.json` 将历史图层包的上传文件哈希、标准化 PNG 哈希和清单哈希绑定。该记录由用户提供的原始 ZIP 校验得到，12 页解码后的 RGB 像素逐页相同。历史服务的 `source_sha256` 指上传字节，服务重新编码后的 `source.png` 使用标准化哈希。新包可携带 `normalized_sha256` 或服务生成的 `request.json`。图层逐文件核对 SHA-256、尺寸、顺序和 straight alpha。

`configs/reviews/NN.json` 是当前会话保存的源图决策，绑定标准化原图哈希与尺寸。`objects` 保存已核对图标的源像素边界。`text_corrections` 按 OCR 行 ID 保存原文、修订文字、可选边界或 `retain_raster` 决策。`checked_lines` 表示进行了源图核对。图标和 Logo 保留原始视觉内容。复杂分隔符、混合样式及有疑问的源字形可以保留栅格。

LayerD 的 alpha 连通域仅生成建议文件。语义视觉对象通过保存的源图决策进入正式提取。脚本执行本地识别与确定性处理，模型分析记录来自当前会话。

## 提交与回退

每个源对象分配证据 ID。独立输出对象或源图底板拥有该证据一次。保护遮罩限制清理范围，注册表保存各 RGBA 图层的可见贡献、遮罩原点与尺寸、修复方式及操作日志。

新版本依次保存候选、工作遮罩、试渲染 IR 与 authoring 哈希记录、实际 PPTX 和区域指标。字体候选根据真实字形像素排序，再进行一次实际 PPTX 字形位置校正。区域 MAE 超过 10 时，重新从原始像素生成底板并撤销相应原生对象。背景修复与新增对象同步提交。

`delivery.json` 是完成标记。中断或编辑探针失败的目录保留试验材料，状态保持未提交。批处理允许保存已完成页和失败清单，全部页成功后组合成一份 PPTX，并逐像素核对组合页与单页预览。

## 验收记录

`registry.json` 和 `operations.json` 记录对象归属及回退原因。`delivery.json` 分别报告原生文字、原生形状、独立图片和栅格保留范围。`edit-probes/report.json` 保存直接编辑最终 PPTX 后的文件、回渲染、变化像素及图标原位置残差。

所有图标分别移动验证，代表性文字执行删除验证，代表性形状执行填充修改验证。已完成计算检查的产物仍保留 `REVIEW_REQUIRED`。PowerPoint、WPS 和人工验收分别记录。

当前样例的文字字体拟合可能触发严格的区域回退。原生文字数量为零时，报告应显示该结果，文字可编辑目标保持待完成。保留的试渲染版本可用于评估字体差异。

## 文字优先修复

`repair-text` 使用 `configs/text-repair-reviews/NN.json` 补齐栅格文字、拆分标签并定义语义段落。一个段落输出一个真实文本框，SlideIR 的可选 `paragraphs` 保留每行样式和局部强调。旧 IR 行为保留。

笔画掩码覆盖框外前景边缘，独立图标和形状提供保护区。复用 LayerD 底层前同时核对其背景颜色和残影。需要 LaMa 时传入完整上下文，仅写回授权掩码像素。HTTP 及 SSH 调用均使用已配置服务，SSH 路径通过服务器本机接口推理，无需端口转发。

修复失败的段落恢复源区域并归还背景证据，保留待处理原因。实际 PPTX 去字渲染再次失败时，该版本保留诊断并停止进入批次。存在待处理段落时命令返回 2。字体视觉误差独立记录，文字优先链路的清理通过判断使用实际去字背景。`text-inventory.json`、`paragraph-map.json`、逐行掩码、工作图层、修复回执、SlideIR、PPTX 和去字预览均按版本保存。

## 原生卡片和连续背景

从冻结的文字修复批次执行 `rebuild-surfaces`，读取 `configs/surface-reviews/NN.json` 中绑定源图哈希的容器、色带和嵌套区域。文字及图标独立保留，卡片背景组件组成真实分组。颜色拟合在源图干净像素上执行，使用纯色或两、三色标线性渐变。源图细纹理按已确认的干净背景策略处理。

```powershell
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py rebuild-surfaces <r3批次batch.json> --pages 03 04 10
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/rebuild.py rebuild-surfaces <r3批次batch.json> --resume <surfaces批次或progress.json>
& services/vision/.venv/Scripts/python.exe -X utf8 ppt-rebuild/src/surface_acceptance.py <surfaces批次batch.json>
```

源背景足迹在残余底图中完整移除。页面中性背景连续恢复，彩色装饰核心保持原像素，外围过渡限制在保存的授权区域中。实际PPTX的四边轮廓用于有限位置校正，文字和图标位置保留。独立去字渲染、背景低频MAE与95分位、轮廓位置、接缝和授权边界检查共同控制提交，同时保留原始像素误差。保留的异形装饰进入背景检查渲染，避免把临时隐藏装饰造成的断边计为修复接缝。

单一卡片的拟合、色差或轮廓检查失败时，恢复其最上层所属卡片及子背景的r3底图像素，其他卡片继续重建。回退清单保留原因，原生文字和已独立图标沿用基线。回退后仍执行独立文字及接缝检查，失败页保留诊断并隔离。后续页继续处理，全部通过后组合。恢复时核对源图、基线IR、复核文件、代码及产物哈希。

简单图标先经基础几何试算，再以最终PPTX的轮廓IoU、轮廓距离和区域MAE决定转换。原PNG始终保留，未通过的图标记录保留原因。复杂SVG匹配和原生路径能力按下一阶段清单单独实施。
