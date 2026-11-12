# 原图自动重建入口

`scripts/image_only_loop.py` 接受一张原图，调用 MoMA 的 `qwen/qwen3.8-max` 分析页面、选择工具、生成重建规划、检查实际产物并进行局部修正。页面运行输入只有原图。OCR 模型目录、API 端点和凭据属于运行环境配置。

本轮开发分支为 `codex/semi-auto-image-only-loop`，基线提交为 `61078de`。第 02 页既有产物作为冻结回归，新图使用 03、05、14 页。首次运行、修正候选和每次有效编译均保存在各自任务目录中。

## 启动

在具备 Pillow、requests、numpy、OpenCV、RapidOCR 的 Python 环境中运行。固定编译器继续使用项目已配置的 Artifact Tool 运行时。OCR 目录包含 `det.onnx`、`rec.onnx`、`cls.onnx`。

```powershell
$env:IMAGE2PPT_OCR_MODELS = 'D:/Code/image2PPT/ppt-rebuild/models'
python scripts/image_only_loop.py --source 'C:/path/page.png'
```

默认读取 `MOMA_API_KEY`。环境变量为空时通过隐藏输入临时读取密钥。端点默认为 `https://moma.cmecloud.cn/v1`，模型默认为 `qwen/qwen3.8-max`。密钥只用于当前进程的 Authorization 请求头。

可用 `--output <新目录>` 指定任务目录，用 `--base-url`、`--model`、`--models` 配置运行环境。默认生成新的 `runs/image-only-*` 目录。

```powershell
python scripts/image_only_loop.py --source 'C:/path/page.png' --output runs/page01
python scripts/image_only_loop.py --source 'C:/path/page.png' --output runs/page01 --resume
```

`result.json` 中的 `best_build.pptx` 和 `best_build.preview` 指向最后一个有效 PPTX 及其实际回渲染图。`best_ir` 指向该次有效编译的 IR。达到本次执行轮数时，可以通过 `--resume` 使用同一任务目录继续。默认每次执行最多 40 轮，连续六次工具失败时保留状态供诊断。

## 模型与工具

模型每轮返回一个 JSON 动作。执行器校验参数、运行工具、保存回执，再将实际图片和结果交给模型。

| 动作 | 实际行为 |
| --- | --- |
| `ocr` | 在封存原图上运行本地 RapidOCR，保存文字、位置及置信度 |
| `inventory` | 记录模型给出的可视对象清单、区域、处理方式及不确定原因，允许逐对象修订 |
| `view` | 查看原图、当前清字底板或最后有效 PPTX 回渲染的局部 |
| `draft` | 根据模型选择的 OCR ID、修正文字、样式与对象集合确定性展开 SlideIR |
| `patch` | 复用既有局部样式修正逻辑，另支持修正源字搜索范围 `source_text_bbox` |
| `fit` | 复用既有源字笔画测量与字体拟合，保留模型决定的文字和强调 |
| `extract` / `extract_many` | 按模型指定区域提取独立素材，支持平色透明背景及清字底板来源 |
| `objects` | 通过增删 ID 局部调整已选图片，也支持显式替换集合，保留当前文字及形状 |
| `remove_text` | 将模型判为不确定或误识别的文字返回栅格底板归属并记录原因 |
| `plate` | 复用源字像素掩码，补充实测源颜色和连通笔画，清除已提取对象，并执行逐文字源字测量与单行溢出收缩 |
| `compile` | 执行固定 `validate`、`compile`，从最终 PPTX 重新导入并渲染 |
| `finish` | 记录绑定原图、PPTX、预览哈希的模型复审及剩余问题 |

字体颜色与源字掩码证据独立处理，避免将目标字体颜色的估计误差带入清字范围。底板记录逐文字区域残影检查与掩码外变化量。含原生文字的面板可先清字，再从清字底板提取独立图片。素材保留其已清除文字的源范围，后续文本范围变化会触发重新检查。

`image_only_typography.py` 复用既有 `model_text_layout.calibrate`，按区域匹配源字尺寸。随后使用本机实际字体测量单行文本，对超出模型规划文本框的字号做局部收缩，保留文字、颜色和加粗分段。自动换行、多段文字及无法测量的字体保持模型字段，并记录为待复核。

编译前检查不同图片素材是否包含相同的源图笔画，发现照片裁剪携带子图标等重复内容时，将对象 ID、交叠范围和重复像素数反馈给模型。重新提取父对象前，应暂时取消该父对象的选择，保留子对象并重新生成底板，避免将已被修复抹除的区域当成原始素材。

素材与原生文字交叠时，执行器比较实际素材和原图的字迹像素。背景边缘能量检测作为独立诊断记录。对象列表的显式 ID 顺序控制图片从底到顶的叠放，父素材应置于其独立子对象下方。

对象数量、布局、配色、ID 与素材选择由本次模型分析产生。既有第 02 页专用背景提取器保留为独立可选能力，通用入口直接调用上述工具。

## 恢复与产物

任务目录保留原图副本、输入哈希、OCR、素材、逐轮模型原始回复、用量、耗时、工具结果、清字掩码及编译产物。`checkpoint.json` 原子保存状态与对话，长对话通过当前运行状态压缩上下文，原始回执继续保留。

任务目录使用进程级文件锁，防止同一目录被并发执行或恢复。模型回复的外层参数封装可做无歧义归一化，原始回复和实际执行动作分别保存。

模型请求发送前持久记录预约。收到 429 时按服务 `Retry-After` 退避。传输超时或未完整记录的响应保持未知状态。恢复时遇到未知响应会要求先核对该次请求，已完整记录的回复可直接重放本地工具。无效候选保留在独立轮次目录中，最后有效结果保持可用。

`REVIEW_REQUIRED` 作为交付状态保留。模型复审、程序检查、真实 PowerPoint 操作、WPS 和人工验收分别记录。独立参考对象清单缺失时，完整可编辑召回率保持未知。

对象清单中的栅格保留决定及尚未完成的素材单独记录。模型清单覆盖状态只描述本次规划内的对象，独立对象召回率继续保持未知。

## 验证

```powershell
python -m unittest discover -s scripts -p test_image_only_loop.py
./image2ppt.ps1 test
python scripts/verify_image_only_edit.py --run runs/page01 --output runs/page01-office-check
```

PowerPoint 检查在副本上修改文字、移动缩放对象、删除对象并保存重开，检查其他对象位置，同时导出去掉全部原生文字的实际页面供残影检查。原 PPTX 的哈希在检查前后核对。

本机已验证的生成环境为 `D:/Code/image2PPT/services/vision/.venv/Scripts/python.exe`。PowerPoint 检查使用安装了 pywin32 的 `D:/Develop/Anaconda3/python.exe`。

新图结果、模型调用汇总及最终检查记录随本轮交付更新。
