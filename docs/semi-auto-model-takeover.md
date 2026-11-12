# 半自动链路的模型接管试验

入口：`python scripts/model_slideir_loop.py`。基线是 `codex/semi-auto-restart`；固定协议、`scripts/main.mjs validate/compile` 和 Artifact Tool 回渲染不变。执行器只封存输入、调模型、限制局部修正、调用既有编译器并保存检查回执；页面结构由运行时视觉模型返回。

第一阶段输入一页原图、已有 OCR JSON、已有 PNG/JPEG 素材目录。执行器复制输入到新目录，记录哈希、尺寸和素材 ID，将原图及素材缩略图发给 OpenAI 兼容视觉模型。首次回复必须是完整 SlideIR 0.1 JSON，后两次回复只能修改已有节点的局部外观字段。每个有效候选都经过固定 `validate` 和 `compile`，后者从最终 PPTX 回渲染 `preview.png`。渲染图、原图、当前 IR 和整页 MAE 会送回模型；最多两轮修正。MAE 仅供比较，不是自动验收。`result.json` 以 `REVIEW_REQUIRED` 结束；PowerPoint/WPS 实际编辑、文字残影和人工审阅仍需分别检查。

录制模型回复的离线回放可用 `scripts/replay_model_response.py`，不会重新调用模型。若哈希匹配的 IR 声明了 `cleaned_background`，`scripts/model_text_layout.py` 会以原图与清字背景板的差值测量每个文字区，按实际源/目标画布比例换算 `size_pt`，并记录候选字体的几何和笔画密度。只有 OCR 证据与模型文字完全一致、两位序号后有足够宽的真实墨迹空隙时，才将合并节点拆成序号与标题；拆分保持字符拼接与证据归属。无清字背景板或拟合不确定时保留模型字段并记录原因。输出包括 `text-layout.json` 和 `text-comparisons/`（每图左侧原图，右侧回渲染）。模型修正回复中的局部 `font` 更新会合并到已有字体字段，不会丢失字体名称。

可运行 `scripts/compare_model_text_previews.py --source <source.png> --before <old-preview.png> --after <new-preview.png> --layout <text-layout.json> --output <new-directory>` 生成逐区前后对照；每图依次为原图、旧回渲染、新回渲染。像素差异只用于定位，不能替代实际 PowerPoint 编辑和人工视觉验收。

```powershell
# 只做输入封存和接口检查，绝不调用模型
python scripts/model_slideir_loop.py --source C:\path\page.png --ocr C:\path\ocr.json --assets-dir C:\path\assets --output .tmp-test\page01-offline

# 真模型：端点与模型名显式配置，密钥只从环境变量读取
$env:SLIDEIR_BASE_URL = 'https://provider.example/v1'
$env:SLIDEIR_MODEL = 'vision-model-id'
$env:SLIDEIR_API_KEY = '<set in the shell, do not commit>'
python scripts/model_slideir_loop.py --source C:\path\page.png --ocr C:\path\ocr.json --assets-dir C:\path\assets --output runs\page01-model --live
```

支持 `--base-url`、`--model`、`--api-key-env`、`--timeout`；对应默认环境变量为 `SLIDEIR_BASE_URL`、`SLIDEIR_MODEL`、`SLIDEIR_API_KEY`、`SLIDEIR_TIMEOUT`。每次新运行必须使用不存在的输出目录。模型请求前写 `reservation.json`；网络超时或响应未知时保留 `UNKNOWN_UNTIL_RECORDED`，不自动重发付费请求。密钥不进入请求记录、IR、日志或命令行。`model-response.json` 保留模型原始 JSON 内容供审核，因此输出目录应按业务素材处理。

已有 CC Switch OpenCode Go 凭据可通过 `--credential-source cc-switch-opencode-go` 在进程内只读加载；它不会被复制进输出目录。已用一张 64×64 的合成红图验证 `qwen3.8-max` 能通过 `https://opencode.ai/inference/go/openai/v1/chat/completions` 读取图像。真实业务页需要另行确认数据外发范围。OpenCode 请求带固定的产品 User-Agent 和同一运行内稳定的 `x-opencode-session`，调用用量存入各轮回执。不同 OpenCode Go 模型是否支持视觉输入应分别用合成图验证。

当前版本不让模型主动调用裁剪、分层或修图工具；素材由外部既有链路预先提供。第二阶段应在一个真实页的模型规划、编译、预览和人工审核跑通后，给模型增加受控的素材处理工具接口。整页栅格、重复文字、遗漏内容和未清除的旧字仍可能通过结构校验，不能用 `validate` 的成功代替视觉验收。
