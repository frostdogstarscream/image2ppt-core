# MoMA 独立图片转 PPT 服务

这条入口由 MoMA 看图、规划划分、生成及修正方案，由 OpenCode Go 对照原图和最终 PPTX 预览进行独立视觉复核。双方职责固定，复核失败不会换成规划模型自行验收。运行时不需要 Codex 会话，不读取先前人工整理的文字清单或字体分组。旧的 Codex 辅助流程保留。

## 启动

在仓库根目录运行：

```powershell
python services/product/launch.py
```

也可双击根目录 `image2ppt-product.cmd`，启动独立后台进程和浏览器。重新启动前应结束本服务自己的进程；不会自动终止同端口的其他程序。

启动器打开 `http://127.0.0.1:8765`，仅监听本机。浏览器携带本次服务的随机访问凭据，随即从地址栏移除。首次使用在网页中输入 MoMA 密钥；密钥仅存服务进程内存，关闭服务后失效。也可由启动环境注入 `MOMA_API_KEY`。地址默认 `https://moma.cmecloud.cn/v1`，模型默认 `qwen/qwen3.8-max`，可用 `MOMA_BASE_URL`、`MOMA_MODEL` 覆盖。现有 `PLANNER_*` 环境变量优先级更高。

每批上传 1–12 张相同尺寸的 PNG/JPEG/WebP，按文件选择顺序排列。单图最长边不超过 4000px、最多 1200 万像素、文件不超过 20MiB；整批最多 100MiB。同时最多接收两项未完成任务，串行执行以限制模型与编译器负载。

安装服务依赖见 `requirements.txt`。本机还必须已有项目固定 Artifact Tool 运行时、Node.js、项目编译器及 Windows 安装字体；这份入口没有将运行时重新分发，也不安装字库。运行 `./image2ppt.ps1 doctor` 检查编译环境。`VISION_NODE` 可以指定 Node 可执行文件。

## 实际执行步骤

1. 上传文件解码并保存为新的源图快照，不改原文件。
2. 枚举本机真实字体与字重，让模型输出 `VisualPage/1`：全文、位置、样式角色、强调、重复关系、形状与局部裁切。
3. 检查来源绑定、坐标、可用字体/字形、文字范围、同类样式及编号结构。01 独立而 02/03 与标题合并会被拒绝。原生文字与裁切图像不得重叠，避免重复文字残留。
4. 转换成现有 SlideIR，调用原有 `scripts/compose.mjs`；不执行模型代码，不修改公共 SlideIR 或 OOXML 适配器。
5. 检查最终 PPTX 的文字、中文/西文字体、字重、字号、颜色，并从最终文件重新导入渲染。
6. OpenCode Go 同时查看原图、最终预览、文字及图形放大对照，检查文字、结构、字体、强调、布局和图形。复核结果绑定原图、计划和预览哈希，包含具体修改意见和源图坐标。
7. 未通过时依据反馈修正相关页面，整批重新编译复核。按用户授权不设模型调用次数上限；仍有改善则继续，连续三次没有改善则停止并保留结果。没有隐藏传输重试。通过后提供 PPTX 和逐页预览。

OpenCode Go 可在网页中配置，也可设置 `OPENCODE_GO_API_KEY`、`OPENCODE_GO_MODEL`，或以 `--opencode-profile <profile.json>` 启动器参数读取明确指定的 Windows CurrentUser DPAPI 配置；不复制或改写原配置。默认接口为 `https://opencode.ai/inference/go/openai/v1`，模型为 `deepseek-v4.1-flash`。已通过合成图片实测确认该接口/模型能读取图片；模型列表也声明图片输入能力。MoMA 始终负责规划与修正，OpenCode Go 始终负责复核。每个请求记录实际模型、阶段和时间，复核调用失败不会降级为 MoMA 自审。

复核包含原图、最终 PPTX 渲染、最明显文字差异的放大对照及墨迹尺寸比。同组多个文本的宽高均小于原图 84% 时不能仅因模型口头通过便交付。该检查可能受背景干扰，结果保留用于诊断，不代替视觉复核。

`SUCCEEDED / AUTOMATED_VISUAL_PASS` 表示自动复核通过，始终保留 `human_acceptance=false`；不表示人工、PowerPoint 或 WPS 验收。视觉模型可能漏检；程序检查源自模型识别出的文字，不能单独证明与原图逐字相同。

`QUALITY_FAILED` 表示连续修正仍无改善，接口不发布成品下载。HTTP/网络/编译故障为 `FAILED`。服务重启把未完成任务设为 `INTERRUPTED`，不会自动再次发送收费请求。历史输入、候选、调用次数和输出保存在 `outputs/product-jobs`；这些文件可能含业务图片和文字，应按业务数据管理。

用户明确选择人工复核时，可使用 `generate_review_batch.py` 生成 12 页复核稿：MoMA 规划，固定编译器校验并生成，OpenCode Go 完成绑定到实际预览的复核后，即使仍有视觉意见也可导出 `REVIEW_REQUIRED / HUMAN_REVIEW_DRAFT`。这不绕过字体可用性、文字范围、对象结构或实际文件检查，也不将复核稿标为自动验收通过。`--resume` 只补跑未完成页；`--assemble-only --resume` 仅合并已经生成的页面，不调用模型。旧候选保留在各页独立目录中。

## 当前能力边界

自动入口支持原生文字、简单形状和局部图片；局部图片可移动缩放，内部像素不可编辑。此版本没有自动背景清理，也尚未接入原生复杂表格、图表的数据恢复、跨页模板联合规划或远端部署。遇到这些内容，模型必须报告限制，不能用整页图片冒充可编辑页面。

当前文字协议没有原生斜体属性；用户已明确允许忽略纯斜体差异。仅当 category=font 且 aspect=italic_only 时该项不阻止验收；字号、粗细、颜色、位置和图形倾斜不在豁免范围内。交付标记 italic_difference_ignored=true，不宣称恢复了斜体。

上传页面、任务队列和模型/编译器调度可独立运行。真实页面保真效果须以实际模型联调记录为准；单元测试和合成页面不代表任意图片都能成功恢复。

## 验证与诊断

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest services/product/tests -q
./image2ppt.ps1 test
python services/product/smoke.py outputs/product-smoke-new
python services/product/run_local.py --image C:/path/page.png --output outputs/product-run-new --prompt-key
```

输出目录必须是新的目录。命令行模式使用与网页任务相同的执行函数，隐藏输入的密钥不保存。页面和日志只报告经过约束的错误码，鉴权失败显示 `model_http_401`，不会回传上游正文或密钥。

API 使用 `Authorization: Bearer <启动器访问凭据>`：`POST /api/jobs` 上传图片，`GET /api/jobs/{id}` 查询，`GET /api/jobs/{id}/files/{name}` 下载经过哈希核验的交付文件。内存密钥配置为 `POST /api/provider-key`，请使用网页密码框，不要把密钥放入 URL 或命令参数。

新增影子对象路线可通过 `POST /api/jobs?object_extraction=auto` 选择，默认仍关闭。MoMA 规划完整对象，Alpha 与必要的私有 SAM 3 形成独立候选，经集合替换、实际 PowerPoint 编辑和 Go 复核后登记稳定 PNG。缺少证据时保留可靠父版本；`REVIEW_REQUIRED` 安全成品仍可下载，界面显示待人工复核。部署、缓存、预算与门禁说明见 [OBJECT_CLOSURE.md](OBJECT_CLOSURE.md)。

本服务默认用于单用户本机验证。对外上线前还需要独立的身份与权限、TLS、租户隔离、配额、数据保留策略以及经过授权的可部署编译运行时。
