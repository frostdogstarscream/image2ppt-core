# 自动图片转 PPT：区域决策闭环 v2

本入口自动调用视觉模型完成规划、复检和限定范围修正。用户不需要打开 Codex
逐页操作，也不需要提交人工 ElementPlan。当前运行方式为**服务器模型与分析 +
本机固定编译与实际 PPTX 回渲染**。它依赖本机已有 Artifact Tool 运行环境，
不是独立 Linux 服务或 Docker 包。

## 组成与边界

- `scripts/auto_workflow.py`：显式启动入口，支持 `preflight`、`run`、`resume`、`serve`。
- `image2ppt_vision.auto_v2`：v2 协调器；复用 `auto_workflow` 的传输、阶段缓存和预算基础设施，保留旧版实现。
- v2 顺序执行：OCR/字体候选 → 模型区域划分 → 既有决策 Agent → 按决策生成 → 实际文件编译 → 分离复检 → 限定修补。
- 在独立工作树、配置、数据目录和端口运行。拒绝恢复 `AutomaticJob/1`；旧任务由原工作树继续处理。
- `services/vision/tools/auto_remote.py`：通过 SSH 调用的服务器程序，没有监听端口。
  在服务器内通过 Dify 已有租户密钥机制读取指定凭据并调用对应 MoMA 端点。
  密钥不写入请求、结果、日志或本机配置；不修改 Dify 默认模型、凭据或应用。
- 分析可以复用原 HTTP 服务，也可使用 `isolated_cli`：在已安装 OCR 的容器里执行
  原 CLI，显式传入新的 `--data`，使用独立数据库和工作进程。该模式不依赖生产队列，
  不重启容器，不覆盖已有服务代码、模型、字体或数据。
- 本机使用共享 `ArtifactCompiler` / `Runner` 导出、回渲染、检查 OOXML，
  并编辑诊断副本验证文字删除。原始 PPTX 始终保留。

## 安装与连接

本机需要 Python 3.12、SSH 客户端，以及项目原有的 Node / Artifact Tool / 字体环境。
使用单独虚拟环境，不需要在本机安装 OCR 或 GPU 推理运行库：

```powershell
python -m venv .venv-auto
.\.venv-auto\Scripts\python.exe -m pip install -r requirements-auto.lock
.\image2ppt.ps1 doctor
```

连接配置与服务器配置模板分别是 `configs/auto-client.example.json` 和
`configs/auto-remote.example.json`。二者都不包含 API key。部署时填写真实主机、
容器名和路径，使用新的远程目录；`isolated_host_root` 必须位于该 OCR 容器已有
数据挂载内，并与 `isolated_container_root` 指向同一目录。

`credential_provider` 和 `credential_model` 选择 Dify 中保存的现有凭据行。
必须恰好匹配一条记录，端点必须匹配 `base_url`；存在歧义时直接失败。
`allowed_models` 是显式允许调用的模型清单，不修改 Dify 中该凭据绑定的默认模型。
服务器脚本兼容 Dify 工厂函数返回应用对象或包含应用的元组。

可以手动把 `auto_remote.py` 和填好的配置复制到新的私有目录，或使用：

```powershell
python scripts/install_auto_remote.py --ssh-host user@server `
  --remote-root /data/workspaces/image2ppt-auto-new `
  --server-config configs/auto-remote.local.json `
  --client-output configs/auto-client.local.json
```

安装器只创建新目录，不覆盖已有目录。首次 SSH 主机信任和登录配置由操作者完成；
运行中使用严格主机校验，不自动接受未知主机。

## 模型验证与批量入口

先用随机字符、形状和颜色的小图验证账号权限、图片输入与 JSON 输出。目录可见
不等于模型可调用。首选通过即使用首选；首选失败才测试第二候选，并记录原因。

```powershell
python scripts/auto_workflow.py preflight `
  --config configs/auto-client.local.json --output outputs/preflight-new

$images = Get-ChildItem 'D:\slides' -File -Filter '*.png' | Sort-Object Name
python scripts/auto_workflow.py run `
  --config configs/auto-client.local.json --inputs $images.FullName `
  --output outputs/automatic --gate-pages 5 8 12 --preflight outputs/preflight-new
```

输入顺序就是最终页序。`--gate-pages` 为从 1 开始的页码，按给定顺序先运行。
默认仅运行前三页（不足三页则运行全部），或指定的代表页。通过也不自动扩展；只有显式传入 `--expand-after-gate` 才会在所有代表页通过后处理其余页面。
省略 `--preflight` 会为新任务重新验证模型；显式复用时校验连接配置和完整证据哈希。
保留 `--preference` 参数作为任务偏好记录；v2 的实际原生／图片范围由绑定的区域决策确定，两种偏好均执行同一套保真和编辑目标门槛。

每页串行执行，最多 **48 次模型请求、90 分钟、每区域 2 轮修补**。初次划分、决策和生成合计最多 32 次，预留至少 16 次给复检与修补。失败、格式重试和重新决策都计入；模型预检单独记录。
中断后可以恢复；预算包含中断经过的时间，不因重启而重置：

```powershell
python scripts/auto_workflow.py resume --job outputs/automatic/auto-0123456789ab
```

在修复运行环境或程序错误后，可用 `resume --job <目录> --retry-failed` 恢复无有效产物的
失败页。程序先保留旧结果到 `previous-results/`；只重试仍在原 5400 秒、48 次模型请求
预算内的页面，已有效的方案和构建继续复用，预算不因恢复而增加。

已经校验通过的分析、模型答复和构建产物不会重复执行。请求发送前保存 request_id；
服务器对已完成请求返回缓存结果。如果请求执行结果不确定，记录该状态，不悄悄
重发同一推理。终态失败任务保留证据；改变连接配置后应创建新任务，不能修改原任务绑定。

SSH 传输失败时最多按原 request_id 取回一次服务器缓存，原传输失败与取回结果分别
保存并绑定哈希；这不是新模型请求。服务端仍在执行或结果不可确定时不会重复推理。
模型校验反馈作为独立后续消息发送，保留原答复和具体源图约束，避免与原始材料混杂。

## 本机验证页面与 API

```powershell
python scripts/auto_workflow.py serve `
  --config configs/auto-client.local.json --data outputs/automatic-v2 --port 18766
```

打开 `http://127.0.0.1:18766`。页面支持多图上传、调整页序、选择偏好、指定代表页、
查看进度并下载产物。服务只监听回环地址，使用本次启动生成的本机会话令牌，
拒绝跨来源请求。页面显示模型质量与人工验收的区别，不要求用户编辑方案。

| 接口 | 行为 |
|---|---|
| `POST /v1/auto-jobs` | multipart 图片列表 `files`，可选 `preference`、逗号分隔 `gate_pages`；返回 202 与 job_id |
| `GET /v1/auto-jobs` | 本机任务列表 |
| `GET /v1/auto-jobs/{id}` | 任务状态、当前阶段、每页结果和产物清单 |
| `GET /v1/auto-jobs/{id}/artifacts/{name}` | 下载交付目录内的产物 |

这些接口需要 `X-Local-Token`；页面自动附加。生产 vision API 的既有接口不变。
不要把本机验证页面直接暴露到公网。

若同一数据目录中的任务已由命令行运行，启动查看页面时加 `--no-resume-pending`，避免服务再接管待恢复任务；新上传任务仍由页面正常执行。只有新配置、新数据目录和 18766 属于 v2，原 18765 服务不需要停止。

## 方案、复检与产物契约

布局模型先输出 `RegionLayoutDraft/1`，包含语义区域、子区域和位置；程序检查 OCR 唯一归属、重叠及整页回退。检测候选只作建议。

逐区域复用 `region_plan.py` 的提示词、特征和决策校验。保存 `AutomaticRegionDecision/1`，绑定源图、布局、区域裁剪、能力快照及模型答复。来源是模型自动决策，`human_acceptance=false`，不生成或改写人工批准。

- `native`：生成允许的原生部分。
- `hybrid`：原生部分与必须保留的源图部分分别映射到真实对象。
- `retain_module`：依据有效决策确定性裁剪整个区域，并披露保留文字。

`RegionElementsDraft/1` 提供对象、稀疏文字覆盖、逐项文字校准记录及决策部分到对象的映射；展开为区域元素，再组装 `ElementPlan/3`。所有对象必须有归属，必须保留部分必须映射到真实图片；禁止以圆点替代保留图标、重复文字、整页图片和全图片切片拼页。仅经过验证的区域范围可超出旧版 18% 裁剪限制；旧协议默认限制不变。

文字候选同时提供原始框、`fit_applied`、`font_confidence`。模型显式确认、校准或标记未解决，保留图片中的文字另行声明。低置信度候选不能靠默认值自动通过。
草案必须声明 `coordinate_space=normalized_1000_xyxy`：两个轴均为 0..1000，框为
左、上、右、下。程序只依据这个明确声明转换为源图像素 xywh，不猜测模型坐标。
提供给规划模型的文字候选框使用同一坐标体系，字体大小仍为源图像素 em。
原始模型答复、展开方案及二者哈希分别保存。文字默认值只是候选，仍需模型复检，
不代表字体已准确识别或页面质量通过。

校验失败时，自动反馈会给出不支持的几何类型和源图裁剪框实际包含／跨越的 OCR ID。
这些诊断由源图坐标计算得到，只用于下一次模型请求，不修改模型答复、裁剪框或方案。
图片文字归属、重复文字和局部保留范围仍须通过同一套严格验证。

图片对象必须分别声明 `source_crop_bbox` 与 `dest_bbox`，
原生形状和文本继续使用 `bbox`。HTTP 入口同时支持旧 /1、/2 方案，拒绝混合坐标。
源图哈希、尺寸、OCR ID、文字归属和字体字段必须验证后才能进入编译。

复检分为三条证据链：程序检查实际原生对象、文字内容与删除结果；视觉模型只比较固定源区域和正常回渲染；残影模型只比较正常图和指定对象删除后的图。删除图绝不进入源图保真复检。另有整页正常预览复检，检查区域间缺失与整体关系。

字体宽度溢出、对比度和同组字号检查是估计诊断，仍需模型查看实际渲染。已批准保留的图片文字不作为原生失败；原生目标仍受实际对象与编辑探针约束。

修补返回完整区域草案，但只允许修改复检指向的对象。比较实际计划差异，拒绝仅改描述或无变化的补丁；重新裁剪必须有 crop 问题，改字必须有 text/numeric/missing 问题。改变处理方式必须重新调用决策 Agent 并生成新决策。冻结区域只有在元素、决策和实际渲染哈希都相同时才能继承复检。

每轮保存已解决、未解决和新增问题；元素变化但渲染无变化不能记为有效修复。版本选择先看已验证问题和区域结果，像素差只用于同等条件比较。源图、答复、对象映射、决策和构建版本均以哈希绑定。

初次生成若在有限格式重试后仍因原生几何、原生／图片映射或必需部分冲突受阻，可调用决策 Agent 重新决策一次，再生成该区域。新旧范围和失败响应分别保留；只有处理方式或原生／保留部分确实改变才进入重新生成，单改置信度与理由不算。所有调用仍属于初始 32 次上限，不借用复检预留额度。

优化提示词后恢复任务时，已通过校验的区域可复用：必须从保存的请求中找回旧提示词，且除提示词外的源图、模型、裁剪及全部父级输入精确一致，输出哈希不变，并再次通过当前协议校验。另存提示词复用记录；其他证据错配仍拒绝。

输出包括每页原始输入、规范化图片、OCR、模型请求/答复、IR、构建记录、版本复检、
PPTX、预览、源图对照、差异图、实际文件删除探针以及 `delivery/report.json`。
业务图片和模型答复只保存在忽略的运行目录，不进入公共源码包。

执行状态与机器质量分开：有效文件不等于质量通过，质量通过也不等于人工接受。
有有效 PPTX 但没有合法且绑定当前版本的模型复检时，执行状态为 `PARTIAL`，
`pipeline_completed=false`；不能记为完整链路通过。
文字、数值、缺失、重复、要求原生的目标不可编辑和探针残留均为阻塞项。确定性的导出错误不能被
模型忽略。模型质量通过且无结构性阻塞项，代表页才允许扩展。像素差异仅辅助比较。
报告始终保留 `human_acceptance=false` 和 `office_acceptance=false`，直到独立验收。

对已完成的任务可运行 `python scripts/check_automatic_delivery.py --job <目录>`，
新增 `VERIFIED-RESULTS.md` 和 `verified-summary.json`，独立核验实际文件、哈希及
删除探针，不重新调用模型，不改写原模型方案或复检意见。

当前自动元素协议生成原生文字、形状和独立图片。形状组成的表格/条形图不是原生
表格/工作簿图表；复杂图像保留的文字如实披露，不冒充可编辑。

## 验证

```powershell
$env:PYTHONPATH="$PWD\services\vision\src;$PWD\src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
New-Item -ItemType Directory -Path .build -Force | Out-Null
python -m pytest services/vision/tests/test_auto_regions.py services/vision/tests/test_auto_workflow.py services/vision/tests/test_element_service.py -q --basetemp .build/pytest-auto-new
.\image2ppt.ps1 test
```

测试中的模拟模型、模拟编译器仅验证协议与故障处理。真实效果应以本轮执行报告、
实际 PPTX 和回渲染图为准；不能由单元测试推断还原准确率。
