# 自动图片转 PPT：效果验证链路

本入口自动调用视觉模型完成规划、复检和限定范围修正。用户不需要打开 Codex
逐页操作，也不需要提交人工 ElementPlan。当前运行方式为**服务器模型与分析 +
本机固定编译与实际 PPTX 回渲染**。它依赖本机已有 Artifact Tool 运行环境，
不是独立 Linux 服务或 Docker 包。

## 组成与边界

- `scripts/auto_workflow.py`：显式启动入口，支持 `preflight`、`run`、`resume`、`serve`。
- `image2ppt_vision.auto_workflow`：任务协调器、阶段缓存、调用预算及代表页门槛。
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
默认先运行前三页（不足三页则运行全部）。代表页存在阻塞问题时，后续页保持未执行。
省略 `--preflight` 会为新任务重新验证模型；显式复用时校验连接配置和完整证据哈希。
`--preference fidelity_first` 将视觉保真偏好传给模型，但不放宽关键文字可编辑性检查。

每页默认串行执行，最多 8 次模型请求（包括重试）、2 轮视觉修正和 1800 秒总时限。
中断后可以恢复；预算包含中断经过的时间，不因重启而重置：

```powershell
python scripts/auto_workflow.py resume --job outputs/automatic/auto-0123456789ab
```

在修复运行环境或程序错误后，可用 `resume --job <目录> --retry-failed` 恢复无有效产物的
失败页。程序先保留旧结果到 `previous-results/`；只重试仍在原 1800 秒、8 次模型请求
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
  --config configs/auto-client.local.json --data outputs/automatic --port 18765
```

打开 `http://127.0.0.1:18765`。页面支持多图上传、调整页序、选择偏好、指定代表页、
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

## 方案、复检与产物契约

程序先依据 OCR 和源图像素测量生成文字候选默认值，保留测量诊断和渲染器版本。
规划模型通过 `ElementPlanDraft/2` 返回完整图形对象和稀疏 `text_overrides`，避免对
上百段文字重复输出未变字段；程序确定性展开为完整 `ElementPlan/3` 再校验。
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

机器复检输出 `AutomaticReview/1`，绑定实际 PPTX、回渲染图、对象清单、编辑探针与
方案版本。修正使用 `ElementPatch/1`，只能修改问题指向的对象与区域。未经指明的对象
冻结；重新裁剪必须有 crop 问题，修改文字内容必须有 text/numeric/missing 问题。
机器记录使用独立的自动策略来源，不生成或覆盖人工审批记录。

复检与修正的模型传输草案分别为 `AutomaticReviewDraft/1`、`ElementPatchDraft/1`，
也必须声明 `normalized_1000_xyxy`，程序转换后保存标准 `AutomaticReview/1` 和
`ElementPatch/1`。未经声明的坐标不会自动猜测转换。

输出包括每页原始输入、规范化图片、OCR、模型请求/答复、IR、构建记录、版本复检、
PPTX、预览、源图对照、差异图、实际文件删除探针以及 `delivery/report.json`。
业务图片和模型答复只保存在忽略的运行目录，不进入公共源码包。

执行状态与机器质量分开：有效文件不等于质量通过，质量通过也不等于人工接受。
有有效 PPTX 但没有合法且绑定当前版本的模型复检时，执行状态为 `PARTIAL`，
`pipeline_completed=false`；不能记为完整链路通过。
文字、数值、缺失、重复、不可编辑和探针残留均为阻塞项。确定性的导出错误不能被
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
python -m pytest services/vision/tests/test_auto_workflow.py services/vision/tests/test_element_service.py -q
.\image2ppt.ps1 test
```

测试中的模拟模型、模拟编译器仅验证协议与故障处理。真实效果应以本轮执行报告、
实际 PPTX 和回渲染图为准；不能由单元测试推断还原准确率。
