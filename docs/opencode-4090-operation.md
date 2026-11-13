# Image2PPT OpenCode：4090 调用与运维

更新：2026-11-13。面向当前 Windows RTX 4090 部署；业务入口源码为 `services/product/product/app.py`，客户端为 `services/product/opencode_client.py`。

## 部署信息与状态

| 项目 | 当前记录 |
|---|---|
| 仓库 / 分支 | frostdogstarscream/image2ppt-core / codex/opencode-mcp-runtime |
| 上次部署功能代码 | dfc89fc520c21bd3e78193b309951ef8d04ad935 |
| SSH | cmcc@192.168.1.103 |
| 4090 应用 / Python | F:/i2oc/app / F:/i2oc/venv/Scripts/python.exe |
| 任务存储 | F:/i2oc/jobs |
| 业务 API | 4090 本机 http://127.0.0.1:8772 |
| OpenCode Server | 4090 本机 http://127.0.0.1:14100，固定版本 2.1.4 |
| Agent 模型 | OpenCode Go / qwen3.8-flash（最后实测配置） |
| MCP | OpenCode 为每个任务启动 Python stdio 服务，绑定 job_id |
| Office | 当前用户 Windows 电脑上的 PowerPoint Worker，认证后主动领取任务 |
| LayerD / LaMa | 4090 本机 18847 / 18841；LaMa 路径 /inpaint |

2026-11-13 文档编写时，192.168.1.103:22 返回 Connection refused；未能重新在线核验部署。以上部署参数来自上一轮实际部署收据，并非今日在线健康证明。不要恢复使用历史 IP 或把 18848 当作 LaMa /inpaint。

正常业务通过 8772 提交，不直接调用 OpenCode 14100。业务令牌、Office Worker 令牌、模型密钥是不同凭据。密钥由宿主注入，不写入仓库、命令历史或任务日志。

## 在 4090 本机提交任务

在 4090 PowerShell 中执行，前提是现有业务服务已启动，且当前会话由凭据管理方式注入 `IMAGE2PPT_PRODUCT_TOKEN`。不要把模型 API key 当作业务令牌。

```powershell
Set-Location F:/i2oc/app
$py = 'F:/i2oc/venv/Scripts/python.exe'
& $py services/product/opencode_client.py --url http://127.0.0.1:8772 submit F:/input/01.png --goal '将原图恢复为尽可能可编辑的 PPTX，保持文字内容、图标完整与布局。自主选择工具，完成 Office 检查和视觉自检；无法恢复的区域保留栅格并明确说明。' --tool-profile all --agent-steps 64 --seconds 3600 --no-agent-thinking
```

输入路径必须在执行客户端的机器上存在。多页按参数顺序传入：`submit F:/input/01.png F:/input/02.png`。返回 JSON 中的 `id` 是任务 ID，提交成功只代表入队。

```powershell
$job = '<提交返回的32位任务ID>'
& $py services/product/opencode_client.py --url http://127.0.0.1:8772 status $job
& $py services/product/opencode_client.py --url http://127.0.0.1:8772 artifacts $job --output F:/i2oc/downloads/job-result
# 仅需停止本任务时调用：
& $py services/product/opencode_client.py --url http://127.0.0.1:8772 cancel $job
```

下载目录应为本次任务的新目录；客户端验证 SHA256，遇到同名不同内容文件会拒绝覆盖。待验收任务可能没有正式产物列表，需保留服务器任务目录排查。

## 从当前电脑调用 4090

服务绑定远端 loopback，通过现有 SSH 配置建立本地转发。先确认没有已经运行的同端口隧道：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -i C:/Users/83895/.ssh/aitoy_tts_4090_ed25519 -L 18772:127.0.0.1:8772 cmcc@192.168.1.103
```

隧道终端保持运行，另开 PowerShell。当前产品工作树下已有 `.runtime/opencode-client.dpapi`，可用同一 Windows 用户的 Python 环境执行：

```powershell
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi submit D:/input/01.png --goal '恢复可编辑 PPTX，保留源内容，执行实际 Office 检查和视觉自检。' --seconds 3600 --no-agent-thinking
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi status <任务ID>
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi artifacts <任务ID> --output outputs/download-new-job
```

该 DPAPI 文件保存在本机、未入库，绑定创建它的 Windows 用户/环境；不要复制到 4090 后假设能解密。配置中的 URL 应对应本机隧道 `http://127.0.0.1:18772`。新客户端须单独安全配置业务令牌。

## HTTP 集成契约

所有以下请求使用 `Authorization: Bearer <业务令牌>`。

| 方法 / 路径 | 用途 |
|---|---|
| POST /api/v1/jobs | multipart/form-data，多张图使用重复的 files 字段，返回 202 与任务 ID |
| GET /api/v1/jobs/{id} | 生命周期、候选状态、已提交对象、验收状态 |
| GET /api/v1/jobs/{id}/events | SSE 事件流 |
| POST /api/v1/jobs/{id}/cancel | 请求取消，保留已有证据 |
| GET /api/v1/jobs/{id}/artifacts | 返回产物 URL 和哈希；使用返回 URL 下载 |

提交选项放在 query 参数中：`goal`、`orchestrator=opencode`、`tool_profile=all|basic`、`agent_steps`、`runtime_seconds`、`candidate_limit`、`model_calls`、`office_validations`、`agent_thinking`。此 /v1/jobs 入口默认转为 OpenCode 编排。

`all` 提供基础工具和高级快捷工具；`basic` 省略 run_perception、run_parallel_pipeline、rebuild_region、repair_region，用于验证 Agent 自主组合基础工具，不是日常推荐限制。

代码支持 1–32 张图；Agent 步数 1–128；运行时间 60–21600 秒；候选上限 1–12；模型调用预算 0–200；Office 预算 1–200。范围表示接口可接收，不代表这些规模已完成真实验收。当前最多允许两个排队/运行任务，超限需等待已有任务结束。算法调用预算与 OpenCode 自身模型回合不是同一个计数器；费用应查看实际调用轨迹。

## Office Worker

Worker 必须运行于具备 PowerPoint 和实际字体的 Windows 环境，使用专用 `IMAGE2PPT_OFFICE_WORKER_TOKEN`，不能使用业务令牌替代。当前电脑经 18772 隧道主动领取任务：

```powershell
# 在安装了项目依赖及 PowerPoint 的当前电脑，产品工作树下执行。
$env:PYTHONPATH = (Resolve-Path services/product).Path
python -m product.runtime.office_bridge --url http://127.0.0.1:18772 --root C:/Users/83895/AppData/Local/Temp/i2office
```

凭据由宿主安全注入到该进程；不要打印环境变量。已有 Worker 时不重复启动。保持交互式 Windows 登录会话可用。Office 操作受资源锁约束；Worker 失联、编辑异常或结果未知应保留 UNKNOWN，不伪造 PASS。

## 服务启动与恢复

已有部署正常运行时只调用 API，不重启。服务宿主使用 `product.runtime.windows_host` 创建独立计划任务，通过命名管道在内存中传入凭据；具体任务名称记录于 F:/i2oc/task.json。该机制不是可在重启后自动找回模型密钥的永久配置；不要直接重跑旧任务的临时命名管道参数。

维护人员在确认现有实例已停止、端口无冲突并已安全注入业务/Office 凭据后，可使用显式前台启动入口：

```powershell
Set-Location F:/i2oc/app
$env:PYTHONPATH = 'F:/i2oc/app/services/product'
$env:IMAGE2PPT_AGENT_PROVIDER = 'opencode_go'
$env:IMAGE2PPT_AGENT_MODEL = 'qwen3.8-flash'
$env:IMAGE2PPT_OFFICE_MODE = 'remote'
$env:IMAGE2PPT_OPENCODE_BIN = 'F:/i2oc/app/opencode.exe'
$env:IMAGE2PPT_NODE = 'F:/code/image2ppt/.runtime/bundle/node.exe'
& F:/i2oc/venv/Scripts/python.exe -m product.runtime.launch --root F:/i2oc/jobs --port 8772 --opencode-port 14100 --prompt-key
```

`--prompt-key` 隐藏输入模型密钥；业务令牌和 Office 令牌仍须预先安全注入。应用启动会检查固定 OpenCode 版本。不要同时启动前台和计划任务实例。GPU 算法服务沿用既有部署。

## 证据与故障定位

任务目录 F:/i2oc/jobs/{id} 下保留：request.json、inputs、runtime/state.json、runtime/runtime.sqlite、runtime/session.json、runtime/transcript.json、runtime/agent-usage.json、runtime/o 下逐操作证据，以及 result.json。正式交付位于 delivery。业务宿主日志为 F:/i2oc/business.log，OpenCode 日志位于 F:/i2oc/jobs/opencode.log。

SSH Connection refused：先核实主机 IP、SSH 服务和网络，不能据此断言 GPU 模型失效。401：核对业务令牌。任务不进入 Office：检查 Worker、隧道及专用令牌。BACKGROUND_REFERENCE_UNRELIABLE：保留候选、采样参考和诊断，不放宽擦除权限。幂等 UNKNOWN：先查原调用收据，不能盲目重复外部推理。

当前限制：最近实际测试产生两个原生文字探索候选，但正式提交被固定背景参考检查阻断，最终 0 个正式原生文字 / 0 个独立 PNG，状态 REVIEW_REQUIRED。Office 对纯栅格打开渲染通过、文字替换 NOT_APPLICABLE，不等于可编辑重建通过。OpenCode 路径无独立 Go 复核；同 Session 自检仍要求真实图像和 Office 证据。自动发布保持关闭。

本说明的客户端参数和接口已按当前代码核对；本次文档交付不包含新的模型推理或成功重建声明。详见 [实际验收记录](acceptance/opencode-identity-repair-20261112.md)。
