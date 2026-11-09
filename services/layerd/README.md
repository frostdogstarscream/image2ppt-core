# LayerD 独立服务

此服务封装官方 LayerD 的低层拆解 API，输出全尺寸 RGBA PNG 图层、重叠合成图、联系表和来源清单。文字仍是图层像素；不调用外部模型 API，不生成原生文本或 PPTX。

## 固定来源

- LayerD：`CyberAgentAILab/LayerD`，提交 `21aef937a0371614adb4d961f52d02409cb8ecc7`。
- BiRefNet：`cyberagent/layerd-birefnet`，提交 `679f743cd001fb5d6360e59e8e1904678c5fa734`。
- LaMa：`simple-lama-inpainting` 的 `v0.1.0/big-lama.pt`。
- 服务器：`36.134.45.38`，独立目录 `/data/image2ppt-layerd`。

源码归档、权重及其哈希在服务器独立目录保存，不加入仓库。运行时模型只读挂载，Hugging Face 离线模式，缓存位于容器临时目录。

Docker 构建需要预先在构建上下文放入上述固定版本的 `upstream/` 源码和实际来源记录 `source-provenance.json`；这两项是部署输入，不随本仓库分发。仅检出仓库不具备完整离线 Docker 构建材料。Windows 部署同样需要预备对应源码与权重。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | HTTP 存活、模型加载状态和 worker 存活 |
| GET | `/ready` | 模型与 worker 就绪返回 200，否则 503 |
| GET | `/v1/capabilities` | 配置与能力边界 |
| POST | `/v1/decompose` | multipart：`file`、`max_iterations`（1–5，默认 3） |
| GET | `/v1/jobs/{id}` | 任务状态与产物列表 |
| GET | `/v1/jobs/{id}/artifacts/{filename}` | 下载清单内产物 |

4090 独立部署可通过 `LAYERD_MAX_ITERATIONS=12` 提高请求上限（允许配置 3–12；未设置仍为 5）。请求默认值仍为 3。客户端先读取 `/v1/capabilities` 核对服务实际接受范围。增加轮数不强迫模型产生前景，不改变提前停止、连通域预算和单 worker 串行处理；应同时记录实际层数、合成误差、耗时及显存峰值。

业务接口使用 Bearer token。服务只发布到服务器的 `127.0.0.1:18840`。token 存在 `/data/image2ppt-layerd/client-token`，不写入命令参数、仓库或报告。

本地连接先开 SSH 隧道：

```powershell
ssh -N -L 18840:127.0.0.1:18840 36.134.45.38
```

已有 token 文件的客户端可以使用：

```text
python services/layerd/client.py --token-file <本地令牌文件> --input page.png --output <新目录>
```

也可在服务器执行发布目录内的 `client.py`，直接引用服务器令牌文件。

## 部署与停止

Compose 使用独立项目 `image2ppt-layerd`，只管理自身容器。部署前确认指定 GPU 没有运行任务；`LAYERD_GPU_UUID` 明确选择一张卡。默认 6 CPU、16 GiB 内存、单 worker 串行执行，PyTorch allocator 预算为可见显存的 70%；这不是显存硬隔离。

```text
docker compose --project-name image2ppt-layerd --file compose.yaml up -d
docker compose --project-name image2ppt-layerd --file compose.yaml stop
```

启动时将未完成旧任务标为 INTERRUPTED，保留所有原始上传、状态及产物，不自动重跑。模型初始化失败时保持不就绪。当前推理没有逐任务强制超时；进程失去响应时，客户端超时不会取消服务器任务，应检查后再决定是否重启本服务。

输入上限为 20 MiB、250 万像素，单帧 PNG/JPEG/WebP，不自动缩小源图。`max_iterations` 控制剥层轮数，不保证独立语义对象数量。固定源码实现返回背景、再由下到上的前景层；服务以实际代码返回顺序合成，未照抄存在相反描述的上游文档。合成误差不证明层归属或遮挡补全正确，真实页面仍需人工检查。

服务使用 `bounded_components_v2`：每轮先测量 alpha 连通域数量，并在实际连通域列表分配前检查所有嵌套颜色掩膜；超过 128 个或估算全画布连通域精修掩码超过 1 GiB 时，跳过该轮可选的前景/背景颜色精修，继续执行抠层、LaMa 补全和 unblending。嵌套颜色检查超出预算时复用 alpha，关闭精修后重试该轮。上游磁盘源码保持原样，服务启动时替换连通域辅助函数，编排使用固定提交的 `_decompose_step`，逐轮策略写入产物清单。此策略避免细密图案触发上游大量全尺寸掩码分配，但可能影响颜色与边缘；应结合原图复核。

## 本地接口检查

```text
python services/layerd/test_service.py
```

这些检查覆盖鉴权、就绪状态、输入限制、产物访问与图层合成顺序；真实 GPU 推理另外验收。

## 2026-11-04 部署验证

当前发布为 `/data/image2ppt-layerd/releases/20261104-r4`，镜像 `image2ppt-layerd:20261104-r4`，容器 `image2ppt-layerd-api-1`，使用物理 GPU 1（Tesla T4）。启动命令应在该发布目录执行，目录内 `.env` 固定 GPU UUID。此前发布、模型和任务数据全部保留。

真实页面均为 1672×941、最多 3 轮，经过上传、排队、GPU 推理、下载、图层哈希验证和人工查看预览：

| 页面 | 图层数（含背景） | 推理及产物生成时间 | PyTorch 峰值显存分配 | 重叠合成 MAE（0–255） |
|---|---:|---:|---:|---:|
| 封面 | 3 | 18.799 秒 | 3400.4 MiB | 0.95689 |
| 第 3 页 | 3 | 7.859 秒 | 3400.4 MiB | 1.84831 |

这是两页顺序验证，耗时不含启动加载与传输，不代表批量吞吐基准。最终容器在两次任务后 healthy、ready，重启次数 0；主机内存峰值约 1.55 GiB，未鉴权业务请求返回 401，首次发布的成功任务仍可查询。依赖 `pip check` 和 5 项本地检查通过。

封面首轮颜色区域检查检测到 7868 个连通域，触发关闭可选颜色精修后重试，避免上游大规模全画布掩膜分配。此前两次验证版本因该问题达到 16 GiB 上限并重启，任务保留为 INTERRUPTED；最终 r4 的两页验证均成功。该处理记录在 manifest 中。

图层能拆出背景、装饰和主要内容，但多段文字及多个图标仍聚在同一前景层。已查看图层和重叠合成图，尚未建立每个对象单独可编辑或整套 PPT 还原的验收结论。

服务器验证文件：`/data/image2ppt-layerd/evidence/deployment-r4-verification.json`；本地副本及页面产物：`outputs/layerd-deployment-20261104/`。
