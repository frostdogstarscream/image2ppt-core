# 36.134.45.38 环境核对

核对时间：2026-11-04 10:43 UTC+8
方式：`ssh 36.134.45.38`（`~/.ssh/config` 里已配 `root` + `id_ed25519_pengcheng_38`）
性质：**只读核对，没有启动、停止、修改、清理任何东西**

---

## 1 结论先说

机器本身状态很好：32 核 / 251 GB / 4×T4 全闲 / 磁盘充裕。

唯一的严重问题是：**你部署的 image2ppt-vision 服务，HTTP 端口还活着，但后台任务队列已经停摆 9 天。** 10 月 26 日提交的那个任务至今还是 `QUEUED`。Docker 里显示 "Up 2 weeks"、`/health` 返回 200，从外面完全看不出这个问题。

---

## 2 机器规格

| 项 | 值 |
|---|---|
| 主机名 | ecs-71253356 |
| 系统 | BigCloud Enterprise Linux V25（中国移动云定制版） |
| 内核 | 6.6.0-108.0.0.133.92.bc25.x86_64 |
| 已运行 | 89 天 |
| CPU | 32 vCPU，load average 3.70 / 3.84 / 3.89 |
| 内存 | 251 GB，已用 31 GB，可用 219 GB |
| 系统盘 `/` | 99 G，已用 54 G（58%），剩 41 G |
| 数据盘 `/data` | 1000 G，已用 182 G（19%），剩 818 G |
| inode | `/` 18%，`/data` 1%，都不紧张 |
| GPU | 4 × Tesla T4（15360 MiB），驱动 580.126.09，**0–3 号全闲（0–3 MiB，0% 利用率）** |
| Docker | 25.0.3，133 个容器 / 78 个在跑 |
| 宿主运行时 | python 3.11.6、git 2.43.0；**没有 node**；宿主没有 nginx（nginx 都在容器里） |

## 3 这是一台共用的机器

78 个容器在跑，`/opt` 下有 `cs-workbench`、`customer-service`、`peixian`、`secScanner`、`story-companion`；`/data` 下有 `dify-platform`、`ruoyi-17461`、`pq-sfas`、`pengcheng-flow`、`aitoy` 等。

对外监听（0.0.0.0）：22、5003、5194、17460–17463、19460、20460–20463。**80/443 都是空的**，没有对外 Web 入口。

含义：任何 `docker system prune`、端口占用、镜像清理都会影响到别人的服务，不能随手做。

---

## 4 image2ppt 的部署拓扑

```
/data/image2ppt-vision/                 ← 部署根目录（58 MB）
├── client-token                        ← 64 字节，API 用的 Bearer token
├── data/           (1.4 G)  ──挂载──▶  /data      （任务、资产、结果）
├── models-ready/   (192 M)  ──挂载──▶  /models    （PP-OCRv4 + SAM）
├── fonts-v0.2.0/   (9 个字体 + manifest) ──▶ /fonts
├── releases/       v0.1.0 · v0.1.0-final · v0.1.0-delivery · v0.2.0 · v0.2.0-delivery · v0.2.1-delivery
└── evidence/ · models/ · *.tar.gz（源码与字体包）
```

| 容器 | 镜像 | 状态 | 端口 |
|---|---|---|---|
| `image2ppt-vision-v1-api-1` | `image2ppt-vision:0.2.1` | Up 2 weeks（2026-10-16T04:36Z 起），`restart=unless-stopped` | 未对外发布 |
| `image2ppt-vision-v1-ingress-1` | `image2ppt-vision:0.2.1` | Up 2 weeks | **127.0.0.1:18761 → 18761** |
| `image2ppt-svc-20261020-api-1` | `image2ppt-vision:0.2.1` | Exited (143) | — |
| `image2ppt-svc-20261020-ingress-1` | `image2ppt-vision:0.2.1` | Exited (143) | — |

进程：`image2ppt-vision serve --host 0.0.0.0 --port 18761`（容器内 python 3.12.11），带一个 multiprocessing 的 worker 子进程。

容器环境（节选）：`VISION_DATA=/data`、`VISION_MODELS=/models`、`VISION_FONT_MANIFEST=/fonts/manifest.json`、`OMP_NUM_THREADS=4`、`SAM2_BUILD_CUDA=0`、`PIP_NO_INDEX=1`。

---

## 5 服务能力（`/health` 与 `/v1/capabilities`）

- `/health` → **200**
- `version: 0.2.1`（与 `releases/v0.2.1-delivery` 对得上）
- `geometry / ocr / segmentation / reconstruction` → **全 true**
- `reconstruction_requires_plan` → true（重建需要外部给计划）
- **`pptx_compilation` → false**（本机不产 PPTX，编译要在有 Node + Artifact 运行时的宿主做；而这台没有 node）
- `network_inference` → **false**（服务自身不联网推理）
- 模型：PP-OCRv4 Chinese（`det.onnx` 4.7 MB、`cls.onnx` 585 KB 等）+ SAM（commit `2b90b9f5`），manifest 齐全
- 字体：9 个文件 + `manifest.json`，挂载正常

---

## 6 需要你拍板的三件事

### ① 队列停摆 9 天（唯一严重项）

| 证据 | 值 |
|---|---|
| 任务总数 | 54：**52 SUCCEEDED / 1 INTERRUPTED / 1 QUEUED** |
| 最后一个成功的任务 | 2026-10-16 12:40（本地） |
| 日志里的异常 | 2026-10-16T23:24:14Z 一条 traceback，**只出现 1 次** |
| 异常内容 | `worker.py:28 loop` → `storage.py:132 next_job` → `storage.py:46 db` → `PRAGMA journal_mode=WAL` → `sqlite3.OperationalError: unable to open database file` |
| 卡住的任务 | `557a7068779d4ee281bfa1ff3518d778`，创建于 2026-10-26T02:58Z，状态 `QUEUED`，**至今 9 天** |
| 现在 | 2026-11-04 10:43 |

读法：worker 线程在那条异常后就没再取过任务，而 HTTP 服务本身活着，所以 `docker ps` 显示 "Up"、健康检查也是 200，监控完全看不出来。这解释了为什么"服务看着在跑，实际上不出活儿"。

我没有重启它（按你的要求）。要不要查根因、还是先重启容器把那个任务取出来，等你定。

### ② 一张 T4 分给了它，但没用上

容器确实被分配了一张 T4（`GPU-452202ca-1679…`），但设备上只有 0–3 MiB 占用、0% 利用率——因为镜像里 `SAM2_BUILD_CUDA=0`、模型是 ONNX、`OMP_NUM_THREADS=4`，走的是 CPU 推理。**4 张 T4 实际上是全闲置的。**

### ③ 系统盘偏紧，但不能乱清

`/` 只剩 41 G，而 Docker 镜像总共 78.11 GB（其中 68.88 GB 标记为可回收）、构建缓存 1.7 GB。但同一台机器上有 78 个别人的容器在跑，**不能执行 `docker system prune`**，只能逐个镜像处理。

### 附：怎么从本机调用它

它只监听 `127.0.0.1:18761`，本机要连得先开隧道：

```powershell
ssh -N -L 18761:127.0.0.1:18761 root@36.134.45.38
```

然后本机 `http://127.0.0.1:18761` 就通了，token 从 `/data/image2ppt-vision/client-token` 读（不要写进脚本或聊天记录）。

---

## 7 这次没有做的事

- 没有重启、停止、启动任何容器或服务
- 没有提交任务（只对已有任务做了 GET 查询）
- 没有清理镜像、容器、缓存、日志
- 没有写入任何文件（没有创建目录、没有改动 token、没有触碰 sqlite）
- 没有修改防火墙、端口或系统配置

所有操作都是 `ls` / `cat` / `docker inspect` / `docker logs` / `curl GET` 这一类只读动作。
