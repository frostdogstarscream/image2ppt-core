# image2ppt-planner

把「一张页面图 + 一份元素清单」变成**通过校验的 `ElementPlan/2`**。

这是整条重建链路里唯一需要模型的一步。模型不产出坐标：所有文字决定的 `id`/`bbox`/`text` 都必须原样来自 image2ppt-vision 产出的 OCR 清单，模型只决定**分类、分组、属性、z 序**。

```text
页面图 ──┐
         ├─▶ image2ppt-vision /v1/jobs (analyze) ──▶ inventory.json
         │                                               │
         └───────────────────────────────────────────────┴─▶ planner ──▶ ElementPlan/2
                                                                              │
                                                        image2ppt-vision (reconstruct_elements)
                                                                              │
                                                                        SlideIR ──▶ 编译器 ──▶ PPTX
```

## 为什么值得单独做一个服务

`validate_element_plan` 是这份契约的唯一权威。planner 每次拿到模型输出都会**先过这个校验器**，不通过就把校验错误原样回给模型重做，最多 3 次。这样"模型胡说"和"计划合法"在系统里是两件看得见的事。

## 契约里最容易踩的三个坑

| 坑 | 真实规则 | 后果 |
|---|---|---|
| 颜色 | **六位十六进制，不带 `#`**（`0B5FA5`，不是 `#0B5FA5`） | 写成 `#` 一律 `invalid color`，整份计划作废 |
| 切片面积 | 单个 `raster_asset` **≤ 整页 18%** | 满幅底图会被直接拒绝 |
| OCR 覆盖 | 每条 OCR id **恰好一次**，`bbox`/`text` 不得改动 | 漏一条或改一个字即失败 |

这三条都在 `tests/test_offline.py` 里有用例覆盖。

## 安装与运行

```bash
export PYTHONPATH=services/planner/src:services/vision/src

# 1) 先探活：验证凭据与校验器接线，不产生任何计划
python -m image2ppt_planner.app probe

# 2) 单页规划
python -m image2ppt_planner.app plan \
  --image page.png \
  --inventory inventory.json \
  --out-dir planner-runs/page-03

# 3) 起 HTTP 服务
python -m image2ppt_planner.app serve --host 127.0.0.1 --port 18810
```

### 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活 + 校验器是否可用 |
| GET | `/v1/capabilities` | 模型、字体清单、配置（**不含密钥**） |
| POST | `/v1/plan` | `{image_path, inventory_path \| inventory, out_dir?}` → `{plan, record}` |

## 配置（全部走环境变量，**不落盘**）

| 变量 | 默认 | 说明 |
|---|---|---|
| `PLANNER_API_KEY` / `MOMA_API_KEY` / `VISION_LLM_API_KEY` | 无 | 三选一，优先级从左到右 |
| `PLANNER_BASE_URL` / `MOMA_BASE_URL` | `https://moma.cmecloud.cn/v1` | OpenAI 兼容入口 |
| `PLANNER_MODEL` | `qwen/qwen3.8-max` | 必须是视觉模型；2026-11-04 实测该模型能读图并正确识别中文标题 |
| `PLANNER_MAX_ATTEMPTS` | `3` | 含首次；失败会带校验错误重试 |
| `PLANNER_MAX_TOKENS` | `16000` | 密集页计划可能很长 |
| `PLANNER_TIMEOUT` | `180` | 单次请求秒数 |
| `PLANNER_ARTIFACTS` | `planner-runs` | 产物根目录 |

密钥只从当前进程环境读取，**不写入任何 JSON、脚本或日志**；`/v1/capabilities` 与 `provenance.json` 只记录 `api_key_present`。

## 产物

每次规划生成一个目录：

```text
planner-runs/<ts>/
├── inventory.compact.json   # 实际发给模型的清单（可复现）
├── image.json               # 实际发送的图片哈希与编码方式
├── response-1.json          # 每次的原始响应，逐次保留
├── plan.json                # 通过校验的最终计划
└── provenance.json          # 尝试次数、模型、耗时、错误序列
```

## 测试

```bash
python tests/test_offline.py     # 或 pytest
```

离线、无网络、无凭据；用桩客户端驱动真实的 `validate_element_plan`，覆盖：正常通过、校验失败后返工、重试预算耗尽、图片重编码。

## 已知边界

- **不做分批**：一次请求送整页清单。目前页面 25–50 条文字，远低于 400/500 上限；如果遇到 150 条以上的密集页，需要按横向条带切分再合并（`image2ppt-vision` 里已有 `prepare_batched_element_jobs.py` 可参考）。
- **不保证字体正确**：`font_family` 只是提案，`resolve_font` 会在服务端按已验证字体目录再核一次，不通过就整份拒绝。
- **不写 PPTX**：编译是确定性环节，交给宿主上带 Node + Artifact Runtime 的编译器。
