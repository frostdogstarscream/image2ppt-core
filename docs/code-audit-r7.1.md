# r7.1 半自动链路 · 代码审计

审计对象：`codex/semi-automatic-workflow` @ `8ea222d`（工作树 `D:\Code\image2PPT-wt-semi`）
规模：`ppt-rebuild/src` 34 个模块 471 KB，`tests` 18 个文件 171 KB，合计 10,661 行
结论：**门禁与事务设计扎实，可复现性强；但代码指纹有盲区、崩溃窗口无自动恢复、四代管线并存。**

---

## 1 系统实际分四层

```
① 模型层（有网络）    MoMA qwen/qwen3.8-max 逐页字体与排版规划      typography_pipeline.py
② 证据层（本机 CV）   LayerD 分层 / PP-OCRv4 / OpenCV / Big-LaMa      layerd_loader, ocr_engine, text_repair
③ 仲裁层（确定性）    对象注册表 + 关系校验 + 事务 + 六道门禁          object_resolver, relation_validator, reconstruction_transaction
④ 落地层（固定）      SlideIR 适配 → Artifact Tool 固定编译器 → 真机 Office 验收   slideir_adapter, compiler_bridge.mjs
```

关键事实：**r7 主线本身不调用任何模型**（`r7_pipeline.py` 无 `httpx/moma/openai` 命中，`environment()` 里写死 `model_calls: 0`）。模型只在前段做字体/排版规划。所以"半自动"= 模型负责猜想 + 确定性代码负责验证与落地。

## 2 命令 → 模块路由

| 命令 | 落到 | 世代 |
|---|---|---|
| `layers / ocr / fonts / mask / scene / eval` | `layerd_loader / ocr_engine / font_matcher / text_mask / scene_builder / visual_eval` | P0（M1–M6） |
| `run / all / export` | `rebuild_pipeline.run_page` | 三分支（text/asset/surface） |
| `batch` | `rebuild_pipeline.batch` | 同上 |
| `repair-text` | `text_repair.repair_batch`（默认 LaMa `:18842`） | 擦字 |
| `rebuild-surfaces` | `surface_rebuild.rebuild_batch` | 原生卡片底 |
| `check-edits` | `rebuild_pipeline.verify_deliveries` | 直接改最终 PPTX |
| **`reconstruct-r7`** | **`r7_pipeline.run`** | **当前主线** |
| **`qa-r7`** | **`r7_pipeline.qa_run`** | 同上 |

四代管线并存于一个目录，`rebuild_pipeline.py`（三分支）与 `r7_pipeline.py` 职责部分重叠。

## 3 r7 主线的执行流程

`run()`（`r7_pipeline.py:496`）：

1. `RunLock(directory)` 独占锁；`environment()` 生成运行指纹写 `run-manifest.json`
2. 指纹不一致 → **拒绝复用**，要求新目录
3. `shadow` 模式：逐页只产注册表与冲突统计，不构造候选
4. `gated` 模式：按**固定业务页序** `01,02,08,03,04,10,11` 再补其余页逐页 `process_page`
5. 全部完成后再次校验环境未变 → `compiler_bridge.mjs compose` 合成 12 页
6. `audit()` 整稿 QA → `package()` 出交付包（含 `delivery.json`、`hard-failures.json`、`qa.json`、逐页对照 HTML）

`process_page()`（`:278`）：

1. 命中 `result.json` 则校验 scene/pptx/依赖哈希后直接复用（幂等检查点）
2. 新建 `execution-<stamp>` 目录，**不复用旧尝试目录**
3. `from_ir()` 产只读影子注册表 → `prepare_spec()` 冻结关系规格（源掩码 + 审核区域，绑源图哈希）
4. 克隆基线 IR → 编译 → `check_relations()` 得到每个元素的 PASS/FAIL/REVIEW_REQUIRED/UNKNOWN/NOT_APPLICABLE
5. 决策：PASS→NO_OP；FAIL 且 `repairable`→REPAIR；否则 REVIEW_REQUIRED（→ 整页 `BASELINE_RETAINED`）
6. 第 1 页走专用 `attempt_cover`（只允许封面容器 y/h 变化）
7. 有修复时最多 **2 次候选**：`build_candidate` → 事务 `prepare` → 编译试版 → 六道门禁 → 全 PASS 才 `commit`
8. 门禁未过 → `rollback`（终态），把失败的 relation 转 raster 再试一次；仍失败则保留基线

## 4 六道门禁的实际判据

| 门禁 | 判据 | 位置 |
|---|---|---|
| `relations` | 每个声明元素独立 PASS，聚合仍按 FAIL > REVIEW > PASS | `relation_validator.check_relations` |
| `frozen` | 除授权对象外，IR 结构逐字段冻结 | `r7_pipeline.frozen_scene:57` |
| `write_boundary` | 渲染像素变化必须落在授权掩码内（掩码外扩 5px） | `check_authorized_changes` |
| `text_clean` | **冻结的源字形掩码残差 ≤1%** + 独立 OCR 在文字区零检出 + 掩码清单集合相等 | `clean_check:136` |
| `raster_fit` | 回退切片 `mae ≤ 2` 且 `p95 ≤ 5` | `:341` |
| `render_integrity` | 试版预览哈希入账 | `:344` |

外加 `ownership`（`validate_ir_owners`）单独校验证据归属。事务 `verify()` 任一门禁非 PASS 即 `NEEDS_REVIEW`，`commit()` 再次拒绝 —— 双保险。

## 5 强项（值得保留）

1. **fail-closed 状态代数**：`qa_status.normalize_check` 把 `UNKNOWN` 降级为 REVIEW_REQUIRED、非法/缺失状态升为 FAIL、`NOT_APPLICABLE` 无理由也升 FAIL。很难靠"没跑到"蒙混过关。
2. **事务不可变**：`prepare` 把基线与被选场景快照进 attempt 目录并记录指纹；`commit` 校验 acceptance 回执身份、证据哈希、父指针；`rollback` 是终态，已提交不可回滚。
3. **环境指纹覆盖广**：基线证据、关系清单、字体清单与每个字体文件、逐页 IR/PPTX/源图/冻结文字掩码、运行时、机器名、Python 版本全部入账。
4. **降级而非造假**：修不好就 `BASELINE_RETAINED` 或改 raster，从不把失败的候选写成成功。

## 6 审计发现的问题

### P1 代码指纹有盲区（`r7_pipeline.py:107-108`）

```python
code={str(p.relative_to(REPO)):sha256(p)
      for folder in [ROOT/'src',REPO/'src/image2ppt',REPO/'scripts']
      for p in folder.glob('*') if p.suffix in ('.py','.mjs')}
```

- `folder.glob('*')` **不递归**，子目录里的代码改动不入指纹
- 不覆盖 `services/windows/*`（CLI、Office 检查脚本逻辑）、`configs/*.yaml`、`scripts/*.ps1`
- 后果：改了 `check_r71_office.ps1` 或 `local_cli.py`，`run-manifest.json` 不变，旧 run 可被继续复用

### P2 崩溃窗口无自动恢复（`r7_pipeline.py:384` 前）

`tx.commit()` 会先写 `active.json`（权威），再写 `commit.json`，最后才写页面 `result.json`。若进程在 commit 之后、`result.json` 之前中断：

- 重跑该页 → `prepare()` 发现 `active.scene_sha256 != baseline.scene_sha256` → 抛 `Baseline is not the currently accepted scene`
- 代码里**没有**从 `active.json` 反向恢复 `result.json` 的路径，只能人工介入

### P3 非 relation 类失败没有降级通道（`:353`）

第二次尝试的 `raster_ids` 只从 `relations` 里取 FAIL 的元素。若失败原因是 `write_boundary` 或 `text_clean`，`raster_ids` 为空 → 直接 break，该页直接保留基线。这是保守的，但意味着"越界"和"擦不干净"两类失败没有第二次机会。

### P4 业务常量写死在代码里

- `if len(batch['pages'])!=12: raise ValueError('Frozen baseline must contain twelve pages')`（`:502`）
- `order=['01','02','08','03','04','10','11']+[...]`（`:514`）
- `package()` 里交付文件名 `徐州高新区AI交流_r7.1_12页_待人工审阅.pptx` 也是硬编码

这套代码目前**只能处理这一个 12 页 deck**，文档也承认；但常量散落在流程层而不是配置里。

### P5 测试与模块级符号强耦合

大量 `monkeypatch.setattr(pipeline,'build_candidate',...)`、`setattr(pipeline,'clean_check',...)`。291 个测试覆盖很密，但重构函数名或拆分模块会大面积破测试，改造成本被低估。

## 7 建议（按收益）

1. **修 P1**：指纹改成 `rglob` 并纳入 `services/windows`、`configs`、`scripts/*.ps1`。改动小、直接堵住"改了代码还能复用旧 run"。
2. **修 P2**：`process_page` 开头加一条恢复分支 —— 若 `active.json` 的 `tx_id` 属于本页且 `commit.json` 存在，直接用 commit 回执重建 `result.json`。
3. **修 P3**：把 `raster_ids` 的候选来源扩展到所有 FAIL 的 gate（越界对象、擦不净的文字），让降级路径覆盖全部失败类型。
4. **抽 P4 常量**：页数、页序、交付文件名进 config。
5. **暂不动 P5**：测试耦合虽不理想，但它是当前唯一的回归网，重构前先补集成级测试。

## 8 本次未深审的部分

`r7_edit_checks.py`（39.8 KB）、`surface_rebuild.py`（41.4 KB）、`text_repair.py`（25.5 KB）只做了函数级通读，未逐行核对判据。这三块是编辑探针、原生卡片底和擦字，属于下一轮审计重点。
