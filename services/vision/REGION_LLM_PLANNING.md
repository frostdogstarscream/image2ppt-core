# Region LLM planner（影子模式）

该流程只提出“区域该原生、混合还是整图保留”的建议，不修改 IR 或 deck。人工决定仍是权威。

## 使用

```bash
# 1. 从已验收 plans 提取人工决定（P12 保持 native）
python scripts/region_shadow.py human-decisions \
  --plans .build/region-shadow/plans.json \
  --output .build/region-shadow/human-decisions.json

# 2. 离线准备 job；不联网
python scripts/region_shadow.py prepare-llm \
  --inventory .build/region-shadow/inventory.json \
  --jobs .build/region-shadow/llm-jobs-v2 \
  --model kimi/kimi-k2.6 \
  --base-url https://moma.cmecloud.cn/v1

# 3. 唯一网络入口；密钥从 stdin 读取，不落盘
python scripts/run_supervised_jobs_secure.py .build/region-shadow/llm-jobs-v2 \
  --jobs P5-example-058-dashboard P6-funds-analysis-chart ... --key-stdin

# 4. 解析保存的响应，严格校验后生成 plans.llm.json
python scripts/region_shadow.py apply-llm \
  --inventory .build/region-shadow/inventory.json \
  --jobs .build/region-shadow/llm-jobs-v2 \
  --output .build/region-shadow/plans.llm.json \
  --human-decisions .build/region-shadow/human-decisions.json

python scripts/region_shadow.py validate --plans .build/region-shadow/plans.llm.json \
  --output .build/region-shadow/validation.llm.json \
  --human-decisions .build/region-shadow/human-decisions.json
python scripts/region_shadow.py score --plans .build/region-shadow/plans.llm.json \
  --output .build/region-shadow/score.llm.json \
  --human-decisions .build/region-shadow/human-decisions.json
```

## 发送与不发送

发送：区域 crop、9 个数值特征、分类、能力快照、prompt。
不发送：人工决定/备注、description、整页、OCR 文本、路径、bbox、region_id。

响应只保留哈希和经过校验的 agent 决策；`response.body` 留在 gitignored 的 `.build`，不发布。
密钥只注入 secure runner 的子进程环境，不写入 job、settings、报告或 git。
