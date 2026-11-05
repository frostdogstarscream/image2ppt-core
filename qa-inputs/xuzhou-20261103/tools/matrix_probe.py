"""Model matrix probe: text reachability + real image input, one model at a time.

Key is read from stdin so it never appears in argv or the process list.
Usage: python3 matrix.py <image_path> [model ...]  < keyfile
"""
import base64
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "https://moma.cmecloud.cn/v1/chat/completions"
KEY = sys.stdin.readline().strip()
IMAGE = sys.argv[1] if len(sys.argv) > 1 else None
MODELS = sys.argv[2:] or [
    "qwen/qwen3.8-max",
    "hacm/qwen2.5-vl-32b-instruct",
    "qwen/qwen3.5-omni-flash",
    "qwen/qwen3.5-omni-plus-realtime",
    "deepseek-v4.1-flash",
    "glm-5.3",
    "qwen/qwen3.5-35b-a3b",
]


def call(model, content, max_tokens=32):
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": content}],
         "max_tokens": max_tokens, "temperature": 0},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE, data=body, method="POST",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            return {"ok": True, "http": 200, "s": round(time.time() - started, 1), "reply": str(text)[:70]}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "http": exc.code, "s": round(time.time() - started, 1),
                "error": exc.read().decode("utf-8", "replace")[:160]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "http": None, "s": round(time.time() - started, 1), "error": f"{type(exc).__name__}"}


data_url = None
if IMAGE:
    with open(IMAGE, "rb") as fh:
        data_url = "data:image/png;base64," + base64.b64encode(fh.read()).decode()

print(f"{'model':34} {'text':>6} {'vision':>7}  note")
for model in MODELS:
    text = call(model, "只回复两个字：收到", 8)
    vision = {"ok": None}
    if text.get("ok") and data_url:
        vision = call(model, [
            {"type": "text", "text": "这张幻灯片页眉最大的标题文字是什么？只输出那几个字，不要解释。"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ], 24)
    tv = "ok" if text.get("ok") else str(text.get("http", "err"))
    vv = "-" if vision.get("ok") is None else ("ok" if vision.get("ok") else str(vision.get("http", "err")))
    note = (vision.get("reply") or text.get("error") or "")[:70].replace("\n", " ")
    print(f"{model:34} {tv:>6} {vv:>7}  {note}")
