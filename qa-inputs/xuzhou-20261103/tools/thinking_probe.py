"""Which switch actually disables reasoning on this gateway, and what does it save?

Sends the same small request four ways and reports wall time + reasoning_tokens.
Key comes from stdin.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "https://moma.cmecloud.cn/v1/chat/completions"
KEY = sys.stdin.readline().strip()
MODEL = "qwen/qwen3.8-max"
PROMPT = "用一句话说明什么是幻灯片。"

VARIANTS = [
    ("baseline", {}),
    ("enable_thinking=false", {"enable_thinking": False}),
    ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("/no_think suffix", {"__suffix": "/no_think"}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
]


def call(extra):
    suffix = extra.pop("__suffix", "")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT + suffix}],
        "max_tokens": 200,
        "temperature": 0,
    }
    payload.update(extra)
    req = urllib.request.Request(
        BASE, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = round(time.time() - started, 1)
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        choice = (data.get("choices") or [{}])[0]
        return {
            "ok": True, "s": elapsed,
            "completion": usage.get("completion_tokens"),
            "reasoning": details.get("reasoning_tokens"),
            "reply_head": (choice.get("message") or {}).get("content", "")[:40],
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return {"ok": False, "http": exc.code, "error": body[:180], "s": round(time.time() - started, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__, "s": round(time.time() - started, 1)}


print(f"{'variant':26} {'ok':>4} {'sec':>6} {'compl':>6} {'reason':>7}  note")
for name, extra in VARIANTS:
    result = call(dict(extra))
    note = result.get("error") or result.get("reply_head") or ""
    print(f"{name:26} {str(result.get('ok')):>4} {result.get('s'):>6} "
          f"{str(result.get('completion')):>6} {str(result.get('reasoning')):>7}  {str(note)[:40]}")
