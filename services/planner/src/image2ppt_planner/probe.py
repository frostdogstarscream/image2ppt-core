"""Capability probe: which gateway models are reachable and which accept images.

Nothing here is required for planning; it exists to answer "can I use this model"
with measurements instead of assumptions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings

CANDIDATES = [
    "qwen/qwen3.8-max",
    "hacm/qwen2.5-vl-32b-instruct",
    "qwen/qwen3.5-omni-flash",
    "qwen/qwen3.5-omni-plus-realtime",
    "deepseek-v4.1-flash",
    "glm-5.3",
]


@dataclass
class ProbeResult:
    model: str
    text: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {"model": self.model, "text": self.text, "vision": self.vision}


async def _post(settings: Settings, model: str, content: Any, max_tokens: int = 16) -> dict:
    url = settings.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=settings.timeout_s) as client:
            response = await client.post(url, headers=headers, json=payload)
        elapsed = round(time.time() - started, 2)
        if response.status_code == 200:
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            return {"ok": True, "http": 200, "elapsed_s": elapsed, "reply": str(text)[:60]}
        return {
            "ok": False,
            "http": response.status_code,
            "elapsed_s": elapsed,
            "error": response.text[:200],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "http": None, "elapsed_s": round(time.time() - started, 2), "error": f"{type(exc).__name__}: {exc}"}


async def probe_models(settings: Settings, models: list[str], image_data_url: str | None = None) -> list[ProbeResult]:
    """Text reachability for every model; vision only for models that passed text."""
    settings.require_key()
    results: list[ProbeResult] = []
    for model in models:
        result = ProbeResult(model=model)
        result.text = await _post(settings, model, "只回复两个字：收到")
        if image_data_url and result.text.get("ok"):
            content = [
                {"type": "text", "text": "这张图里最大的字写的是什么？只回答那几个字。"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
            result.vision = await _post(settings, model, content, max_tokens=32)
        elif image_data_url:
            result.vision = {"ok": False, "skipped": "文本调用失败，未测图片"}
        results.append(result)
    return results


def summarise(results: list[ProbeResult]) -> dict:
    usable = [r.model for r in results if r.text.get("ok")]
    vision_ok = [r.model for r in results if r.vision.get("ok")]
    return {
        "models_tested": len(results),
        "text_ok": usable,
        "vision_ok": vision_ok,
        "recommended": vision_ok[0] if vision_ok else (usable[0] if usable else None),
        "detail": [r.row() for r in results],
    }
