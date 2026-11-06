"""Minimal OpenAI-compatible chat client for the moma gateway (vision capable).

No secret ever reaches a log line, an artifact file, or an HTTP response.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass

import httpx

from .config import Settings


class ModelError(RuntimeError):
    """Raised when the gateway cannot be used; the message never contains the key."""


@dataclass
class ChatReply:
    text: str
    model: str
    usage: dict
    finish_reason: str | None
    request_sha256: str
    raw: dict

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "request_sha256": self.request_sha256,
        }


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class MomaClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        settings.require_key()
        self.settings = settings
        self._transport = transport

    async def chat(self, messages: list[dict]) -> ChatReply:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        if self.settings.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.settings.disable_thinking:
            # Measured on the moma gateway: this removes reasoning_tokens entirely.
            payload["enable_thinking"] = False

        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.settings.transport_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self.settings.timeout_s, transport=self._transport) as client:
                    response = await client.post(url, headers=headers, json=payload)
                if response.status_code in (400, 422) and "response_format" in payload:
                    if self.settings.transport_attempts == 1:
                        raise ModelError(f"HTTP {response.status_code}: {response.text[:300]}")
                    # Some gateways reject response_format; retry once without it.
                    payload.pop("response_format", None)
                    continue
                if response.status_code in (400, 422) and "enable_thinking" in payload:
                    if self.settings.transport_attempts == 1:
                        raise ModelError(f"HTTP {response.status_code}: {response.text[:300]}")
                    payload.pop("enable_thinking", None)
                    continue
                if response.status_code == 429:
                    await asyncio.sleep(5 * attempt)
                    last_error = ModelError("网关限流（429），已重试")
                    continue
                if response.status_code >= 500:
                    await asyncio.sleep(3 * attempt)
                    last_error = ModelError(f"网关错误（{response.status_code}），已重试")
                    continue
                if response.status_code in (401, 403):
                    raise ModelError(
                        f"鉴权失败（HTTP {response.status_code}）。请确认 PLANNER_API_KEY 有效且有权访问模型 {self.settings.model}。"
                    )
                if response.status_code != 200:
                    raise ModelError(f"网关返回 HTTP {response.status_code}: {response.text[:300]}")
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
                return ChatReply(
                    text=text,
                    model=data.get("model") or self.settings.model,
                    usage=data.get("usage") or {},
                    finish_reason=choice.get("finish_reason"),
                    request_sha256=_sha({k: v for k, v in payload.items() if k != "messages"}),
                    raw=data,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ModelError(f"网络错误：{type(exc).__name__}，已重试")
                await asyncio.sleep(2 * attempt)
        raise last_error or ModelError("模型调用失败")


def strip_code_fence(text: str) -> str:
    """Models occasionally wrap JSON in a fence even when told not to."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned
