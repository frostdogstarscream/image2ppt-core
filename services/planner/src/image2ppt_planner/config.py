"""Runtime settings, all from the environment. Secrets are never written to disk."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://moma.cmecloud.cn/v1"
# Measured on 2026-11-04: this model answers text and reads a real slide image through
# the moma gateway. hacm/qwen2.5-vl-32b-instruct returned 402 (no package on the account).
DEFAULT_MODEL = "qwen/qwen3.8-max"


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 16000
    timeout_s: float = 180.0
    max_attempts: int = 3
    transport_attempts: int = 3
    max_image_side: int = 2000
    max_image_bytes: int = 4 * 1024 * 1024
    response_format: str = "json_object"
    disable_thinking: bool = True
    require_native_text: bool = True
    artifacts_root: Path = field(default_factory=lambda: Path("planner-runs"))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_url=_first_env("PLANNER_BASE_URL", "MOMA_BASE_URL", "VISION_LLM_BASE_URL") or DEFAULT_BASE_URL,
            api_key=_first_env("PLANNER_API_KEY", "MOMA_API_KEY", "VISION_LLM_API_KEY") or "",
            model=_first_env("PLANNER_MODEL", "MOMA_MODEL", "VISION_LLM_MODEL") or DEFAULT_MODEL,
            temperature=float(_first_env("PLANNER_TEMPERATURE") or 0.0),
            max_tokens=int(_first_env("PLANNER_MAX_TOKENS") or 16000),
            timeout_s=float(_first_env("PLANNER_TIMEOUT") or 180.0),
            max_attempts=int(_first_env("PLANNER_MAX_ATTEMPTS") or 3),
            max_image_side=int(_first_env("PLANNER_MAX_IMAGE_SIDE") or 2000),
            response_format=_first_env("PLANNER_RESPONSE_FORMAT") or "json_object",
            disable_thinking=(_first_env("PLANNER_THINKING") or "off").lower() not in ("on", "1", "true"),
            require_native_text=(_first_env("PLANNER_ALLOW_RASTER_TEXT") or "0").lower() not in ("1", "true", "on"),
            artifacts_root=Path(_first_env("PLANNER_ARTIFACTS") or "planner-runs"),
        )

    def require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "没有找到模型凭据。请通过环境变量注入（任选其一）："
                "PLANNER_API_KEY / MOMA_API_KEY / VISION_LLM_API_KEY。"
                "不要把密钥写进代码、JSON 或提交记录。"
            )

    def describe(self) -> dict:
        """Safe to log and to return over HTTP: never contains the key."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_present": bool(self.api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_attempts": self.max_attempts,
            "response_format": self.response_format,
            "disable_thinking": self.disable_thinking,
            "require_native_text": self.require_native_text,
        }
