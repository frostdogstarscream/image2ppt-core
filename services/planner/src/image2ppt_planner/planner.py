"""Planning loop: inventory -> model -> validated ElementPlan/2.

Every accepted plan has passed the authoritative validator from
``image2ppt_vision.supervised_elements``. A response that fails validation is
sent back to the model with the exact error, at most ``max_attempts`` times.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .client import ChatReply, MomaClient, strip_code_fence
from .config import Settings
from .contract import SYSTEM_PROMPT, build_user_prompt, repair_prompt

DEFAULT_FONTS = ["微软雅黑", "黑体", "宋体", "等线", "思源黑体", "思源宋体"]


def _digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inventory(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        return source
    return json.loads(Path(source).read_text(encoding="utf-8"))


def available_fonts() -> list[str]:
    try:
        from image2ppt_vision.typography import font_status  # type: ignore

        families = font_status().get("families") or []
        if families:
            return list(families)
    except Exception:
        pass
    return DEFAULT_FONTS


def compact_inventory(inventory: dict, fonts: list[str]) -> dict:
    """The model-facing view: only what it needs, with row height as a size hint."""
    texts = []
    for t in inventory.get("texts", []):
        x, y, w, h = t["bbox"]
        texts.append(
            {
                "id": t["id"],
                "bbox": [x, y, w, h],
                "text": t["text"],
                "px": round(h * 0.78, 1),
            }
        )
    candidates = []
    for c in inventory.get("candidates", []):
        candidates.append(
            {
                "id": c["id"],
                "bbox": c["bbox"],
                "hint": c.get("geometry_hint"),
                "ocr_inside": c.get("ocr_inside", []),
            }
        )
    return {
        "schema": inventory.get("schema"),
        "source_sha256": inventory["source_sha256"],
        "image_size": inventory["image_size"],
        "fonts": fonts,
        "note": "texts[].bbox 与 text 必须原样抄进 text_decisions；candidates[].id 可用作 objects[].candidate_id",
        "texts": texts,
        "candidates": candidates,
    }


def prepare_image(path: str | Path, settings: Settings) -> tuple[str, dict]:
    """Return an OpenAI-compatible data URL plus a small note about any re-encode."""
    path = Path(path)
    raw = path.read_bytes()
    note = {"source": str(path), "original_bytes": len(raw), "source_sha256": _digest_file(path)}
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        width, height = img.size
        longest = max(width, height)
        resized = False
        if longest > settings.max_image_side:
            scale = settings.max_image_side / longest
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
            resized = True
        if path.suffix.lower() == ".png" and not resized and len(raw) <= settings.max_image_bytes:
            mime, payload = "image/png", raw
        else:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=92, subsampling=0)
            mime, payload = "image/jpeg", buffer.getvalue()
            resized = True
    note.update(
        {
            "mime": mime,
            "bytes": len(payload),
            "reencoded": resized,
            "sent_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii"), note


def _validate(plan: Any, inventory: dict) -> None:
    """Two gates, both authoritative.

    ``validate_payload`` is the service's admission check (allowed field names and
    bounds) and runs first, because a plan can be semantically fine yet still be
    rejected at submission for an unknown key such as a malformed gradient.
    ``validate_element_plan`` then checks the semantics against the inventory.
    """
    from image2ppt_vision.supervised_elements import validate_element_plan  # type: ignore

    try:
        from image2ppt_vision.element_service import validate_payload  # type: ignore
    except Exception:  # pragma: no cover - older service builds lack it
        validate_payload = None
    if validate_payload is not None:
        validate_payload(plan)
    validate_element_plan(plan, inventory)


ERROR_HINTS = {
    "plan source binding mismatch": "source_sha256 或 image_size 与清单不一致，请原样照抄清单的值。",
    "OCR decisions must cover exactly once": "text_decisions 没有逐条覆盖清单里的 OCR id，请补齐且不要重复。",
    "object bound exceeded": "objects 数量或 objects+texts 总数超限（≤400 / ≤500），请合并小物件。",
    "large raster fallback rejected": "某个 raster_asset 面积超过整页 18%，必须拆小或改成 native_shape。",
    "invalid or duplicate object id": "id 只能含 A-Za-z0-9_-，长度 1–70，且全局唯一。",
    "unsupported native geometry": "geometry 不在允许列表内。",
    "ocr decisions": "text_decisions 与清单不匹配。",
    # The validator indexes some nested fields directly, so a missing key surfaces as KeyError.
    "width_pt": "native_shape 的 line 对象必须同时给 color 和 width_pt（0–20 的数字）。",
    "color": "颜色必须是六位十六进制、不带 # 的字符串，例如 0B5FA5。",
    "invalid element plan fields": "计划里出现了契约不允许的字段名或多余的键，重点检查 fill_gradient 必须是 {angle_deg, stops:[{offset,color}]}。",
}


def explain_error(exc: Exception) -> str:
    message = str(exc)
    if not message:
        message = type(exc).__name__
    for needle, hint in ERROR_HINTS.items():
        if needle in message:
            return f"{type(exc).__name__}: {message} → {hint}"
    return f"{type(exc).__name__}: {message}"


@dataclass
class PlanResult:
    plan: dict
    attempts: int
    model: str
    usage: dict = field(default_factory=dict)
    artifacts: str | None = None
    image_note: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_record(self) -> dict:
        return {
            "attempts": self.attempts,
            "model": self.model,
            "usage": self.usage,
            "artifacts": self.artifacts,
            "image": self.image_note,
            "errors": self.errors,
            "elapsed_s": round(self.elapsed_s, 2),
            "objects": len(self.plan.get("objects", [])),
            "texts": len(self.plan.get("text_decisions", [])),
        }


def _require_native_text(plan: dict, settings: Settings) -> None:
    """Every text decision must stay editable unless the caller explicitly allows rasters.

    A label kept inside a crop still looks right, but the reader cannot change it.
    Treating that as a planning error - and telling the model exactly which ids -
    is what keeps "all text is editable" a property instead of an accident.
    """
    if not getattr(settings, "require_native_text", True):
        return
    retained = [t["id"] for t in plan.get("text_decisions", []) if t.get("action") == "retained_in_asset"]
    if retained:
        raise ValueError(
            "text_retained_in_raster: %s — 这些文字被留在切片里，读者无法编辑。"
            "除确实压在照片/插画上无法分离的情况，请改成 action=\"native\" 并给出 font_family/font_confidence/font_evidence；"
            "若是切片压住了文字，请把该 raster_asset 拆小或改成 native_shape。" % ", ".join(retained[:8])
        )


async def plan_page(
    settings: Settings,
    image_path: str | Path,
    inventory: dict | str | Path,
    *,
    fonts: list[str] | None = None,
    out_dir: str | Path | None = None,
    client: MomaClient | Any | None = None,
) -> PlanResult:
    """Ask the model for a plan and return it only once the validator accepts it."""
    settings.require_key()
    inventory = load_inventory(inventory)
    fonts = fonts or available_fonts()
    data_url, image_note = prepare_image(image_path, settings)
    payload = compact_inventory(inventory, fonts)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(payload)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    artifacts = Path(out_dir) if out_dir else settings.artifacts_root / time.strftime("%Y%m%d-%H%M%S")
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "inventory.compact.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifacts / "image.json").write_text(json.dumps(image_note, ensure_ascii=False, indent=2), encoding="utf-8")

    client = client or MomaClient(settings)
    errors: list[str] = []
    started = time.time()
    last_reply: ChatReply | None = None

    for attempt in range(1, settings.max_attempts + 1):
        reply: ChatReply = await client.chat(messages)
        last_reply = reply
        (artifacts / f"response-{attempt}.json").write_text(
            json.dumps(reply.raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        text = strip_code_fence(reply.text)
        try:
            plan = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"attempt {attempt}: JSON 解析失败 {exc}")
            messages.append({"role": "assistant", "content": reply.text[:2000]})
            messages.append(
                {
                    "role": "user",
                    "content": f"输出不是合法 JSON（{exc}）。请只输出一个 JSON 对象，不要代码围栏、不要解释。",
                }
            )
            continue
        try:
            plan = _validate_with_repair(plan, inventory)
            _require_native_text(plan, settings)
        except Exception as exc:  # noqa: BLE001 - validator raises plain ValueError
            detail = explain_error(exc)
            errors.append(f"attempt {attempt}: {detail}")
            messages.append({"role": "assistant", "content": reply.text[:2000]})
            messages.append({"role": "user", "content": repair_prompt(detail, attempt)})
            continue

        (artifacts / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        result = PlanResult(
            plan=plan,
            attempts=attempt,
            model=reply.model,
            usage=reply.usage,
            artifacts=str(artifacts),
            image_note=image_note,
            errors=errors,
            elapsed_s=time.time() - started,
        )
        (artifacts / "provenance.json").write_text(
            json.dumps(result.to_record(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    raise RuntimeError(
        "模型连续 %d 次未产出可校验的计划。错误序列：%s"
        % (settings.max_attempts, " | ".join(errors[-3:]) if errors else "未知")
    )


def _validate_with_repair(plan: Any, inventory: dict) -> dict:
    _validate(plan, inventory)
    return plan
