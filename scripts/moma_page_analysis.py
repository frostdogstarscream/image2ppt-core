"""Three bounded visual analyses for one page, with durable call reservations.

Each invocation sends exactly one MoMA request. The API key is read from stdin
and kept only in memory; unknown responses are never retried automatically.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from model_slideir_loop import digest, write_json

ENDPOINT = "https://moma.cmecloud.cn/v1/chat/completions"
MODEL = "qwen/qwen3.8-max"


def retry_after_seconds(value: str | None) -> float | None:
    """Parse the service's Retry-After delta seconds or HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _image(path: Path) -> dict:
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}


def _prompt(phase: str, ir: dict, candidates: dict | None) -> str:
    if phase == "style":
        nodes = [{k: n.get(k) for k in ("id", "text", "bbox", "color", "bold", "runs")}
                 for n in ir["nodes"] if n["type"] == "text"]
        return ("你是 PPT 截图中的文字强调证据审查员。第一张图是原图。"
                "只按原图判断每个现有文字节点的字重、颜色、是否有同一句内的局部加粗或红色强调。"
                "现有 bold 字段只是旧拟合结果，不能当作证据。不得改写任何文字、坐标或节点 ID。"
                "返回纯 JSON: {schema:'style-review/1',source_sha256,items:[{id,bold,"
                "runs:[{text,bold,color}] 或 null,evidence,confidence}],red_present,notes}. "
                "items 覆盖全部文字节点；若无局部样式，runs 为 null。若有 runs，拼接必须等于完整节点文字。"
                "color 为六位十六进制 RGB，无红字时不要制造红色。"
                "原图与当前文字清单如下：" + json.dumps({"source_sha256": ir["source_sha256"],
                                                "nodes": nodes}, ensure_ascii=False))
    if phase == "assets":
        return ("你是截图对象边界与归属审查员。第一张图是原图，第二张图是已经清除文字和独立图标的背景板。"
                "请核查程序自动检测出的左侧面板、右侧五条底板、底部横幅，以及左侧三个勾选框。"
                "只根据图像指出候选区域是否对应完整对象，有无文字/图标混入、重叠或漏取。"
                "不得自行创造源图没有的对象。返回纯 JSON: {schema:'asset-review/1',"
                "source_sha256,plate_sha256,objects:[{candidate_id,accept,role,reason}],notes}. "
                "objects 必须恰好覆盖下列 10 个 candidate_id，每个仅一次，包括三个 checkbox-*；"
                "勾选框也写在 objects 中，不要另设 checkboxes 列表。"
                "source_sha256 必须严格复制第一张原图的哈希 " + ir["source_sha256"] +
                "，plate_sha256 必须严格复制第二张背景板的哈希 " +
                candidates["source_plate_sha256"] + "，这两个哈希不同，不可互换。"
                "accept 只能为布尔值；若不确定完整边界，填 false 并说明。"
                "程序候选：" + json.dumps(candidates, ensure_ascii=False))
    if phase == "review":
        return ("你是 PPT 最终回渲染审查员。第一张图是原图，第二张图是最终 PPTX 回渲染。"
                "检查整体风格、文字完整与强调、是否有重叠或越界、独立底板及图标的外观。"
                "轻微字体或斜体差异不是主要问题。不要把图像相似度当作对象可编辑验证。"
                "返回纯 JSON: {schema:'render-review/1',source_sha256,"
                "preview_sha256,critical_issues:[字符串],style_issues:[字符串],"
                "asset_issues:[字符串],overall_visual_status:'pass'或'needs_review',notes:[字符串]}。"
                "source_sha256 必须严格复制原图哈希 " + ir["source_sha256"] +
                "；preview_sha256 必须严格复制回渲染哈希 " +
                candidates["preview_sha256"] + "；不可互换。"
                "输入：" + json.dumps(candidates, ensure_ascii=False))
    raise ValueError("unknown phase")


def run(args: argparse.Namespace, api_key: str | None = None) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite a model-call record")
    source = args.source.resolve(strict=True)
    ir = json.loads(args.ir.resolve(strict=True).read_text(encoding="utf-8"))
    if digest(source) != ir["source_sha256"]:
        raise ValueError("source hash does not match IR")
    candidates = (json.loads(args.candidates.resolve(strict=True).read_text(encoding="utf-8"))
                  if args.candidates else None)
    images = [source]
    if args.phase == "assets":
        if not args.plate:
            raise ValueError("assets phase needs the cleaned plate")
        images.append(args.plate.resolve(strict=True))
    if args.phase == "review":
        if not args.preview:
            raise ValueError("review phase needs a final PPTX preview")
        images.append(args.preview.resolve(strict=True))
    prompt = _prompt(args.phase, ir, candidates)
    image_hashes = {p.name + "-" + str(index): digest(p)
                    for index, p in enumerate(images)}
    output.mkdir(parents=True)
    reservation = {"status": "UNKNOWN_UNTIL_RECORDED", "phase": args.phase,
                   "model": MODEL, "endpoint": ENDPOINT, "source_sha256": digest(source),
                   "images": image_hashes,
                   "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                   "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "reservation.json", reservation)
    key = (api_key if api_key is not None else
           getpass.getpass("MoMA key (memory only): ")).strip()
    if not key:
        write_json(output / "reservation.json", {**reservation, "status": "NO_REQUEST"})
        raise ValueError("MoMA API key missing from stdin; no request sent")
    content = [{"type": "text", "text": prompt}, *[_image(p) for p in images]]
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "temperature": 0, "max_tokens": args.max_tokens, "stream": False,
               "enable_thinking": False}
    try:
        response = requests.post(ENDPOINT, headers={"Authorization": "Bearer " + key,
                                                    "Content-Type": "application/json"},
                                 json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        write_json(output / "failure.json", {"status": "RESPONSE_UNKNOWN",
                                               "error_type": type(exc).__name__})
        raise RuntimeError("MoMA transport failed; response unknown, do not retry") from None
    finally:
        key = ""
    if response.status_code != 200:
        try:
            body = response.json() if response.headers.get("Content-Type", "").startswith(
                "application/json") else {}
        except ValueError:
            body = {}
        error = body.get("error", {}) if isinstance(body, dict) else {}
        if not isinstance(error, dict):
            error = {}
        write_json(output / "failure.json", {"status": "HTTP_ERROR",
                                               "http_status": response.status_code,
                                               "retry_after": response.headers.get("Retry-After"),
                                               "retry_after_seconds": retry_after_seconds(
                                                   response.headers.get("Retry-After")),
                                               "error_code": error.get("code"),
                                               "error_type": error.get("type")})
        write_json(output / "reservation.json", {**reservation,
                                                   "status": "HTTP_ERROR_RECORDED",
                                                   "http_status": response.status_code})
        raise RuntimeError(f"MoMA HTTP {response.status_code}; response body withheld")
    raw = response.json()
    write_json(output / "raw-response.json", raw)
    write_json(output / "reservation.json", {**reservation, "status": "RESPONSE_RECORDED"})
    message = raw["choices"][0]["message"]["content"]
    if isinstance(message, list):
        message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
    if not isinstance(message, str):
        raise ValueError("MoMA returned no text")
    cleaned = message.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
    try:
        answer = json.loads(cleaned)
    except json.JSONDecodeError:
        write_json(output / "parse.json", {"status": "INVALID_JSON"})
        raise
    write_json(output / "answer.json", answer)
    result = {"status": "RECORDED", "phase": args.phase, "model": raw.get("model", MODEL),
              "usage": raw.get("usage", {}), "finish_reason": raw["choices"][0].get("finish_reason"),
              "source_sha256": digest(source), "images": image_hashes}
    write_json(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["style", "assets", "review"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--plate", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-tokens", type=int, default=5000)
    options = parser.parse_args()
    print(json.dumps(run(options), ensure_ascii=False, indent=2))
