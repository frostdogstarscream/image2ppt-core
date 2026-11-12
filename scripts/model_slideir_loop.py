"""Model-authored SlideIR, fixed validation/compile, and two bounded repair turns.

The model owns page planning. This runner only seals input provenance, checks the
fixed protocol, compiles it, and returns the actual PPTX preview for review.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ("You reconstruct one screenshot as editable SlideIR 0.1 JSON. Preserve all visible "
          "wording, numbers, layout and proportions. Use native text and shapes only for evidence "
          "you can identify. Keep uncertain visual areas as supplied image assets. Do not overlay "
          "native text on raster pixels containing the same text. Return JSON only. "
          "Do not invent source content or asset paths.")
REPAIR = ("Inspect the source and the actual PPTX preview plus diagnostics. Return JSON "
          "{\"patches\":[{\"id\":\"existing-node-id\",\"changes\":{\"bbox\":[x,y,w,h],"
          "\"font\":{...},\"color\":\"RRGGBB\",\"fill\":\"RRGGBB\"}}],"
          "\"review_notes\":[\"...\"]}. Change only nodes needing local correction. "
          "Never change text, evidence ownership, assets, IDs, z order, or page metadata.")
FORMAT_RULES = (
    "Return the SlideIR object itself, with no slide/elements wrapper. Required top-level fields: "
    "schema_version='0.1', source_image='source/page.png', source_sha256, canvas "
    "{source_px,target_px}, background {color,asset_id?}, review_status='needs_review', "
    "review_notes as a string array, evidence, assets, groups, nodes. Colors are six hex digits "
    "without '#'. Asset purpose is only 'module' or 'cleaned_background'; a cleaned full-slide "
    "plate belongs in background.asset_id, not as an image node. Nodes need unique integer z. "
    "Text nodes need id,type='text',bbox=[x,y,w,h],z,evidence_ids,text,font "
    "{latin,ea,size_pt},color; font.size_pt is points at 96 DPI, not source glyph pixels. "
    "Image nodes need id,type='image',bbox,z,asset_id,evidence_ids=[]. Assets need id,path,"
    "purpose,sha256,contains_evidence_ids. Text evidence wording must equal its node text. "
    "Every evidence ID has exactly one owner. Keep unresolved_evidence_ids=[] only if all "
    "visible content is represented; otherwise retain uncertain pixels in the supplied raster."
)
PATCH_FIELDS = {"bbox", "font", "color", "fill", "line", "align", "wrap", "insets", "vertical_anchor"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_data(path: Path, max_side: int = 1800) -> str:
    import io
    with Image.open(path) as image:
        image = rgb_page(image)
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def rgb_page(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA" or "transparency" in image.info:
        return Image.alpha_composite(Image.new("RGBA", image.size, "white"),
                                     image.convert("RGBA")).convert("RGB")
    return image.convert("RGB")


def request_model(base_url: str, model: str, key: str, timeout: float,
                  messages: list[dict], session_id: str, max_tokens: int) -> dict:
    if not base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
        raise ValueError("base_url must use HTTPS or local HTTP")
    body = json.dumps({"model": model, "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens},
                      ensure_ascii=False).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json",
                                          "User-Agent": "image2ppt-model-takeover/0.1",
                                          "x-opencode-session": session_id})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"model HTTP {exc.code}; response withheld") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"model transport {type(exc).__name__}; response unknown, do not resend automatically") from None
    choice = data["choices"][0]
    content = choice["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise ValueError("model returned no text")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
    return {"answer": json.loads(cleaned), "model": data.get("model", model),
            "usage": data.get("usage", {}), "finish_reason": choice.get("finish_reason")}


def node_cmd() -> str:
    return os.environ.get("IMAGE2PPT_NODE") or shutil.which("node") or "node"


def cc_switch_go_key() -> str:
    """Read the existing OpenCode Go credential in memory; never expose it."""
    store = Path.home() / ".cc-switch/cc-switch.db"
    connection = sqlite3.connect(store.as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute("select settings_config from providers "
                                 "where app_type='opencode' and name='OpenCode Go'").fetchone()
        key = json.loads(row[0])["options"]["apiKey"] if row else ""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("OpenCode Go credential unavailable")
        return key.strip()
    finally:
        connection.close()


def compiler(ir: Path, action: str) -> dict:
    completed = subprocess.run([node_cmd(), str(ROOT / "scripts/main.mjs"), action,
                                "--input", str(ir)], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=1800)
    if completed.returncode:
        # Compiler diagnostics can include source wording; keep them in the local run only.
        raise ValueError((completed.stderr or completed.stdout)[-4000:])
    start = completed.stdout.find("{")
    if start < 0:
        raise ValueError("compiler returned no JSON result")
    return json.loads(completed.stdout[start:])


def seal(ir: dict, source: Path, assets: dict[str, dict]) -> dict:
    with Image.open(source) as image:
        size = list(image.size)
    import copy
    ir = copy.deepcopy(ir)
    if ir.get("schema_version") == "SlideIR 0.1":
        ir["schema_version"] = "0.1"
    elif ir.get("schema_version") != "0.1":
        raise ValueError("unsupported model SlideIR version")
    if ir.get("source_image") not in (None, "source/page.png"):
        raise ValueError("model tried to change source_image")
    if ir.get("source_sha256") not in (None, digest(source)):
        raise ValueError("model tried to change source_sha256")
    if ir.get("canvas", {}).get("source_px") != size:
        raise ValueError("model source canvas does not match image")
    ir["source_image"] = "source/page.png"
    ir["source_sha256"] = digest(source)
    ir["review_status"] = "needs_review"
    if isinstance(ir.get("review_notes"), str):
        ir["review_notes"] = [ir["review_notes"]]
    def canonical_colors(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("color", "fill") and isinstance(child, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", child):
                    value[key] = child[1:]
                else:
                    canonical_colors(child)
        elif isinstance(value, list):
            for child in value:
                canonical_colors(child)
    canonical_colors(ir)
    provided = {a["id"]: a for a in ir.get("assets", [])}
    if set(provided) - set(assets):
        raise ValueError("model referenced unavailable asset")
    for aid, value in provided.items():
        if value.get("path") != assets[aid]["path"] or value.get("sha256") != assets[aid]["sha256"]:
            raise ValueError("model changed asset path or hash")
    return ir


def patch(ir: dict, answer: dict) -> dict:
    import copy
    result = copy.deepcopy(ir)
    nodes = {n["id"]: n for n in result["nodes"]}
    if set(answer) - {"patches", "review_notes"} or not isinstance(answer.get("patches"), list):
        raise ValueError("repair response must contain patches")
    seen = set()
    for entry in answer["patches"]:
        nid, changes = entry["id"], entry["changes"]
        if nid in seen or nid not in nodes or not isinstance(changes, dict) or set(changes) - PATCH_FIELDS:
            raise ValueError("invalid or duplicate local patch")
        seen.add(nid)
        nodes[nid].update(changes)
    notes = answer.get("review_notes", [])
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise ValueError("invalid review notes")
    result["review_notes"] = ir["review_notes"] + notes
    return result


def mae(source: Path, preview: Path) -> float:
    with Image.open(source) as left, Image.open(preview) as right:
        a = left.convert("RGB")
        b = right.convert("RGB").resize(a.size, Image.Resampling.BILINEAR)
        return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3


def run(args, model_call=request_model) -> dict:
    if not 0 < getattr(args, "max_tokens", 8000) <= 16000:
        raise ValueError("max_tokens must be 1..16000")
    if not 256 <= getattr(args, "image_side", 1400) <= 2000:
        raise ValueError("image_side must be 256..2000")
    max_repairs = getattr(args, "max_repairs", 2)
    if max_repairs not in (0, 1, 2):
        raise ValueError("max_repairs must be 0..2")
    source = Path(args.source).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite an existing run")
    ocr = json.loads(Path(args.ocr).read_text(encoding="utf-8")) if args.ocr else []
    with Image.open(source) as image:
        original_size = list(image.size)
    if isinstance(ocr, dict):
        if ocr.get("source_sha256") and ocr["source_sha256"] != digest(source):
            raise ValueError("OCR source hash does not match the input image")
        if ocr.get("image_size") and ocr["image_size"] != original_size:
            raise ValueError("OCR image size does not match the input image")
    key = cc_switch_go_key() if getattr(args, "credential_source", None) == "cc-switch-opencode-go" else os.environ.get(args.api_key_env, "")
    if args.live and not (args.base_url and args.model and key):
        raise ValueError("live run requires base_url, model and configured API key environment variable")
    output.mkdir(parents=True)
    (output / "source").mkdir()
    (output / "assets").mkdir()
    with Image.open(source) as image:
        rgb_page(image).save(output / "source/page.png")
    page = output / "source/page.png"
    assets = {}
    if args.assets_dir:
        for file in sorted(Path(args.assets_dir).resolve(strict=True).iterdir()):
            if file.is_file() and file.suffix.lower() in (".png", ".jpg", ".jpeg"):
                aid = file.stem
                if aid in assets:
                    raise ValueError("duplicate asset ID")
                target = output / "assets" / file.name
                shutil.copyfile(file, target)
                with Image.open(target) as image:
                    dimensions = list(image.size)
                assets[aid] = {"path": "assets/" + file.name, "sha256": digest(target),
                               "size": dimensions}
    with Image.open(page) as image:
        source_size = list(image.size)
    manifest = {"source_sha256": digest(page), "original_sha256": digest(source),
                "source_size": source_size,
                "ocr": ocr, "assets": assets, "protocol": "SlideIR 0.1",
                "model": args.model, "base_url": args.base_url, "key_present": bool(key),
                "max_repairs": max_repairs}
    session_id = "image2ppt-" + uuid.uuid4().hex
    manifest["session_id"] = session_id
    write_json(output / "input-manifest.json", manifest)
    if not args.live:
        result = {"status": "OFFLINE_READY", "run": str(output), "model_calls": 0}
        write_json(output / "result.json", result)
        return result
    prompt = ("Create one complete SlideIR JSON object. " + FORMAT_RULES +
              " Source/OCR/assets manifest: " +
              json.dumps(manifest, ensure_ascii=False) + "\nUse only listed asset IDs, paths and hashes. "
              "Every visible evidence item needs one owner. Keep review_status needs_review.")
    if getattr(args, "seed_response", None):
        previous = json.loads(Path(args.seed_response).read_text(encoding="utf-8"))
        previous = previous.get("answer", previous)
        prompt += ("\nYour previous model answer was a preliminary visual plan, not SlideIR. "
                   "Convert it to the exact public SlideIR 0.1 object; use your observed content "
                   "and do not copy a synthetic example. Previous answer: " +
                   json.dumps(previous, ensure_ascii=False) +
                   "\nRequired top-level keys: schema_version, source_image, source_sha256, "
                   "canvas, background, review_status, review_notes, evidence, assets, groups, nodes. "
                   "No wrapper such as slide or elements. source_image='source/page.png'; "
                   "source_sha256 must equal manifest source_sha256. canvas.source_px equals manifest "
                   "source_size; target_px should preserve aspect ratio. background.color is six-digit RGB. "
                   "Each text node needs id,type='text',bbox=[x,y,w,h],integer z,evidence_ids,text,font "
                   "{latin,ea,size_pt},color. Each image node needs id,type='image',bbox,z,asset_id,"
                   "evidence_ids=[]. Each referenced asset needs id,path,purpose,sha256,"
                   "contains_evidence_ids; asset paths and hashes must exactly match the manifest. "
                   "Evidence entries need id,kind,bbox, and text for text evidence. Every evidence id "
                   "must have exactly one owner. Use groups=[] if unsure, review_status='needs_review', "
                   "unresolved_evidence_ids=[]; output JSON only.")
    visual_inputs = [{"type": "image_url", "image_url":
                      {"url": image_data(page, getattr(args, "image_side", 1400))}}]
    for aid, item in assets.items():
        visual_inputs.extend([{"type": "text", "text": "Available asset ID: " + aid},
                              {"type": "image_url", "image_url":
                               {"url": image_data(output / item["path"], 512)}}])
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": prompt}, *visual_inputs]}]
    attempts = []
    current = None
    for turn in range(1 + max_repairs):
        folder = output / f"turn-{turn}"
        folder.mkdir()
        # A durable reservation avoids silently repeating an uncertain paid request.
        write_json(folder / "reservation.json", {"status": "UNKNOWN_UNTIL_RECORDED", "turn": turn,
                                                   "created_at": datetime.now(timezone.utc).isoformat(),
                                                   "model": args.model})
        try:
            reply = model_call(args.base_url, args.model, key, args.timeout, messages,
                               session_id, getattr(args, "max_tokens", 8000))
        except Exception as exc:
            # A timeout or broken connection may have succeeded remotely. Persist
            # the uncertainty and stop; never turn it into an automatic retry.
            result = {"status": "MODEL_RESPONSE_UNKNOWN", "run": str(output),
                      "model_calls_reserved": turn + 1, "failed_turn": turn,
                      "error_type": type(exc).__name__, "best_turn": None,
                      "human_acceptance": False}
            write_json(folder / "check.json", result)
            write_json(output / "result.json", result)
            return result
        write_json(folder / "model-response.json", {"answer": reply["answer"], "model": reply["model"],
                                                     "usage": reply.get("usage", {}),
                                                     "finish_reason": reply.get("finish_reason")})
        write_json(folder / "reservation.json", {"status": "RESPONSE_RECORDED", "turn": turn})
        if turn and isinstance(reply["answer"], dict) and reply["answer"].get("patches") == []:
            record = {"turn": turn, "status": "NO_CHANGE"}
            write_json(folder / "check.json", record)
            attempts.append(record)
            break
        try:
            candidate = seal(reply["answer"], page, assets) if turn == 0 else patch(current, reply["answer"])
            if current is None:
                current = candidate
            ir_file = folder / "slideir.json"
            write_json(ir_file, candidate)
            write_json(folder / "authoring.json", {"format_version": "1", "tool": "model-slideir-loop",
                                                      "model_id": reply["model"],
                                                      "created_at": datetime.now(timezone.utc).isoformat(),
                                                      "ir_sha256": digest(ir_file)})
            # Compiler resolves paths relative to IR; place identical immutable inputs beside it.
            shutil.copytree(output / "source", folder / "source")
            shutil.copytree(output / "assets", folder / "assets")
            validation = compiler(ir_file, "validate")
            build = compiler(ir_file, "compile")
            preview = Path(build.get("preview") or build.get("preview_png") or "")
            if not preview.is_file():
                raise ValueError("compiler returned no final PPTX preview")
            score = mae(page, preview)
            current = candidate
            record = {"turn": turn, "status": "COMPILED", "ir": str(ir_file), "build": build,
                      "validation": validation, "page_mae": score}
            write_json(folder / "check.json", record)
            attempts.append(record)
            if turn == max_repairs:
                break
            messages = [{"role": "system", "content": REPAIR},
                        {"role": "user", "content": [{"type": "text", "text":
                         json.dumps({"diagnostics": validation, "page_mae": score,
                                     "current_ir": current}, ensure_ascii=False)},
                         {"type": "image_url", "image_url": {"url": image_data(page)}},
                         {"type": "image_url", "image_url": {"url": image_data(preview)}}]}]
        except (ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            record = {"turn": turn, "status": "FAILED", "error": str(exc)[-2000:]}
            write_json(folder / "check.json", record)
            attempts.append(record)
            if turn == max_repairs or current is None:
                break
            messages = [{"role": "system", "content": REPAIR},
                        {"role": "user", "content": [{"type": "text", "text":
                         json.dumps({"error": record["error"], "current_ir": current}, ensure_ascii=False)},
                         {"type": "image_url", "image_url": {"url": image_data(page)}}]}]
    compiled = [a for a in attempts if a["status"] == "COMPILED"]
    best = min(compiled, key=lambda x: x["page_mae"]) if compiled else None
    result = {"status": "REVIEW_REQUIRED" if best else "FAILED", "run": str(output),
              "model_calls": len(attempts), "best_turn": best["turn"] if best else None,
              "best_ir": best["ir"] if best else None, "best_build": best["build"] if best else None,
              "attempts": [{k: v for k, v in a.items() if k != "build"} for a in attempts],
              "human_acceptance": False}
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--ocr")
    parser.add_argument("--assets-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("SLIDEIR_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("SLIDEIR_MODEL", ""))
    parser.add_argument("--api-key-env", default="SLIDEIR_API_KEY")
    parser.add_argument("--credential-source", choices=["cc-switch-opencode-go"])
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("SLIDEIR_TIMEOUT", "180")))
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--image-side", type=int, default=1400)
    parser.add_argument("--seed-response", help="Previous model response to correct into SlideIR")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] not in ("FAILED", "MODEL_RESPONSE_UNKNOWN") else 2
    except Exception as exc:
        print(f"model-slideir-loop: {type(exc).__name__}: {str(exc)[-500:]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
