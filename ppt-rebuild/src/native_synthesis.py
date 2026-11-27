"""Entry point for observation, component planning and blank-canvas authoring."""

import asyncio
import shutil
import sys
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from rebuild_pipeline import ArtifactBackend, REPO, compile_ir, stamp
from ocr_engine import detect
from slideir_adapter import sha256, write_json

sys.path.insert(0, str(REPO / "services/planner/src"))
from image2ppt_planner.config import Settings
from image2ppt_planner.client import MomaClient, strip_code_fence
from image2ppt_planner.planner import prepare_image
from native_plan import make_plan


async def inspect_scenery(plan, pack, source, directory, settings):
    description = plan["background"].get("scenery")
    if not pack or str(description or "").lower().strip() in ("", "none", "null", "无"):
        return None
    from layerd_loader import load_page
    from PIL import ImageChops

    page = load_page(pack)
    with Image.open(source) as original, Image.open(page.source_path) as bound:
        if (
            original.size != bound.size
            or ImageChops.difference(original.convert("RGB"), bound.convert("RGB")).getbbox()
        ):
            raise ValueError("Scenery pack belongs to a different source")
    path = directory / "scenery-candidate.png"
    bottom = page.layers[0]
    if bottom.role != "background":
        return None
    with Image.open(bottom.path) as raw, Image.open(source) as src:
        if raw.convert("RGBA").getchannel("A").getextrema()[0] < 255:
            return None
        candidate = raw.convert("RGB")
        if not ImageChops.difference(candidate, src.convert("RGB")).getbbox():
            return None
        candidate.save(path)
    recognized = [l.text for l in detect(path) if l.score >= 0.65]
    if recognized:
        write_json(
            directory / "scenery-review.json", {"accepted": False, "visible_readings": recognized}
        )
        return None
    data, _ = prepare_image(path, settings)
    prompt = (
        "Inspect this proposed slide background ALONE. List each visible business mark, any letters, logos, "
        "numbers, panels, buttons, photos in cards, ghost lettering, or recognizable outlines. Return JSON "
        '{"scenic":true only for a genuine clean ornamental landscape or texture,"marks":[...]}'
    )
    try:
        reply = await MomaClient(settings).chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data}},
                    ],
                }
            ]
        )
        result = json.loads(strip_code_fence(reply.text))
        accept = result.get("scenic") is True and result.get("marks") == []
        record = {
            "accepted": accept,
            "findings": result,
            "call": reply.to_record(),
            "sha256": sha256(path),
        }
    except (RuntimeError, ValueError) as error:
        accept = False
        record = {"accepted": False, "reason": type(error).__name__}
    write_json(directory / "scenery-review.json", record)
    return path if accept else None


def run_native(
    image,
    output,
    *,
    pack=None,
    layerd_url=None,
    token_file=None,
    inpaint_url=None,
    max_iterations=3,
    timeout=1200,
    probes=True,
):
    image = Path(image).resolve()
    directory = Path(output).resolve() / (image.stem + "-" + stamp())
    directory.mkdir(parents=True)
    source = directory / "source.png"
    with Image.open(image) as original:
        original.convert("RGB").save(source)
    with Image.open(source) as src:
        size = src.size
    lines = [asdict(line) for line in detect(source)]
    write_json(
        directory / "ocr.json",
        {"input_sha256": sha256(image), "source_sha256": sha256(source), "lines": lines},
    )
    print(f"[native] {image.name}: {len(lines)} OCR lines, observing whole page", flush=True)
    settings = Settings.from_env()
    plan = asyncio.run(make_plan(source, size, lines, settings, directory))
    scenery = asyncio.run(inspect_scenery(plan, pack, source, directory, settings))
    return render_native_page(
        directory,
        source,
        plan,
        input_sha256=sha256(image),
        pack=pack,
        layerd_url=layerd_url,
        token_file=token_file,
        inpaint_url=inpaint_url,
        max_iterations=max_iterations,
        timeout=timeout,
        probes=probes,
        scenery=scenery,
    )


def render_native_page(
    directory,
    source,
    plan,
    *,
    input_sha256,
    pack,
    layerd_url,
    token_file,
    inpaint_url,
    max_iterations,
    timeout,
    probes=True,
    scenery=None,
):
    from native_canvas import build_canvas

    if sha256(source) != plan["source_sha256"]:
        raise ValueError("Component plan does not belong to this source image")
    ir, report = build_canvas(
        directory,
        source,
        plan,
        pack=pack,
        layerd_url=layerd_url,
        token_file=token_file,
        inpaint_url=inpaint_url,
        max_iterations=max_iterations,
        timeout=timeout,
        scenery=scenery,
    )
    path = directory / "slideir.json"
    write_json(path, ir)
    write_json(directory / "object-report.json", report)
    write_json(
        directory / "authoring.json",
        {
            "ir_sha256": sha256(path),
            "tool": "native_synthesis",
            "model_id": plan["model_calls"][-1]["model"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "review_notes": ir["review_notes"],
        },
    )
    backend = ArtifactBackend(REPO, timeout)
    result = compile_ir(backend, path)
    if probes:
        from native_layout import overflow_adjustments
        from native_office import measure_office_fields

        initial = measure_office_fields(result["pptx"])
        changes = overflow_adjustments(ir, initial)
        final = initial
        if changes:
            for node in ir["nodes"]:
                if node["id"] in changes:
                    node["font"]["size_pt"] = changes[node["id"]]
            sidecar = directory / "authoring.json"
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            path.rename(directory / "before-office-fit.json")
            sidecar.rename(directory / "before-office-fit-authoring.json")
            write_json(path, ir)
            write_json(sidecar, dict(previous, ir_sha256=sha256(path)))
            result = compile_ir(backend, path)
            final = measure_office_fields(result["pptx"])
        write_json(
            directory / "office-text-fit.json",
            dict(
                before=initial,
                adjusted=changes,
                after=final,
                unresolved=list(overflow_adjustments(ir, final)),
            ),
        )
    for field, name in (("pptx", "reconstructed.pptx"), ("preview", "preview.png")):
        shutil.copyfile(result[field], directory / name)
    delivery = dict(
        directory=str(directory),
        pptx=str(directory / "reconstructed.pptx"),
        preview=str(directory / "preview.png"),
        input_sha256=input_sha256,
        native_variant="native_synthesis",
        build=result,
    )
    write_json(directory / "delivery.json", delivery)
    return delivery
