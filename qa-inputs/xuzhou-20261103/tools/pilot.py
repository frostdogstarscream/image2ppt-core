"""Pilot: real page -> CV inventory -> planner -> validated ElementPlan/2.

Runs inside a one-off container from the image2ppt-vision:0.2.1 image with the
0.3.0 sources mounted ahead of site-packages on PYTHONPATH.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

SOURCE = "/data/pilot/page03.png"
ANALYSIS = "/data/pilot/analysis-p03.json"
INVENTORY_DIR = "/data/pilot/inv"
PLAN_DIR = "/data/pilot/plan"
MAX_CANDIDATES = 120


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    from image2ppt_vision.element_inventory import build_inventory

    analysis = load(ANALYSIS)
    if Path(INVENTORY_DIR).exists():
        import shutil

        shutil.rmtree(INVENTORY_DIR)
    inventory = build_inventory(SOURCE, analysis, INVENTORY_DIR, maximum=MAX_CANDIDATES)
    print(json.dumps({
        "step": "inventory",
        "texts": len(inventory["texts"]),
        "candidates": inventory["candidate_count"],
        "image_size": inventory["image_size"],
    }, ensure_ascii=False))

    from image2ppt_planner import Settings, plan_page

    settings = Settings.from_env()
    print(json.dumps({"step": "planning", "model": settings.model, "key_present": bool(settings.api_key)}, ensure_ascii=False))
    result = asyncio.run(plan_page(settings, SOURCE, inventory, out_dir=PLAN_DIR))

    plan = result.plan
    page_area = inventory["image_size"][0] * inventory["image_size"][1]
    rasters = [o for o in plan["objects"] if o["kind"] == "raster_asset"]
    natives = [o for o in plan["objects"] if o["kind"] == "native_shape"]
    native_text = [t for t in plan["text_decisions"] if t["action"] == "native"]
    retained = [t for t in plan["text_decisions"] if t["action"] == "retained_in_asset"]
    sizes = sorted({t["font_size_px"] for t in native_text})
    geom = {}
    for o in natives:
        geom[o.get("geometry")] = geom.get(o.get("geometry"), 0) + 1

    report = {
        "step": "validated",
        "attempts": result.attempts,
        "elapsed_s": round(result.elapsed_s, 1),
        "usage": result.usage,
        "objects": len(plan["objects"]),
        "native_shapes": len(natives),
        "raster_assets": len(rasters),
        "raster_area_fraction_total": round(sum(o["bbox"][2] * o["bbox"][3] for o in rasters) / page_area, 4),
        "max_single_raster_fraction": round(max((o["bbox"][2] * o["bbox"][3] for o in rasters), default=0) / page_area, 4),
        "geometry": geom,
        "texts_total": len(plan["text_decisions"]),
        "texts_native": len(native_text),
        "texts_retained": len(retained),
        "distinct_font_sizes": len(sizes),
        "font_sizes": sizes,
        "fonts_used": sorted({t.get("font_family") for t in native_text}),
        "errors": result.errors,
        "artifacts": result.artifacts,
    }
    Path("/data/pilot/report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
