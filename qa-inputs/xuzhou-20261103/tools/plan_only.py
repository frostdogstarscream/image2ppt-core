"""Plan one page against an inventory produced by the real service (real OCR ids)."""
import asyncio
import json
import sys
from pathlib import Path

from image2ppt_planner import Settings, plan_page


def main() -> int:
    inv_path, image, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    inventory = json.loads(Path(inv_path).read_text(encoding="utf-8"))
    settings = Settings.from_env()
    print(json.dumps({"model": settings.model, "thinking": not settings.disable_thinking,
                      "texts": len(inventory["texts"]), "candidates": len(inventory.get("candidates", []))},
                     ensure_ascii=False), flush=True)

    result = asyncio.run(plan_page(settings, image, inventory, out_dir=out_dir))
    plan = result.plan
    page = inventory["image_size"][0] * inventory["image_size"][1]
    rasters = [o for o in plan["objects"] if o["kind"] == "raster_asset"]
    natives = [o for o in plan["objects"] if o["kind"] == "native_shape"]
    native_text = [t for t in plan["text_decisions"] if t["action"] == "native"]
    retained = [t for t in plan["text_decisions"] if t["action"] == "retained_in_asset"]
    report = {
        "attempts": result.attempts,
        "elapsed_s": round(result.elapsed_s, 1),
        "usage": result.usage,
        "objects": len(plan["objects"]),
        "native_shapes": len(natives),
        "raster_assets": len(rasters),
        "raster_area_fraction": round(sum(o["bbox"][2] * o["bbox"][3] for o in rasters) / page, 4),
        "max_raster_fraction": round(max((o["bbox"][2] * o["bbox"][3] for o in rasters), default=0) / page, 4),
        "texts_total": len(plan["text_decisions"]),
        "texts_native": len(native_text),
        "texts_retained": len(retained),
        "distinct_font_sizes": len({t["font_size_px"] for t in native_text}),
        "fonts": sorted({t.get("font_family") for t in native_text}),
        "errors": result.errors,
    }
    Path("/data/pilot/report-plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.plan else 1


if __name__ == "__main__":
    sys.exit(main())
