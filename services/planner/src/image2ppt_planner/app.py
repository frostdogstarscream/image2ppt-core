"""HTTP service and CLI for the planner."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Settings
from .planner import available_fonts, load_inventory, plan_page

app = FastAPI(title="image2ppt-planner", version="0.1.0")


def _validator_ready() -> tuple[bool, str]:
    try:
        from image2ppt_vision.supervised_elements import validate_element_plan  # noqa: F401

        return True, "image2ppt_vision.supervised_elements.validate_element_plan"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


class PlanRequest(BaseModel):
    image_path: str
    inventory_path: str | None = None
    inventory: dict | None = None
    out_dir: str | None = None
    fonts: list[str] | None = None


@app.get("/health")
def health() -> dict:
    ready, detail = _validator_ready()
    return {"status": "alive", "validator": ready, "validator_detail": detail}


@app.get("/v1/capabilities")
def capabilities() -> dict:
    ready, detail = _validator_ready()
    return {
        "version": "0.1.0",
        "validates_with_authoritative_schema": ready,
        "validator_detail": detail,
        "fonts": available_fonts(),
        "settings": Settings.from_env().describe(),
        "outputs": "ElementPlan/2",
        "calls_model": True,
    }


@app.post("/v1/plan")
async def plan(request: PlanRequest) -> dict:
    settings = Settings.from_env()
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="没有配置模型凭据（PLANNER_API_KEY / MOMA_API_KEY / VISION_LLM_API_KEY）")
    if request.inventory is None and not request.inventory_path:
        raise HTTPException(status_code=422, detail="需要 inventory_path 或 inventory 之一")
    inventory = request.inventory if request.inventory is not None else load_inventory(request.inventory_path)
    try:
        result = await plan_page(
            settings,
            request.image_path,
            inventory,
            fonts=request.fonts,
            out_dir=request.out_dir,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"plan": result.plan, "record": result.to_record()}


def _cmd_plan(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    inventory = load_inventory(args.inventory)
    result = asyncio.run(
        plan_page(settings, args.image, inventory, out_dir=args.out_dir, fonts=args.fonts)
    )
    print(json.dumps(result.to_record(), ensure_ascii=False, indent=2))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    ready, detail = _validator_ready()
    header = {"validator": ready, "validator_detail": detail, "settings": settings.describe()}
    if not settings.api_key:
        print(json.dumps(header, ensure_ascii=False, indent=2))
        print("没有凭据，未发起调用。", file=sys.stderr)
        return 2

    async def _run() -> int:
        from .client import MomaClient
        from .probe import CANDIDATES, probe_models, summarise

        if args.models:
            models = args.models
        else:
            client = MomaClient(settings)
            try:
                await client.chat([{"role": "user", "content": "只回复两个字：收到"}])
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({**header, "ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
                return 1
            models = CANDIDATES

        image_url = None
        if args.image:
            from .planner import prepare_image

            image_url, note = prepare_image(args.image, settings)
            header["probe_image"] = note

        results = await probe_models(settings, models, image_url)
        print(json.dumps({**header, **summarise(results)}, ensure_ascii=False, indent=2))
        return 0 if any(r.text.get("ok") for r in results) else 1

    return asyncio.run(_run())


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image2ppt-planner", description="Image page -> validated ElementPlan/2")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="run one planning pass")
    p_plan.add_argument("--image", required=True)
    p_plan.add_argument("--inventory", required=True)
    p_plan.add_argument("--out-dir")
    p_plan.add_argument("--fonts", nargs="*")
    p_plan.set_defaults(func=_cmd_plan)

    p_probe = sub.add_parser("probe", help="check credentials, and measure which models accept text and images")
    p_probe.add_argument("--models", nargs="*", help="model ids to test; default is the built-in candidate list")
    p_probe.add_argument("--image", help="page image used for the vision check")
    p_probe.set_defaults(func=_cmd_probe)

    p_serve = sub.add_parser("serve", help="run the HTTP service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=18810)
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
