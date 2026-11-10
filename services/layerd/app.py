"""Isolated, upload-only LayerD service. Raster layers are not native PPT text."""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import queue
import re
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(os.environ.get("LAYERD_DATA", "/data"))
MODELS = Path(os.environ.get("LAYERD_MODELS", "/models"))
TOKEN_FILE = Path(os.environ.get("LAYERD_TOKEN_FILE", "/run/secrets/client-token"))
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 2_500_000
MAX_ITERATIONS = int(os.environ.get("LAYERD_MAX_ITERATIONS", "5"))
if not 3 <= MAX_ITERATIONS <= 12:
    raise ValueError("LAYERD_MAX_ITERATIONS must be between 3 and 12")
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
JOBS: queue.Queue[str | None] = queue.Queue(maxsize=8)
STATE = {"ready": False, "loading": True, "error_type": None, "current_job": None}
ADMISSION=threading.Lock()
STOP = threading.Event()
THREAD: threading.Thread | None = None


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(path)


def authenticate(authorization: str | None = Header(default=None)) -> None:
    expected = TOKEN_FILE.read_text().strip()
    if not expected or not hmac.compare_digest(authorization or "", "Bearer " + expected):
        raise HTTPException(401, "Bearer token required")


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "Unknown job")
    path = ROOT / "jobs" / job_id
    if not (path / "status.json").is_file():
        raise HTTPException(404, "Unknown job")
    return path


def render_contact(layers: list[Image.Image], output: Path) -> None:
    import numpy as np
    thumbs = []
    for layer in layers:
        thumb = ImageOps.contain(layer, (640, 360))
        yy, xx = np.indices((thumb.height, thumb.width))
        pattern = np.where(((yy // 12 + xx // 12) % 2)[..., None], 220, 250)
        rgb = np.broadcast_to(pattern, (thumb.height, thumb.width, 3)).astype("uint8")
        canvas = Image.fromarray(rgb).convert("RGBA")
        canvas.alpha_composite(thumb)
        thumbs.append(canvas.convert("RGB"))
    sheet = Image.new("RGB", (1280, math.ceil(len(thumbs) / 2) * 380), "#dddddd")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 2) * 640, (i // 2) * 380 + 10))
    sheet.save(output)


class RefinementBudgetExceeded(RuntimeError):
    def __init__(self, components: int, estimated_bytes: int):
        super().__init__("Optional color refinement exceeds component memory budget")
        self.components = components
        self.estimated_bytes = estimated_bytes


def budgeted_components(mask):
    """Check every nested color mask before the upstream allocates its list."""
    import cv2
    import numpy as np
    assert mask.dtype == bool
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    count -= 1
    estimated = count * mask.size * 8
    if count > 128 or estimated > 1024**3:
        raise RefinementBudgetExceeded(count, estimated)
    return [labels == label for label in range(1, count + 1)]


def bounded_decompose(model, image: Image.Image, max_iterations: int):
    """Bound optional full-canvas component refinement in the pinned upstream.

    The upstream makes several H*W masks per connected component. Dense fine
    artwork can exhaust host RAM even when CUDA memory is comfortably available.
    Only optional color refinement is skipped; matting, LaMa and unblending remain.
    """
    import cv2
    import numpy as np
    current = image.convert("RGB")
    foregrounds = []
    steps = []
    original_matting = model.matting_model
    original_flags = (model.fg_refine, model.bg_refine)
    try:
        for iteration in range(max_iterations):
            alpha = original_matting(current)
            components, _ = cv2.connectedComponents((alpha > 0.005).astype("uint8"))
            components -= 1
            estimated_masks = components * image.width * image.height * 8
            refine = components <= 128 and estimated_masks <= 1024**3
            model.fg_refine = original_flags[0] and refine
            model.bg_refine = original_flags[1] and refine
            # Reuse the already measured alpha instead of running the model twice.
            model.matting_model = lambda _: alpha
            reason = "within_budget" if refine else "component_memory_budget"
            nested_budget = None
            print(f"LayerD iteration={iteration} alpha_components={components} refinement={refine}", flush=True)
            try:
                foreground, background = model._decompose_step(current)
            except RefinementBudgetExceeded as error:
                # Nested per-color regions can be numerous even with one alpha CC.
                # Reuse alpha, retry this iteration without optional refinement.
                model.fg_refine = model.bg_refine = False
                refine = False
                reason = "nested_component_memory_budget"
                nested_budget = {"components": error.components,
                                 "estimated_masks_mib": round(error.estimated_bytes / 2**20, 1)}
                print(f"LayerD iteration={iteration} refinement_retry={nested_budget}", flush=True)
                foreground, background = model._decompose_step(current)
            steps.append({"iteration": iteration, "mask_components": components,
                          "estimated_component_masks_mib": round(estimated_masks / 2**20, 1),
                          "color_refinement": refine,
                          "reason": reason, "nested_budget": nested_budget})
            if foreground is None:
                break
            foregrounds.append(foreground)
            current = background
    finally:
        model.matting_model = original_matting
        model.fg_refine, model.bg_refine = original_flags
    return [current.convert("RGBA"), *reversed(foregrounds)], steps


def process(model, path: Path) -> None:
    import numpy as np
    import torch
    request = json.loads((path / "request.json").read_text())
    start = time.monotonic()
    write_json(path / "status.json", {"id": path.name, "status": "RUNNING", "started": time.time()})
    with Image.open(path / "source.png") as im:
        source = im.convert("RGBA")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        layers, refinement_steps = bounded_decompose(model, source, request["max_iterations"])
    # The pinned implementation returns background followed by reversed peel order:
    # bottom to top. Its prose documentation contains a conflicting order description.
    composite = Image.new("RGBA", source.size, (0, 0, 0, 0))
    records = []
    for i, layer in enumerate(layers):
        layer = layer.convert("RGBA")
        if layer.size != source.size:
            raise ValueError("Layer size differs from input")
        filename = f"layer-{i:02d}.png"
        layer.save(path / filename)
        composite.alpha_composite(layer)
        alpha = np.asarray(layer.getchannel("A"))
        records.append({"file": filename, "z": i, "role": "background" if i == 0 else "foreground",
                        "sha256": digest((path / filename).read_bytes()), "size": list(layer.size),
                        "alpha_bbox": layer.getchannel("A").getbbox(),
                        "alpha_coverage": float((alpha > 0).mean()),
                        "partial_alpha_fraction": float(((alpha > 0) & (alpha < 255)).mean())})
    composite.save(path / "composite.png")
    render_contact(layers, path / "contact.png")
    original = np.asarray(source.convert("RGB"), dtype=np.float32)
    rendered = np.asarray(composite.convert("RGB"), dtype=np.float32)
    delta = original - rendered
    mse = float((delta * delta).mean())
    manifest = {"schema": "LayerDOutput/1", "id": path.name, "source_sha256": request["source_sha256"],
                "image_size": list(source.size), "layer_order": "back_to_front", "layers": records,
                "max_iterations": request["max_iterations"], "matting_process_size": int(os.getenv("LAYERD_PROCESS_SIZE", "1024")),
                "elapsed_s": round(time.monotonic() - start, 3),
                "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "composite_mae_255": round(float(np.abs(delta).mean()), 5),
                "composite_psnr_db": round(10 * math.log10(255**2 / mse), 5) if mse else None,
                "upstream_revision": os.getenv("LAYERD_REVISION"), "native_text": False,
                "model_binding":STATE.get('model_binding'),"pixel_method":"bounded_components_v2-r10/1",
                "refinement_policy": "bounded_components_v2", "refinement_steps": refinement_steps,
                "review_status": "REVIEW_REQUIRED", "human_acceptance": False}
    write_json(path / "manifest.json", manifest)
    files = ["source.png", "request.json", "manifest.json", "composite.png", "contact.png"] + [r["file"] for r in records]
    write_json(path / "status.json", {"id": path.name, "status": "SUCCEEDED", "finished": time.time(),
                                      "elapsed_s": manifest["elapsed_s"], "layer_count": len(layers), "artifacts": files})


def worker() -> None:
    try:
        import torch
        from layerd import LayerD
        from layerd.models import helpers
        helpers.divide_mask_to_connected_components = budgeted_components
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(0.70)
        size = int(os.getenv("LAYERD_PROCESS_SIZE", "1024"))
        model_files=[p for p in MODELS.rglob('*') if p.is_file() and p.suffix.lower() in ('.pt','.pth','.safetensors','.bin','.json','.py')]
        if not model_files:raise RuntimeError('offline model files missing')
        model_hashes={}
        for p in sorted(model_files):
            with p.open('rb') as f:model_hashes[str(p.relative_to(MODELS))]=hashlib.file_digest(f,'sha256').hexdigest()
        STATE['model_binding']=digest(json.dumps(model_hashes,sort_keys=True).encode())
        # Shared with private SAM on this physical GPU, across services/processes.
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from resource_lease import lease
        STATE.update(ready=True, loading=False)
        while not STOP.is_set():
            try:
                job_id = JOBS.get(timeout=0.5)
            except queue.Empty:
                continue
            if job_id is None:
                break
            STATE["current_job"] = job_id
            path = ROOT / "jobs" / job_id
            try:
                with lease('gpu-'+os.environ.get('IMAGE2PPT_GPU_ID','0')):
                    model=None
                    try:
                        model=LayerD(matting_hf_card=str(MODELS / "birefnet"), matting_process_size=(size,size),device="cuda")
                        process(model,path)
                    finally:
                        del model
                        import gc
                        gc.collect();torch.cuda.empty_cache()
            except Exception as error:
                traceback.print_exc()
                write_json(path / "status.json", {"id": job_id, "status": "FAILED", "error_type": type(error).__name__, "finished": time.time()})
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            finally:
                STATE["current_job"] = None
                JOBS.task_done()
    except Exception as error:
        traceback.print_exc()
        STATE.update(ready=False, loading=False, error_type=type(error).__name__)
    finally:
        STATE["ready"] = False


@asynccontextmanager
async def lifespan(app):
    global THREAD
    if len(TOKEN_FILE.read_text().strip()) < 32:
        raise RuntimeError("Service token is missing or too short")
    (ROOT / "jobs").mkdir(parents=True, exist_ok=True)
    for status_path in (ROOT / "jobs").glob("*/status.json"):
        record = json.loads(status_path.read_text())
        if record.get("status") in ("QUEUED", "RUNNING"):
            write_json(status_path, {"id": record["id"], "status": "INTERRUPTED", "reason": "ServiceRestart"})
    THREAD = threading.Thread(target=worker, name="layerd-worker", daemon=True)
    THREAD.start()
    yield
    STOP.set()
    THREAD.join(timeout=10)


app = FastAPI(title="image2ppt-layerd", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"alive": True, "ready": STATE["ready"] and THREAD is not None and THREAD.is_alive(),
            "loading": STATE["loading"], "worker_alive": THREAD is not None and THREAD.is_alive(),
            "error_type": STATE["error_type"]}


@app.get("/ready")
def ready():
    state = health()
    return JSONResponse(state, status_code=200 if state["ready"] else 503)


@app.get("/v1/capabilities", dependencies=[Depends(authenticate)])
def capabilities():
    return {"service": "image2ppt-layerd", "upstream_revision": os.getenv("LAYERD_REVISION"),
            "output": "RGBA PNG layers", "layer_order": "back_to_front", "native_text": False,
            "pptx_compilation": False, "device": "cuda", "network_inference": False,
            "max_iterations": [1, MAX_ITERATIONS], "max_pixels": MAX_PIXELS, "max_bytes": MAX_BYTES,
            "matting_process_size": int(os.getenv("LAYERD_PROCESS_SIZE", "1024")),
            "model_binding":STATE.get('model_binding'),"pixel_method":"bounded_components_v2-r10/1",
            "queue_size": JOBS.qsize(), "current_job": STATE["current_job"], **health()}


@app.post("/v1/decompose", status_code=202, dependencies=[Depends(authenticate)])
async def decompose(file: UploadFile = File(...), max_iterations: int = Form(default=3, ge=1, le=MAX_ITERATIONS)):
    if not health()["ready"]:
        raise HTTPException(503, "Model worker is not ready")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 20 MiB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ("PNG", "JPEG", "WEBP") or getattr(im, "n_frames", 1) != 1:
                raise HTTPException(400, "Use a single-frame PNG, JPEG or WebP")
            if im.width * im.height > MAX_PIXELS:
                raise HTTPException(413, "Image exceeds 2.5 million pixels")
            image = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Invalid image")
    binding={'source':digest(data),'iterations':max_iterations,'process_size':os.getenv('LAYERD_PROCESS_SIZE','1024'),
             'upstream':os.getenv('LAYERD_REVISION'),'models':STATE.get('model_binding'),'pixel_method':'bounded_components_v2-r10/1'}
    cache_key=digest(json.dumps(binding,sort_keys=True).encode())
    # A missing model binding is never assumed to be a reusable real result.
    job_id=cache_key[:32] if binding['models'] else uuid.uuid4().hex
    with ADMISSION:
        path=ROOT/'jobs'/job_id
        if path.exists():
            prior=json.loads((path/'request.json').read_text(encoding='utf-8'))
            if prior.get('cache_key')!=cache_key:raise HTTPException(409,'LayerD identity collision')
            return json.loads((path/'status.json').read_text(encoding='utf-8'))
        if JOBS.full():raise HTTPException(429,'Queue is full')
        path.mkdir();(path/'original.bin').write_bytes(data);image.save(path/'source.png')
        write_json(path/'request.json',{'source_sha256':digest(data),'normalized_sha256':digest((path/'source.png').read_bytes()),
            'max_iterations':max_iterations,'created':time.time(),'cache_key':cache_key,'model_binding':binding['models']})
        write_json(path/'status.json',{'id':job_id,'status':'QUEUED','created':time.time()})
        try:JOBS.put_nowait(job_id)
        except queue.Full:
            write_json(path/'status.json',{'id':job_id,'status':'FAILED','error_type':'QueueFull'})
            raise HTTPException(429,'Queue is full')
    return {"id": job_id, "status": "QUEUED"}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(authenticate)])
def status(job_id: str):
    return json.loads((job_path(job_id) / "status.json").read_text())


@app.get("/v1/jobs/{job_id}/artifacts/{filename}", dependencies=[Depends(authenticate)])
def artifact(job_id: str, filename: str):
    path = job_path(job_id)
    record = json.loads((path / "status.json").read_text())
    if filename not in record.get("artifacts", []):
        raise HTTPException(404, "Unknown artifact")
    return FileResponse(path / filename)
