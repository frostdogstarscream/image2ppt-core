"""Big-LaMa inpainting service.

Removes text from a page image: the caller sends the page plus a mask, the model
reconstructs the masked pixels from the surrounding structure (a learned model,
not a blurred fill).

Endpoints
  GET  /health                      device / model info, no payload
  POST /inpaint                     multipart: image=<png>, mask=<png>  -> PNG
  POST /inpaint-json                {"image": b64, "mask": b64}         -> {"image": b64}

The mask convention is "white (>=128) = erase".
"""
import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL = None
DEVICE = "cuda"


def load_model(path):
    global MODEL, DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = torch.jit.load(path, map_location=DEVICE)
    MODEL.eval()
    if DEVICE == "cuda":
        MODEL.to(memory_format=torch.channels_last)
    # Warm up so the first real request does not pay the context cost.
    dummy = torch.zeros((1, 3, 64, 64), device=DEVICE)
    dmask = torch.zeros((1, 1, 64, 64), device=DEVICE)
    with torch.no_grad():
        MODEL(dummy, dmask)
    return DEVICE


def _to_tensor(image, mask):
    h, w = image.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    img = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    msk = torch.from_numpy((mask >= 128).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    msk = F.pad(msk, (0, pad_w, 0, pad_h), mode="reflect")
    return img.to(DEVICE), msk.to(DEVICE), (h, w)


@torch.no_grad()
def inpaint(image, mask):
    img, msk, (h, w) = _to_tensor(image, mask)
    result = MODEL(img, msk)
    if isinstance(result, dict):
        result = result.get("inpainted", result.get("output"))
    result = result[:, :, :h, :w]
    out = result.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (out * 255.0).round().astype(np.uint8)


def _png(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.array(im.convert("RGB"))


def _parse_multipart(body, boundary):
    parts = {}
    delimiter = b"--" + boundary
    for chunk in body.split(delimiter):
        if not chunk or chunk in (b"--\r\n", b"--"):
            continue
        head, _, payload = chunk.partition(b"\r\n\r\n")
        if not _:
            continue
        name = None
        for line in head.split(b"\r\n"):
            if b'name="' in line:
                name = line.split(b'name="')[1].split(b'"')[0].decode()
        if name:
            parts[name] = payload.rstrip(b"\r\n")
    return parts


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[lama] " + (fmt % args), flush=True)

    def _send(self, code, payload, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json")

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {
                "status": "alive",
                "device": DEVICE,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "model": "big-lama",
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        started = time.time()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        try:
            if self.path.startswith("/inpaint-json"):
                payload = json.loads(body.decode("utf-8"))
                image = _png(base64.b64decode(payload["image"]))
                mask = np.array(Image.open(io.BytesIO(base64.b64decode(payload["mask"]))).convert("L"))
            elif self.path.startswith("/inpaint"):
                boundary = self.headers["Content-Type"].split("boundary=")[-1].encode()
                parts = _parse_multipart(body, boundary)
                image = _png(parts["image"])
                mask = np.array(Image.open(io.BytesIO(parts["mask"])).convert("L"))
            else:
                self._send_json(404, {"error": "not found"})
                return
            if image.shape[:2] != mask.shape[:2]:
                self._send_json(400, {"error": "mask size mismatch"})
                return
            out = inpaint(image, mask)
            buffer = io.BytesIO()
            Image.fromarray(out).save(buffer, format="PNG")
            data = buffer.getvalue()
            if self.path.startswith("/inpaint-json"):
                self._send_json(200, {"image": base64.b64encode(data).decode(), "ms": round((time.time() - started) * 1000)})
            else:
                self._send(200, data, "image/png")
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18840)
    ap.add_argument("--model", default="/app/models/big-lama.pt")
    args = ap.parse_args()
    device = load_model(args.model)
    print(f"[lama] model loaded on {device}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[lama] listening on {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
