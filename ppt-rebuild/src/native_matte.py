"""Optional object-sized mask providers. Service locations and tokens belong to the deployment."""

import base64
import hashlib
import io
import json
import os
import time
from subprocess import SubprocessError

import numpy as np
from PIL import Image
from layerd_remote import _request, _read_token
from ssh_windows import ssh_python


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def sam_mask(image, label, endpoint, token_file, timeout):
    import httpx

    data = png_bytes(image)
    token = _read_token(token_file)
    w, h = image.size
    plan = dict(
        source_sha256=hashlib.sha256(data).hexdigest(),
        image_size=[w, h],
        objects=[dict(id="graphic", bbox=[0, 0, w, h], prompt_en=label[:300] or "icon")],
    )
    # Let httpx encode the multipart form for both the loopback and SSH transports.
    request = httpx.Request(
        "POST",
        "https://localhost",
        files={"file": ("crop.png", data, "image/png")},
        data={"plan": json.dumps(plan), "version": "sam3"},
    )
    result = json.loads(
        _request(
            endpoint,
            "/v1/jobs",
            token,
            data=request.read(),
            content_type=request.headers["content-type"],
            timeout=min(timeout, 60),
        )
    )
    job = "/v1/jobs/" + result["id"]
    name = result["mapping"]["graphic"] + ".npz"
    deadline = time.monotonic() + timeout
    while result["status"] not in ("SUCCEEDED", "FAILED", "INTERRUPTED"):
        if time.monotonic() >= deadline:
            raise TimeoutError("SAM object timed out")
        time.sleep(1)
        result = json.loads(_request(endpoint, job, token, timeout=min(timeout, 30)))
    if result["status"] != "SUCCEEDED":
        raise ValueError("SAM did not return a mask")
    data = _request(endpoint, job + "/artifacts/" + name, token, timeout=min(timeout, 60))
    if hashlib.sha256(data).hexdigest() != result["artifacts"][name]:
        raise ValueError("SAM artifact hash mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as pack:
        if not len(pack["scores"]):
            raise ValueError("SAM found no candidates")
        return np.asarray(pack["masks"][np.argmax(pack["scores"])], dtype=np.uint8) * 255


def rmbg_mask(image, host, worker, model, timeout):
    # Model/worker paths are deployment configuration. Only PNG bytes cross the SSH boundary.
    code = """
import base64,json,subprocess,sys,tempfile
from pathlib import Path
request=json.load(sys.stdin)
with tempfile.TemporaryDirectory(prefix='native-matte-') as tmp:
    crop=Path(tmp)/'crop.png'
    out=Path(tmp)/'alpha.png'
    crop.write_bytes(base64.b64decode(request['png'],validate=True))
    subprocess.run([sys.executable,request['worker'],request['model'],str(crop),str(out)],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=request['timeout'])
    sys.stdout.buffer.write(out.read_bytes())
"""
    payload = dict(
        png=base64.b64encode(png_bytes(image)).decode(), worker=worker, model=model, timeout=timeout
    )
    result = ssh_python(json.dumps(payload).encode(), code, host=host, timeout=timeout)
    with Image.open(io.BytesIO(result)) as mask:
        return np.asarray(mask.convert("L"))


def optional_matte(crop, obj, timeout, attempts):
    host = os.getenv("IMAGE2PPT_NATIVE_RMBG_HOST")
    sam = os.getenv("IMAGE2PPT_NATIVE_SAM_URL")
    providers = []
    if host:
        providers.append(
            (
                "RMBG",
                lambda: rmbg_mask(
                    crop,
                    host,
                    os.environ["IMAGE2PPT_NATIVE_RMBG_WORKER"],
                    os.environ["IMAGE2PPT_NATIVE_RMBG_MODEL"],
                    timeout,
                ),
            )
        )
    if sam:
        providers.append(
            (
                "SAM",
                lambda: sam_mask(
                    crop, obj["label"], sam, os.environ["IMAGE2PPT_NATIVE_SAM_TOKEN_FILE"], timeout
                ),
            )
        )
    for name, call in providers:
        row = {"id": obj["id"], "model": name, "accepted": False}
        attempts.append(row)
        try:
            alpha = call()
            if alpha.shape != (crop.height, crop.width) or not 0.015 < (alpha > 128).mean() < 0.95:
                row["reason"] = "no isolated foreground"
                continue
            matte = crop.convert("RGBA")
            matte.putalpha(Image.fromarray(alpha))
            row["accepted"] = True
            return matte
        except (OSError, ValueError, RuntimeError, KeyError, SubprocessError) as error:
            row["reason"] = type(error).__name__
    return None
