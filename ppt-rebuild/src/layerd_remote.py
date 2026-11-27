"""Call the LayerD HTTP API on a loopback service, optionally through SSH."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib import request as urlrequest



def _read_token(token_file):
    token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("LayerD token file is empty")
    return token


def _http(url, path, token, data=None, content_type=None, timeout=60):
    headers = {"Authorization": "Bearer " + token}
    if content_type:
        headers["Content-Type"] = content_type
    req = urlrequest.Request(url.rstrip("/") + path, data=data, headers=headers)
    with urlrequest.urlopen(req, timeout=timeout) as response:
        return response.read()


def _ssh_http(ssh_target, port, path, token, data=None, content_type=None, timeout=60):
    """POST/GET the Windows loopback LayerD API through OpenSSH. Token stays in stdin."""
    import base64

    from ssh_windows import DEFAULT_HOST, ssh_python

    payload = {
        "method": "POST" if data is not None else "GET",
        "path": path,
        "port": int(port),
        "token": token,
        "timeout": int(timeout),
        "content_type": content_type,
        "body": base64.b64encode(data).decode("ascii") if data is not None else None,
    }
    compact = (
        "import sys,json,base64,urllib.request;"
        "p=json.load(sys.stdin);"
        "h={'Authorization':'Bearer '+p['token']};"
        "ct=p.get('content_type');"
        "h.update({'Content-Type':ct} if ct else {});"
        "body=base64.b64decode(p['body']) if p.get('body') is not None else None;"
        "r=urllib.request.Request('http://127.0.0.1:'+str(p['port'])+p['path'],data=body,headers=h,method=p['method']);"
        "sys.stdout.buffer.write(urllib.request.urlopen(r,timeout=p['timeout']).read())"
    )
    target = ssh_target or DEFAULT_HOST
    if "@" not in target:
        # Bare host from the URL: keep the configured Windows account.
        user = DEFAULT_HOST.split("@", 1)[0] if "@" in DEFAULT_HOST else "CMCC"
        target = f"{user}@{target}"
    return ssh_python(json.dumps(payload).encode("utf-8"), compact, host=target, timeout=timeout)


def _request(endpoint, path, token, data=None, content_type=None, timeout=60):
    if endpoint.startswith("ssh://"):
        parsed = urlsplit(endpoint)
        if not parsed.hostname or parsed.port is None:
            raise ValueError("ssh LayerD URL needs host and port, e.g. ssh://CMCC@192.168.1.103:18840")
        host = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname
        return _ssh_http(host, parsed.port, path, token, data=data,
                         content_type=content_type, timeout=timeout)
    return _http(endpoint, path, token, data=data, content_type=content_type, timeout=timeout)


def decompose(image, output, *, endpoint="ssh://192.168.1.103:18840", token_file=None,
              max_iterations=3, timeout=1200):
    """Upload ``image``, wait for LayerD, download artifacts into a new ``output`` directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new LayerD output directory")
    token = _read_token(token_file)
    capabilities = json.loads(_request(endpoint, "/v1/capabilities", token, timeout=60))
    lower, upper = capabilities["max_iterations"]
    if not lower <= max_iterations <= upper:
        raise ValueError(f"Service supports max_iterations {lower}..{upper}; requested {max_iterations}")
    source = Path(image)
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="max_iterations"\r\n\r\n{max_iterations}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input{source.suffix}"\r\n'
            f'Content-Type: {mimetypes.guess_type(source.name)[0] or "application/octet-stream"}\r\n\r\n').encode()
    body += source.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    status = json.loads(_request(endpoint, "/v1/decompose", token, data=body,
                                 content_type="multipart/form-data; boundary=" + boundary, timeout=120))
    job_id = status["id"]
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status = json.loads(_request(endpoint, "/v1/jobs/" + job_id, token, timeout=60))
        if status["status"] != last:
            print(json.dumps({"layerd": status["status"], "id": job_id}), flush=True)
            last = status["status"]
        if last in ("FAILED", "INTERRUPTED"):
            raise RuntimeError("Job " + job_id + " " + last + ": " + str(status.get("error_type")))
        if last == "SUCCEEDED":
            break
        time.sleep(2)
    else:
        raise TimeoutError("LayerD job still running: " + job_id)
    output.mkdir(parents=True)
    for filename in status["artifacts"]:
        if Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("Unsafe artifact filename")
        (output / filename).write_bytes(_request(endpoint, f"/v1/jobs/{job_id}/artifacts/{filename}",
                                                 token, timeout=180))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for layer in manifest["layers"]:
        if hashlib.sha256((output / layer["file"]).read_bytes()).hexdigest() != layer["sha256"]:
            raise ValueError("Artifact hash mismatch")
    return output, manifest
