"""Upload an image, wait for LayerD, and verify downloaded artifact hashes."""
import argparse
import hashlib
import json
import mimetypes
import time
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18840")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iterations", type=int, default=3, choices=range(1, 13))
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory")
    token = Path(args.token_file).read_text().strip()

    def request(path, data=None, content_type=None):
        headers = {"Authorization": "Bearer " + token}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.read()

    capabilities = json.loads(request("/v1/capabilities"))
    lower, upper = capabilities["max_iterations"]
    if not lower <= args.max_iterations <= upper:
        raise ValueError(f"Service supports max_iterations {lower}..{upper}; requested {args.max_iterations}")
    source = Path(args.input)
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="max_iterations"\r\n\r\n{args.max_iterations}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input{source.suffix}"\r\n'
            f'Content-Type: {mimetypes.guess_type(source.name)[0] or "application/octet-stream"}\r\n\r\n').encode()
    body += source.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    receipt=output.with_suffix('.remote.json')
    binding={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'url':args.url.rstrip('/'),'max_iterations':args.max_iterations}
    if args.resume and receipt.exists():
        saved=json.loads(receipt.read_text(encoding='utf-8'))
        if any(saved.get(k)!=v for k,v in binding.items()):raise ValueError('LayerD resume request binding changed')
        status=json.loads(request('/v1/jobs/'+saved['id']))
    else:
        status=json.loads(request('/v1/decompose',body,'multipart/form-data; boundary='+boundary))
        receipt.parent.mkdir(parents=True,exist_ok=True)
        temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps({**binding,'id':status['id']}),encoding='utf-8');temporary.replace(receipt)
    job_id = status["id"]
    print(json.dumps(status), flush=True)
    deadline = time.monotonic() + args.timeout
    last = None
    while time.monotonic() < deadline:
        status = json.loads(request("/v1/jobs/" + job_id))
        if status["status"] != last:
            print(json.dumps(status), flush=True)
            last = status["status"]
        if last in ("FAILED", "INTERRUPTED"):
            raise RuntimeError("Job " + job_id + " " + last + ": " + str(status.get("error_type")))
        if last == "SUCCEEDED":
            break
        time.sleep(2)
    else:
        raise TimeoutError("Job is still recorded on the server: " + job_id)
    output.mkdir(parents=True,exist_ok=args.resume)
    for filename in status["artifacts"]:
        if Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("Unsafe artifact filename")
        (output / filename).write_bytes(request(f"/v1/jobs/{job_id}/artifacts/{filename}"))
    manifest = json.loads((output / "manifest.json").read_text())
    for layer in manifest["layers"]:
        if hashlib.sha256((output / layer["file"]).read_bytes()).hexdigest() != layer["sha256"]:
            raise ValueError("Artifact hash mismatch")
    print(json.dumps({"output": str(output.resolve()), "manifest": manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
