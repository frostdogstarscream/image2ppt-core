"""Run a stdin/stdout Python relay on a Windows host reached through OpenSSH."""
from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path


DEFAULT_HOST = os.environ.get("IMAGE2PPT_SSH_HOST", "CMCC@192.168.1.103")
DEFAULT_IDENTITY = os.environ.get(
    "IMAGE2PPT_SSH_IDENTITY",
    str(Path.home() / ".ssh" / "id_ed25519_desktop_g68g4qr"),
)
# LayerD/LaMa listen on the Windows loopback; use the Windows interpreter there.
DEFAULT_WIN_PYTHON = os.environ.get(
    "IMAGE2PPT_SSH_WIN_PYTHON",
    r"F:\Anaconda\python.exe",
)


def ssh_python(payload: bytes, code: str, *, host: str | None = None, timeout: int = 120) -> bytes:
    """Execute ``code`` with Windows Python on ``host``, feeding ``payload`` on stdin."""
    host = host or DEFAULT_HOST
    identity = DEFAULT_IDENTITY
    win_python = DEFAULT_WIN_PYTHON
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    # One remote argv for OpenSSH: Windows python reads the relay from -c,
    # stdin carries only the HTTP/JSON payload. Avoids WSL/bash quote splits.
    remote = (
        f'"{win_python}" -c '
        f'"exec(__import__(\'base64\').b64decode(\'{encoded}\').decode())"'
    )
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        "-o", "IdentitiesOnly=yes",
        "-i", identity,
        host,
        remote,
    ]
    last = None
    for attempt in range(4):
        last = subprocess.run(cmd, input=payload, capture_output=True, timeout=timeout + 30)
        if last.returncode != 255 or attempt == 3:
            break
        time.sleep(10 * (attempt + 1))
    if last.returncode:
        raise RuntimeError(
            "SSH Windows relay failed: "
            + last.stderr.decode("utf-8", errors="replace")[-1800:]
        )
    return last.stdout
