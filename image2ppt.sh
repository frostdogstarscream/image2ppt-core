#!/usr/bin/env bash
# Bash entry point for the image2ppt toolchain.
#
# The toolchain is a Windows PowerShell script, but Claude Code's shell is bash,
# where `./image2ppt.ps1 <cmd>` is a syntax error. This shim forwards every
# argument to image2ppt.ps1 so the same command text works in both shells.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v powershell.exe >/dev/null 2>&1; then
  ps=powershell.exe
elif command -v pwsh >/dev/null 2>&1; then
  ps=pwsh
else
  echo "image2ppt: 找不到 PowerShell（powershell.exe / pwsh）；本工具链目前需要 Windows PowerShell。" >&2
  exit 2
fi

exec "$ps" -NoProfile -ExecutionPolicy Bypass -File "$here/image2ppt.ps1" "$@"
