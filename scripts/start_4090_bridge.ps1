param(
    [string]$RemoteHost = '10.36.154.54',
    [string]$BindAddress = '10.36.154.44',
    [string]$RemoteUser = 'cmcc',
    [string]$IdentityFile = (Join-Path $env:USERPROFILE '.ssh\aitoy_tts_4090_ed25519'),
    [string]$AgentProvider = 'opencode_go',
    [string]$AgentModel = 'qwen3.8-max',
    [int]$LocalTunnelPort = 18772,
    [switch]$BootRemoteIfStopped
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ProfilePath = Join-Path $RepoRoot '.runtime\opencode-client.dpapi'
if (-not (Test-Path $ProfilePath)) {
    throw "Missing local DPAPI profile: $ProfilePath"
}

Write-Host "[1/4] Checking wired Ethernet SSH to $RemoteUser@$RemoteHost (from $BindAddress)..."
$remoteCheck = @'
import urllib.request
for name, url in [('api', 'http://127.0.0.1:8772/'), ('opencode', 'http://127.0.0.1:14100/global/health'), ('layerd', 'http://127.0.0.1:18847/health')]:
    try:
        code = urllib.request.urlopen(url, timeout=2).status
    except Exception:
        code = 0
    print(f'{name}={code}')
'@
$statusOut = $remoteCheck | ssh -o BatchMode=yes -b $BindAddress -i $IdentityFile "$RemoteUser@$RemoteHost" "F:\i2oc\venv\Scripts\python.exe -"
Write-Host ($statusOut -join ' | ')

if (($statusOut -match 'api=0') -and $BootRemoteIfStopped) {
    Write-Host "[2/4] Remote 8772 is stopped (e.g. after 4090 reboot). Prompting hidden model key for in-memory bootstrap ($AgentProvider / $AgentModel)..."
    $secKey = Read-Host -Prompt "Enter $AgentProvider API key (hidden, never saved to disk)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secKey)
    try {
        $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        $env:PYTHONPATH = Join-Path $RepoRoot 'services\product'
        $env:_I2P_TMP_AGENT_KEY = $plainKey
        $bootScript = @"
import json, os, subprocess
from product.runtime.client import load_profile
prof = load_profile(r'$ProfilePath')
payload = {
    'business_token': prof['business_token'],
    'office_token': prof['office_token'],
    'agent_provider': '$AgentProvider',
}
if '$AgentProvider' == 'opencode_go':
    payload['agent_api_key'] = os.environ['_I2P_TMP_AGENT_KEY']
    payload['agent_model'] = '$AgentModel'
else:
    payload['moma'] = os.environ['_I2P_TMP_AGENT_KEY']
res = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-b', '$BindAddress', '-i', '$IdentityFile', '$RemoteUser@$RemoteHost',
     r'F:\i2oc\venv\Scripts\python.exe', r'F:\i2oc\app\services\product\product\runtime\windows_host.py', '--root', r'F:\i2oc'],
    input=json.dumps(payload), text=True, capture_output=True, check=True
)
print(res.stdout.strip())
"@
        $bootScript | python -
    } finally {
        Remove-Item Env:\_I2P_TMP_AGENT_KEY -ErrorAction SilentlyContinue
        if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    }
    Start-Sleep -Seconds 3
} elseif ($statusOut -match 'api=0') {
    Write-Warning "Remote 8772 API is not running on 4090. Re-run with -BootRemoteIfStopped to securely inject model key and start 8772 + OpenCode 14100 ($AgentModel)."
}

Write-Host "[3/4] Ensuring local SSH tunnel 127.0.0.1:$LocalTunnelPort -> ${RemoteHost}:8772..."
$existingTunnel = Get-NetTCPConnection -LocalPort $LocalTunnelPort -State Listen -ErrorAction SilentlyContinue
if (-not $existingTunnel) {
    $sshProc = Start-Process -FilePath 'ssh.exe' -ArgumentList @(
        '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
        '-b', $BindAddress, '-i', $IdentityFile,
        '-L', "${LocalTunnelPort}:127.0.0.1:8772", "$RemoteUser@$RemoteHost"
    ) -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    Write-Host "Started SSH tunnel PID=$($sshProc.Id)"
} else {
    Write-Host "SSH tunnel already listening on 127.0.0.1:$LocalTunnelPort"
}

Write-Host "[4/4] Ensuring local PowerPoint office_bridge worker..."
$workerRunning = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'product\.runtime\.office_bridge' }
$OfficeRoot = Join-Path $env:TEMP 'i2office'
if (-not $workerRunning) {
    $workerLauncher = @"
import os, subprocess, sys
sys.path.insert(0, r'$RepoRoot\services\product')
from product.runtime.client import load_profile
prof = load_profile(r'$ProfilePath')
env = os.environ.copy()
env['PYTHONPATH'] = r'$RepoRoot\services\product'
env['IMAGE2PPT_OFFICE_WORKER_TOKEN'] = prof['office_token']
p = subprocess.Popen([sys.executable, '-m', 'product.runtime.office_bridge', '--url', 'http://127.0.0.1:$LocalTunnelPort', '--root', r'$OfficeRoot'], env=env, creationflags=subprocess.CREATE_NO_WINDOW)
print('Started office_bridge PID=' + str(p.pid))
"@
    $workerLauncher | python -
} else {
    Write-Host "office_bridge worker already running."
}
Write-Host "Bridge check complete."
