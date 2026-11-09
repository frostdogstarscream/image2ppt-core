param(
  [Parameter(Mandatory=$true)][string]$ServiceRoot,
  [Parameter(Mandatory=$true)][string]$ExperimentRoot,
  [string]$TaskName='Image2PPT-Private-SAM',
  [int]$Port=18843
)
$ErrorActionPreference='Stop'
$rootPath=[IO.Path]::GetFullPath($ServiceRoot)
$experimentPath=[IO.Path]::GetFullPath($ExperimentRoot)
if (!(Test-Path -LiteralPath (Join-Path $rootPath 'service.py'))) { throw 'Deploy service.py and single_frame.py first' }
$pythonPath=Join-Path $experimentPath 'venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $pythonPath)) { throw 'Existing isolated Python missing' }
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) { throw 'Task exists; use a new deployment task name' }
New-Item -ItemType Directory -Path (Join-Path $rootPath 'jobs') -Force | Out-Null
$tokenPath=Join-Path $rootPath 'client-token'
if (!(Test-Path -LiteralPath $tokenPath)) {
  $bytes=New-Object byte[] 48
  $rng=[Security.Cryptography.RandomNumberGenerator]::Create()
  try {$rng.GetBytes($bytes)} finally {$rng.Dispose()}
  [IO.File]::WriteAllText($tokenPath,[Convert]::ToBase64String($bytes))
}
$owner=[Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls.exe $tokenPath /inheritance:r /grant:r ($owner+':F') '*S-1-5-18:F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Token ACL failed' }
$config=@{
  jobs=(Join-Path $rootPath 'jobs');token_file=$tokenPath;python=$pythonPath;
  source=(Join-Path $experimentPath 'sam3-official');code_revision='f1647e8d0c1d280991f4d1b7d974c2704537fe66';
  models=@{sam3=@{checkpoint=(Join-Path $experimentPath 'weights/sam3.pt');sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'}}
}
$configPath=Join-Path $rootPath 'config.json'
[IO.File]::WriteAllText($configPath,($config | ConvertTo-Json -Depth 8),(New-Object Text.UTF8Encoding($false)))
$arguments='"'+(Join-Path $rootPath 'service.py')+'" --config "'+$configPath+'" --host 127.0.0.1 --port '+$Port
$windowlessPython=Join-Path (Split-Path $pythonPath) 'pythonw.exe'
if (!(Test-Path -LiteralPath $windowlessPython)) {throw 'Windowless Python missing'}
$action=New-ScheduledTaskAction -Execute $windowlessPython -Argument $arguments -WorkingDirectory $rootPath
$principal=New-ScheduledTaskPrincipal -UserId $owner -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $TaskName
[pscustomobject]@{task=$TaskName;host='127.0.0.1';port=$Port;config=$configPath;token_file=$tokenPath} | ConvertTo-Json
