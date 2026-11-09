param([Parameter(Mandatory=$true)][string]$ServiceRoot,[Parameter(Mandatory=$true)][string]$TaskName)
$ErrorActionPreference='Stop'
$rootPath=[IO.Path]::GetFullPath($ServiceRoot)
$configPath=Join-Path $rootPath 'config.json'
$config=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$task=Get-ScheduledTask -TaskName $TaskName
if ($task.Actions.Count -ne 1 -or !$task.Actions[0].Arguments.Contains($rootPath)) {throw 'Task does not belong to this deployment'}
$active=Get-ChildItem -LiteralPath $config.jobs -Directory | ForEach-Object {
  $statusPath=Join-Path $_.FullName 'status.json'
  if (Test-Path -LiteralPath $statusPath) {
    $state=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($state.status -in @('QUEUED','RUNNING')) {$_.FullName}
  }
}
if ($active) {throw 'SAM jobs active; wait before updating'}
Stop-ScheduledTask -TaskName $TaskName
for ($attempt=0;$attempt -lt 40;$attempt++) {
  $listener=Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 18843 -State Listen -ErrorAction SilentlyContinue
  if (!$listener) {break}
  Start-Sleep -Milliseconds 250
}
if ($listener) {throw 'Previous listener is still alive; do not launch a second GPU service'}
$pythonw=Join-Path (Split-Path $config.python) 'pythonw.exe'
if (!(Test-Path -LiteralPath $pythonw)) {throw 'Windowless Python missing'}
$action=New-ScheduledTaskAction -Execute $pythonw -Argument $task.Actions[0].Arguments -WorkingDirectory $rootPath
Set-ScheduledTask -TaskName $TaskName -Action $action | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Output 'PRIVATE_SAM_UPDATED'
