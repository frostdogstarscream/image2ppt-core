param([Parameter(Mandatory=$true)][string]$ServiceRoot,[Parameter(Mandatory=$true)][string]$TaskName)
$ErrorActionPreference='Stop'
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
Get-ScheduledTaskInfo -TaskName $TaskName | Select-Object LastRunTime,LastTaskResult
Get-NetTCPConnection -LocalPort 18843 -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort
$logPath=Join-Path $ServiceRoot 'config.log'
if (Test-Path -LiteralPath $logPath) {Get-Content -LiteralPath $logPath -Tail 35}
