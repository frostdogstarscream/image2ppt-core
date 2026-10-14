[CmdletBinding()]
param(
  [Parameter(Position=0, Mandatory=$true)]
  [ValidateSet('doctor','demo','prepare','compile','validate','inspect','perturb','test','replay','regress','resume')]
  [string]$Command,
  [string]$InputPath,
  [string]$NodeId,
  [ValidateSet('delete','move','replace')][string]$Action,
  [string]$Value,
  [string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$runtimeBase = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'
$nodeExecutable = Join-Path $runtimeBase 'node\bin\node.exe'
$pythonExecutable = Join-Path $runtimeBase 'python\python.exe'
if ($env:IMAGE2PPT_NODE) { $nodeExecutable = $env:IMAGE2PPT_NODE }
if ($env:IMAGE2PPT_PYTHON) { $pythonExecutable = $env:IMAGE2PPT_PYTHON }
if ($Command -in @('replay','regress','resume')) {
  if (-not $InputPath) { throw "$Command 需要 -InputPath。" }
  $coreArguments = @((Join-Path $PSScriptRoot 'scripts\core_cli.py'), $Command)
  if ($Command -eq 'resume') { $coreArguments += @('--run', $InputPath) }
  else {
    $inputOption = if ($Command -eq 'replay') { '--ir' } else { '--suite' }
    if (-not $OutputPath) { $OutputPath = Join-Path $PSScriptRoot 'outputs\core' }
    $coreArguments += @($inputOption, $InputPath, '--output', $OutputPath)
  }
  & $pythonExecutable -X utf8 @coreArguments
} elseif ($Command -eq 'prepare') {
  if (-not $InputPath) { throw 'prepare 需要 -InputPath。' }
  & $pythonExecutable -X utf8 (Join-Path $PSScriptRoot 'scripts\prepare.py') prepare $InputPath
} elseif ($Command -eq 'test') {
  & $nodeExecutable --test (Join-Path $PSScriptRoot 'tests\ir.test.mjs')
} else {
  $commandArgs = @((Join-Path $PSScriptRoot 'scripts\main.mjs'), $Command)
  if ($InputPath) { $commandArgs += @('--input', $InputPath) }
  if ($NodeId) { $commandArgs += @('--node', $NodeId) }
  if ($Action) { $commandArgs += @('--action', $Action) }
  if ($Value) { $commandArgs += @('--value', $Value) }
  & $nodeExecutable @commandArgs
}
if ($LASTEXITCODE -ne 0) { throw "image2ppt 执行失败，退出码 $LASTEXITCODE。" }

