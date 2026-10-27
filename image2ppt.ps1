[CmdletBinding()]
param(
  [Parameter(Position=0, Mandatory=$true)]
  [ValidateSet('doctor','demo','prepare','compile','validate','inspect','perturb','test','replay','regress','resume','compose','layered')]
  [string]$Command,
  [string[]]$InputPath,
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
  if ($Command -eq 'resume') { $coreArguments += @('--run') + $InputPath }
  else {
    $inputOption = if ($Command -eq 'replay') { '--ir' } else { '--suite' }
    if (-not $OutputPath) { $OutputPath = Join-Path $PSScriptRoot 'outputs\core' }
    $coreArguments += @($inputOption) + $InputPath + @('--output', $OutputPath)
  }
  & $pythonExecutable -X utf8 @coreArguments
} elseif ($Command -eq 'prepare') {
  if (-not $InputPath) { throw 'prepare 需要 -InputPath。' }
  & $pythonExecutable -X utf8 (Join-Path $PSScriptRoot 'scripts\prepare.py') prepare $InputPath
} elseif ($Command -eq 'test') {
  & $nodeExecutable --test (Join-Path $PSScriptRoot 'tests\ir.test.mjs')
} elseif ($Command -eq 'layered') {
  if (-not $InputPath) { throw 'layered 需要 -InputPath，传入一张或多张页面图片。' }
  $layeredArgs = @((Join-Path $PSScriptRoot 'scripts\layered.py'), '--inputs') + @($InputPath | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
  if ($OutputPath) { $layeredArgs += @('--output', $OutputPath) }
  if ($env:IMAGE2PPT_LAYERED_WORKER) { $layeredArgs += @('--worker-config', $env:IMAGE2PPT_LAYERED_WORKER) }
  if ($env:IMAGE2PPT_MODEL_CONFIG) { $layeredArgs += @('--model-config', $env:IMAGE2PPT_MODEL_CONFIG) }
  & $pythonExecutable -X utf8 @layeredArgs
} elseif ($Command -eq 'compose') {
  if (-not $InputPath) { throw 'compose 需要 -InputPath，按页序传入一个或多个 IR。' }
  $irPaths = @($InputPath | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
  $commandArgs = @((Join-Path $PSScriptRoot 'scripts\compose.mjs')) + $irPaths
  if ($OutputPath) { $commandArgs += @('--output', $OutputPath) }
  & $nodeExecutable @commandArgs
} else {
  $commandArgs = @((Join-Path $PSScriptRoot 'scripts\main.mjs'), $Command)
  if ($InputPath) { $commandArgs += @('--input') + $InputPath }
  if ($NodeId) { $commandArgs += @('--node', $NodeId) }
  if ($Action) { $commandArgs += @('--action', $Action) }
  if ($Value) { $commandArgs += @('--value', $Value) }
  & $nodeExecutable @commandArgs
}
if ($LASTEXITCODE -ne 0) { throw "image2ppt 执行失败，退出码 $LASTEXITCODE。" }

