#requires -Version 5.1
# Optional: -InpaintUrl http://127.0.0.1:18841/inpaint  (Big-LaMa, see lama_server.py)
param([string]$InpaintUrl = '')
# Local batch: for every page in .tmp-test/deck/<id>/recon/calibrated,
#   plate -> gradient fit -> compile -> measure -> gate(5) -> compile -> measure
# Writes .tmp-test/deck-summary.json
$ErrorActionPreference = 'Stop'
$repo = 'D:\Code\image2PPT'
$py = Join-Path $repo 'services\vision\.venv\Scripts\python.exe'
$root = Join-Path $repo '.tmp-test\deck'
$srcDir = Join-Path $repo 'qa-inputs\xuzhou-20261103\source'
$summaryPath = Join-Path $repo '.tmp-test\deck-summary.json'

function Newest([string]$dir) {
  $o = Get-ChildItem $dir -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
  if (-not $o) { return $null }
  return (Join-Path $o[0].FullName 'preview.png')
}

function Measure-Page([string]$ir, [string]$src, [string]$preview, [string]$out) {
  & $py (Join-Path $repo 'scripts\ir_refine.py') measure --ir $ir --source $src --preview $preview --out $out 2>&1 | Out-Null
  return (Get-Content $out -Raw -Encoding UTF8 | ConvertFrom-Json)
}

$results = @()
$pages = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Sort-Object Name
foreach ($page in $pages) {
  $id = $page.Name
  $cal = Join-Path $page.FullName 'recon\calibrated'
  $ir0 = Join-Path $cal 'slideir.json'
  if (-not (Test-Path $ir0)) { Write-Host "[skip] $id no calibrated IR"; continue }
  $src = Get-ChildItem $srcDir -Filter "$id`_*.png" | Select-Object -First 1
  if (-not $src) { Write-Host "[skip] $id no source image"; continue }

  Write-Host "=== [$id] $($src.Name)"
  $plate = Join-Path $page.FullName 'plate.png'
  $v1 = Join-Path $page.FullName 'v1'
  $h5 = Join-Path $page.FullName 'h5'
  foreach ($d in @($v1, $h5)) { Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue }

  # 1) plate + gradient-only IR
  $plateArgs = @('plate','--ir',$ir0,'--source',$src.FullName,'--out',$plate)
  if ($InpaintUrl) { $plateArgs += @('--inpaint-url', $InpaintUrl) }
  & $py (Join-Path $repo 'scripts\ir_refine.py') @plateArgs 2>&1 | Out-Null
  & $py (Join-Path $repo 'scripts\ir_refine.py') apply --ir $ir0 --source $src.FullName `
      --report (Join-Path $page.FullName 'plate.json') --plate $plate --out $v1 --threshold 9999 2>&1 | Out-Null

  # 2) compile v1 and measure it
  $oir = Join-Path $v1 'slideir.json'
  & (Join-Path $repo 'image2ppt.ps1') compile -InputPath $oir 2>&1 | Out-Null
  $pv1 = Newest (Join-Path $v1 'outputs')
  if (-not $pv1) { Write-Host "[fail] $id v1 compile"; $results += [pscustomobject]@{page=$id;status='compile_failed'}; continue }
  $m1 = Measure-Page $oir $src.FullName $pv1 (Join-Path $page.FullName 'm1.json')

  # 3) gate at 5, compile, measure
  & $py (Join-Path $repo 'scripts\ir_refine.py') apply --ir $oir --source $src.FullName `
      --report (Join-Path $page.FullName 'm1.json') --plate $plate --out $h5 --threshold 5 2>&1 | Out-Null
  $hir = Join-Path $h5 'slideir.json'
  & (Join-Path $repo 'image2ppt.ps1') compile -InputPath $hir 2>&1 | Out-Null
  $ph5 = Newest (Join-Path $h5 'outputs')
  if (-not $ph5) { Write-Host "[fail] $id h5 compile"; $results += [pscustomobject]@{page=$id;status='compile_failed'}; continue }
  $m5 = Measure-Page $hir $src.FullName $ph5 (Join-Path $page.FullName 'm5.json')

  $irH = Get-Content $hir -Raw -Encoding UTF8 | ConvertFrom-Json
  $imgArea = ($irH.nodes | Where-Object { $_.type -eq 'image' } | ForEach-Object { $_.bbox[2] * $_.bbox[3] } | Measure-Object -Sum).Sum
  $canvas = $irH.canvas.source_px
  $raster = [math]::Round(100.0 * $imgArea / ($canvas[0] * $canvas[1]), 1)
  $shapes = ($irH.nodes | Where-Object { $_.type -eq 'shape' }).Count
  $texts = ($irH.nodes | Where-Object { $_.type -eq 'text' }).Count

  $row = [pscustomobject]@{
    page       = $id
    status     = 'ok'
    baseline   = $m1.page_mean_diff
    gated      = $m5.page_mean_diff
    text_px    = $m5.nodes.where{ $_.type -eq 'text' } | ForEach-Object { $_.mean_diff } | Measure-Object -Average | Select-Object -ExpandProperty Average
    raster_pct = $raster
    shapes     = $shapes
    texts      = $texts
    preview    = $ph5
  }
  $row.text_px = [math]::Round($row.text_px, 2)
  $results += $row
  Write-Host ("  baseline {0:n2} -> gated {1:n2} | raster {2}% | shapes {3} | texts {4}" -f $row.baseline, $row.gated, $row.raster_pct, $row.shapes, $row.texts)
}

$results | ConvertTo-Json -Depth 4 | Set-Content $summaryPath -Encoding UTF8
Write-Host "`nsummary -> $summaryPath"
$results | Format-Table -AutoSize
