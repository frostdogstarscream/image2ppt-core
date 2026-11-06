[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$InputPptx,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [ValidateSet('PowerPoint','WPS')][string]$Application = 'PowerPoint',
    [int]$ExpectedSlides = 12,
    [int]$CoverSlide = 1,
    [int]$EditSlide = 3,
    [int[]]$ReferenceSlides = @(2,8),
    [string]$CoverTarget = 'paragraph-l003',
    [string]$ParagraphTarget = 'paragraph-enterprise-body',
    [string]$IconTarget = 'icon-crane-group',
    [string]$CardTarget = 'surface-03-01-body',
    [string]$CoverReplacement = ([string][char]0x6D4B),
    [string]$ParagraphReplacement = 'Edit check',
    [double]$IconDeltaXPt = 18,
    [double]$IconDeltaYPt = 9,
    [ValidatePattern('^[0-9A-Fa-f]{6}$')][string]$CardFillHex = 'DDEEDD',
    [int]$ExportWidth = 1672,
    [int]$ExportHeight = 941,
    [string]$WorkingFileName = 'source-copy.pptx',
    [string]$EditedFileName = 'edited-copy.pptx',
    [string]$ReportFileName = 'report.json'
)

# Application automation, never human acceptance. No security settings/alerts are
# changed. Only presentations opened here are closed; existing documents survive.
$ErrorActionPreference = 'Stop'
$inputFullPath = [IO.Path]::GetFullPath($InputPptx)
$runDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath $inputFullPath -PathType Leaf)) { throw 'Input PPTX does not exist.' }
if ([IO.Path]::GetExtension($inputFullPath) -ine '.pptx') { throw 'Input must be a PPTX file.' }
if (Test-Path -LiteralPath $runDirectory) { throw 'Refusing to reuse any output directory.' }
if ($ExpectedSlides -ne 12) { throw 'This scoped r7.1 application check requires exactly twelve slides.' }
if ($ExportWidth -le 0 -or $ExportHeight -le 0) { throw 'Export dimensions must be positive.' }
if ($CoverSlide -eq $EditSlide) { throw 'Cover and body edit slides must differ.' }
foreach ($number in @($CoverSlide,$EditSlide) + $ReferenceSlides) {
    if ($number -lt 1 -or $number -gt $ExpectedSlides) { throw 'Requested slide number is outside the deck.' }
}
if (-not $CoverReplacement -or -not $ParagraphReplacement -or
    $CoverReplacement.Contains("`r") -or $CoverReplacement.Contains("`n") -or
    $ParagraphReplacement.Contains("`r") -or $ParagraphReplacement.Contains("`n")) {
    throw 'Replacement strings must be nonempty single-line strings.'
}
$outputNames = @($WorkingFileName,$EditedFileName,$ReportFileName)
foreach ($name in $outputNames) {
    if ([IO.Path]::GetFileName($name) -cne $name -or $name -in @('', '.', '..')) {
        throw 'Output file names must be plain file names, not paths.'
    }
}
if (@($outputNames | ForEach-Object { $_.ToLowerInvariant() } | Select-Object -Unique).Count -ne 3) {
    throw 'Working, edited and report file names must be distinct.'
}
if ([IO.Path]::GetExtension($WorkingFileName) -ine '.pptx' -or [IO.Path]::GetExtension($EditedFileName) -ine '.pptx') {
    throw 'Diagnostic presentation file names must end in .pptx.'
}
[void](New-Item -ItemType Directory -Path $runDirectory)
$reportPath = Join-Path $runDirectory $ReportFileName
$workingPath = Join-Path $runDirectory $WorkingFileName
$editedPath = Join-Path $runDirectory $EditedFileName
$processName = if ($Application -eq 'PowerPoint') { 'POWERPNT' } else { 'wpp' }
$initialProcesses = @(Get-Process -Name $processName -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$report = [ordered]@{
    schema = 'OfficeApplicationCheck/1'; status = 'REVIEW_REQUIRED'
    verification_type = 'APP_AUTOMATED'; human_acceptance = $false
    application = $Application; application_version = $null; com_progid = $null
    powerpoint_checked = $false; wps_checked = $false
    application_open_checked = $false; application_edit_checked = $false
    script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    input = $inputFullPath; output = $runDirectory; expected_slides = $ExpectedSlides
    input_sha256_before = (Get-FileHash -LiteralPath $inputFullPath -Algorithm SHA256).Hash
    input_sha256_after = $null; input_original_unchanged = $false
    initial_process_ids = $initialProcesses; presentations_before_open = $null
    opened_slide_count = $null; saved_slide_count = $null; reopened_slide_count = $null
    checks = [ordered]@{}; requested_edits = @(); exports = @(); registrations = @()
    errors = @(); cleanup_errors = @(); application_quit = $false
    close_scope = 'only_presentation_references_opened_by_this_script'
    evidence_scope = 'application_open_edit_save_reopen_and_semantic_package_checks; not_pixel_or_human_acceptance'
}
$app = $null; $presentations = $null; $presentation = $null; $reopened = $null
$sourceSnapshot = $null; $applicationUnavailable = $false
$allReferences = New-Object System.Collections.Generic.List[object]

function Release-ComReference {
    param($Reference)
    if ($null -ne $Reference -and [Runtime.InteropServices.Marshal]::IsComObject($Reference)) {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($Reference)
    }
}

function Get-ComRegistrations {
    param([string[]]$ProgIds)
    $found = @()
    foreach ($view in @([Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32)) {
        $classes = $null
        try {
            $classes = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::ClassesRoot, $view)
            foreach ($progId in $ProgIds) {
                $key = $null
                try {
                    $key = $classes.OpenSubKey($progId + '\CLSID')
                    if ($null -ne $key -and $key.GetValue('')) {
                        $found += [ordered]@{ progid = $progId; clsid = [string]$key.GetValue(''); registry_view = [string]$view }
                    }
                }
                finally { if ($null -ne $key) { $key.Dispose() } }
            }
        }
        finally { if ($null -ne $classes) { $classes.Dispose() } }
    }
    return $found
}

function Read-ZipXml {
    param($Package, [string]$Part)
    $entry = $Package.GetEntry($Part)
    if ($null -eq $entry) { throw "Required OOXML part missing: $Part" }
    $stream = $entry.Open(); $reader = New-Object IO.StreamReader($stream)
    try {
        $doc = New-Object Xml.XmlDocument
        $doc.XmlResolver = $null
        $doc.LoadXml($reader.ReadToEnd())
        return ,$doc
    }
    finally { $reader.Dispose(); $stream.Dispose() }
}

function New-Namespaces {
    param($Document)
    $ns = New-Object Xml.XmlNamespaceManager($Document.NameTable)
    $ns.AddNamespace('p','http://schemas.openxmlformats.org/presentationml/2006/main')
    $ns.AddNamespace('a','http://schemas.openxmlformats.org/drawingml/2006/main')
    $ns.AddNamespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
    return ,$ns
}

function Get-PackageSnapshot {
    param([string]$Path)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $package = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $pres = Read-ZipXml $package 'ppt/presentation.xml'; $ns = New-Namespaces $pres
        $rels = Read-ZipXml $package 'ppt/_rels/presentation.xml.rels'
        $relationships = @{}
        foreach ($rel in $rels.DocumentElement.ChildNodes) {
            if ($rel.LocalName -eq 'Relationship' -and $rel.GetAttribute('TargetMode') -ne 'External') {
                $relationships[$rel.GetAttribute('Id')] = $rel.GetAttribute('Target')
            }
        }
        $slides = @(); $order = @()
        foreach ($id in $pres.SelectNodes('/p:presentation/p:sldIdLst/p:sldId', $ns)) {
            $rid = $id.GetAttribute('id', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
            if (-not $relationships.ContainsKey($rid)) { throw 'Slide relationship missing.' }
            $resolved = [Uri]::new([Uri]'https://ooxml.invalid/ppt/presentation.xml', $relationships[$rid])
            if ($resolved.Host -ne 'ooxml.invalid' -or $resolved.Query -or $resolved.Fragment) { throw 'Invalid slide part relationship.' }
            $part = $resolved.AbsolutePath.TrimStart('/')
            $doc = Read-ZipXml $package $part; $slideNs = New-Namespaces $doc
            $objects = [ordered]@{}
            foreach ($shape in $doc.SelectNodes('//p:sp | //p:pic | //p:grpSp', $slideNs)) {
                $nv = $shape.SelectSingleNode('./p:nvSpPr/p:cNvPr | ./p:nvPicPr/p:cNvPr | ./p:nvGrpSpPr/p:cNvPr', $slideNs)
                if ($null -eq $nv) { throw 'Shape identity missing from OOXML.' }
                $name = $nv.GetAttribute('name')
                if (-not $name -or $objects.Contains($name)) { throw 'Missing or duplicate object name in slide.' }
                $transform = $shape.SelectSingleNode('./p:spPr/a:xfrm | ./p:grpSpPr/a:xfrm', $slideNs)
                $bounds = @()
                if ($null -ne $transform) {
                    foreach ($field in @('off','ext','chOff','chExt')) {
                        $partNode = $transform.SelectSingleNode('./a:'+$field, $slideNs)
                        if ($null -ne $partNode) {
                            foreach ($attr in @($partNode.Attributes | Sort-Object Name)) {
                                $bounds += $field+'.'+$attr.Name+'='+$attr.Value
                            }
                        }
                    }
                    $bounds += 'rotation=' + $(if ($transform.HasAttribute('rot')) { $transform.GetAttribute('rot') } else { '0' })
                }
                $text = @(); $runs = @()
                foreach ($paragraph in $shape.SelectNodes('./p:txBody/a:p', $slideNs)) {
                    $text += -join @($paragraph.SelectNodes('./a:r/a:t | ./a:fld/a:t', $slideNs) | ForEach-Object { $_.InnerText })
                    $runs += [int]$paragraph.SelectNodes('./a:r', $slideNs).Count
                }
                $objects[$name] = [ordered]@{ type = $shape.LocalName; bounds = $bounds; paragraphs = $text; run_counts = $runs }
            }
            $order += [string]$id.GetAttribute('id')
            $slides += [ordered]@{ part = $part; objects = $objects }
        }
        return [ordered]@{ order = $order; slides = $slides }
    }
    finally { $package.Dispose() }
}

function Find-NamedShape {
    param($Shapes, [string]$ExpectedName)
    for ($shapeIndex = 1; $shapeIndex -le $Shapes.Count; $shapeIndex++) {
        $shape = $Shapes.Item($shapeIndex)
        if ([string]::Equals([string]$shape.Name, $ExpectedName, [StringComparison]::Ordinal)) { return ,$shape }
        if ([int]$shape.Type -eq 6) {
            $children = $null
            try {
                $children = $shape.GroupItems
                $found = Find-NamedShape -Shapes $children -ExpectedName $ExpectedName
                if ($null -ne $found) { return ,$found }
            }
            finally { Release-ComReference $children; Release-ComReference $shape }
        }
        else { Release-ComReference $shape }
    }
    return $null
}

function Require-Shape {
    param($Deck, [int]$SlideNumber, [string]$ExpectedName)
    $slide = $null; $shapes = $null
    try {
        $slide = $Deck.Slides.Item($SlideNumber); $shapes = $slide.Shapes
        $found = Find-NamedShape $shapes $ExpectedName
        if ($null -eq $found) { throw "Required actual application object missing: slide $SlideNumber / $ExpectedName" }
        [void]$allReferences.Add($found)
        return ,$found
    }
    finally { Release-ComReference $shapes; Release-ComReference $slide }
}

function Get-ParagraphState {
    param($Shape)
    $frame = $null; $range = $null; $paragraphs = $null; $result = @()
    try {
        if ([int]$Shape.HasTextFrame -ne -1) { throw 'Target does not have an editable text frame.' }
        $frame = $Shape.TextFrame; $range = $frame.TextRange; $paragraphs = $range.Paragraphs()
        for ($i = 1; $i -le $paragraphs.Count; $i++) {
            $paragraph = $null; $font = $null
            try {
                $paragraph = $range.Paragraphs($i,1); $font = $paragraph.Font
                $result += [ordered]@{
                    text = ([string]$paragraph.Text).TrimEnd([char[]]@([char]13,[char]10))
                    font_name = [string]$font.Name; font_east_asian = [string]$font.NameFarEast
                    size = [double]$font.Size; bold = [int]$font.Bold; italic = [int]$font.Italic
                }
            }
            finally { Release-ComReference $font; Release-ComReference $paragraph }
        }
        return ,$result
    }
    finally { Release-ComReference $paragraphs; Release-ComReference $range; Release-ComReference $frame }
}

function Replace-ParagraphContents {
    param($Shape, [string]$Replacement, $SourceObject)
    $before = Get-ParagraphState $Shape
    if ($before.Count -ne $SourceObject.paragraphs.Count -or @($SourceObject.run_counts | Where-Object { $_ -ne 1 }).Count) {
        throw 'This scoped application edit requires one source run per paragraph and unchanged paragraph count.'
    }
    $frame = $null; $range = $null
    try {
        $frame = $Shape.TextFrame; $range = $frame.TextRange
        # Work backwards and leave paragraph separators and paragraph formatting intact.
        for ($i = $before.Count; $i -ge 1; $i--) {
            $paragraph = $null; $content = $null
            try {
                $paragraph = $range.Paragraphs($i,1)
                $length = $before[$i-1].text.Length
                if ($length -lt 1) { throw 'Cannot replace an empty source paragraph.' }
                $content = $paragraph.Characters(1,$length); $content.Text = $Replacement
            }
            finally { Release-ComReference $content; Release-ComReference $paragraph }
        }
    }
    finally { Release-ComReference $range; Release-ComReference $frame }
    return ,$before
}

function Assert-ParagraphEdit {
    param($Shape, $Before, [string]$Replacement)
    $after = Get-ParagraphState $Shape
    if ($after.Count -ne $Before.Count) { throw 'Saved paragraph count changed.' }
    for ($i=0; $i -lt $after.Count; $i++) {
        if ($after[$i].text -cne $Replacement) { throw 'Saved short replacement differs from request.' }
        foreach ($field in @('font_name','font_east_asian','size','bold','italic')) {
            if ($after[$i][$field] -cne $Before[$i][$field]) { throw "Saved paragraph style changed: $field" }
        }
    }
    return ,$after
}

function Get-SlideOrder {
    param($Deck)
    $order = @()
    for ($i=1; $i -le $Deck.Slides.Count; $i++) {
        $slide = $Deck.Slides.Item($i)
        try { $order += [string]$slide.SlideID } finally { Release-ComReference $slide }
    }
    return ,$order
}

function Export-CheckedSlide {
    param($Deck, [int]$Number, [string]$Stage)
    $slide = $null
    try {
        $slide = $Deck.Slides.Item($Number)
        $name = 'page-{0:D2}-{1}-{2}.png' -f $Number,$Application.ToLowerInvariant(),$Stage
        $path = Join-Path $runDirectory $name
        if (Test-Path -LiteralPath $path) { throw 'Refusing to overwrite a slide export.' }
        $slide.Export($path, 'PNG', $ExportWidth, $ExportHeight)
        $file = Get-Item -LiteralPath $path
        if ($file.Length -le 0) { throw 'Application exported an empty image.' }
        return [ordered]@{ slide = $Number; stage = $Stage; path = $path; bytes = $file.Length
            application = $Application; sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
    }
    finally { Release-ComReference $slide }
}

function Assert-SavedPackage {
    param($Before, $After)
    if (($Before.order -join ',') -cne ($After.order -join ',')) { throw 'OOXML slide order changed.' }
    for ($i=0; $i -lt $Before.slides.Count; $i++) {
        $old = $Before.slides[$i].objects; $new = $After.slides[$i].objects
        if (($old.Keys -join '|') -cne ($new.Keys -join '|')) { throw "OOXML object membership/order changed on slide $($i+1)." }
        foreach ($name in $old.Keys) {
            if ($old[$name].type -cne $new[$name].type) { throw 'Saved object type changed.' }
            $movingIcon = ($i+1 -eq $EditSlide -and $name -ceq $IconTarget)
            if (-not $movingIcon -and ($old[$name].bounds -join '|') -cne ($new[$name].bounds -join '|')) {
                throw "Saved untargeted object geometry changed: slide $($i+1) / $name"
            }
            $replacement = $null
            if ($i+1 -eq $CoverSlide -and $name -ceq $CoverTarget) { $replacement = $CoverReplacement }
            if ($i+1 -eq $EditSlide -and $name -ceq $ParagraphTarget) { $replacement = $ParagraphReplacement }
            if ($null -ne $replacement) {
                if ($old[$name].paragraphs.Count -ne $new[$name].paragraphs.Count) { throw 'OOXML paragraph count changed.' }
                foreach ($text in $new[$name].paragraphs) { if ($text -cne $replacement) { throw 'OOXML requested replacement missing.' } }
                if (($old[$name].run_counts -join ',') -cne ($new[$name].run_counts -join ',')) { throw 'OOXML run structure changed.' }
            }
            elseif (($old[$name].paragraphs -join "`n") -cne ($new[$name].paragraphs -join "`n")) {
                throw "OOXML untargeted text changed: slide $($i+1) / $name"
            }
        }
    }
    return [ordered]@{ status = 'PASS'; scope = 'slide_order; object_membership_order_type; untargeted_geometry_text; targeted_paragraph_run_structure' }
}

try {
    $progIds = if ($Application -eq 'PowerPoint') { @('PowerPoint.Application','PowerPoint.Application.16') }
               else { @('KWPP.Application','WPP.Application') }
    $report.registrations = @(Get-ComRegistrations $progIds)
    if ($report.registrations.Count -eq 0) {
        $applicationUnavailable = $true
        $report.status = 'NOT_RUN'
        $report['not_run_reason'] = 'requested_application_has_no_registered_COM_class'
    }
    else {
        Copy-Item -LiteralPath $inputFullPath -Destination $workingPath
        if ((Get-FileHash -LiteralPath $workingPath -Algorithm SHA256).Hash -cne $report.input_sha256_before) {
            throw 'Diagnostic input copy differs from the supplied source.'
        }
        $sourceSnapshot = Get-PackageSnapshot $workingPath
        if ($sourceSnapshot.slides.Count -ne $ExpectedSlides) { throw 'Input OOXML does not contain twelve slides.' }
        $sourceSnapshot | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath (Join-Path $runDirectory 'source-structure.json') -Encoding UTF8
        $report.com_progid = $report.registrations[0].progid
        $app = New-Object -ComObject $report.com_progid
        $report.application_version = [string]$app.Version
        $presentations = $app.Presentations
        $report.presentations_before_open = [int]$presentations.Count
        # ReadOnly=False, Untitled=True, WithWindow=False. The source itself is never opened.
        $presentation = $presentations.Open($workingPath, 0, -1, 0)
        if ([int]$presentation.ReadOnly -ne 0) { throw 'Application did not provide an editable diagnostic copy.' }
        $report.opened_slide_count = [int]$presentation.Slides.Count
        if ($report.opened_slide_count -ne $ExpectedSlides) { throw 'Application opened a different slide count.' }
        $orderBefore = Get-SlideOrder $presentation
        if (($orderBefore -join ',') -cne ($sourceSnapshot.order -join ',')) { throw 'Application opened a different slide order.' }
        $report.checks['source_open'] = [ordered]@{ status = 'PASS'; untitled_copy = $true; slide_order = $orderBefore }
        $report.application_open_checked = $true
        foreach ($number in @(@($CoverSlide,$EditSlide) + $ReferenceSlides | Sort-Object -Unique)) {
            $report.exports += Export-CheckedSlide $presentation $number 'before'
        }
        $cover = Require-Shape $presentation $CoverSlide $CoverTarget
        $body = Require-Shape $presentation $EditSlide $ParagraphTarget
        $icon = Require-Shape $presentation $EditSlide $IconTarget
        $card = Require-Shape $presentation $EditSlide $CardTarget
        if ([int]$icon.Type -ne 6) { throw 'Icon target is not an actual group.' }
        if ([int]$card.Type -ne 1) { throw 'Card target is not a native AutoShape.' }
        $coverBefore = Replace-ParagraphContents $cover $CoverReplacement $sourceSnapshot.slides[$CoverSlide-1].objects[$CoverTarget]
        $bodyBefore = Replace-ParagraphContents $body $ParagraphReplacement $sourceSnapshot.slides[$EditSlide-1].objects[$ParagraphTarget]
        $expectedLeft = [double]$icon.Left + $IconDeltaXPt
        $expectedTop = [double]$icon.Top + $IconDeltaYPt
        $icon.Left = [single]$expectedLeft; $icon.Top = [single]$expectedTop
        $red = [Convert]::ToInt32($CardFillHex.Substring(0,2),16)
        $green = [Convert]::ToInt32($CardFillHex.Substring(2,2),16)
        $blue = [Convert]::ToInt32($CardFillHex.Substring(4,2),16)
        $fillRgb = $red + 256*$green + 65536*$blue
        $card.Fill.Solid(); $card.Fill.ForeColor.RGB = $fillRgb
        $report.requested_edits = @(
            [ordered]@{ slide=$CoverSlide; target=$CoverTarget; action='replace_each_paragraph'; value=$CoverReplacement; before=$coverBefore },
            [ordered]@{ slide=$EditSlide; target=$ParagraphTarget; action='replace_each_paragraph'; value=$ParagraphReplacement; before=$bodyBefore },
            [ordered]@{ slide=$EditSlide; target=$IconTarget; action='move_group'; expected_left_pt=$expectedLeft; expected_top_pt=$expectedTop },
            [ordered]@{ slide=$EditSlide; target=$CardTarget; action='native_fill'; expected_rgb_hex=$CardFillHex; expected_ole_rgb=$fillRgb }
        )
        if (Test-Path -LiteralPath $editedPath) { throw 'Refusing to overwrite edited copy.' }
        # ppSaveAsOpenXMLPresentation = 24. Never call Save() on the input file.
        $presentation.SaveAs($editedPath,24)
        $report['edited_copy'] = $editedPath
        $report['edited_copy_sha256'] = (Get-FileHash -LiteralPath $editedPath -Algorithm SHA256).Hash
        $savedSnapshot = Get-PackageSnapshot $editedPath
        $report.saved_slide_count = $savedSnapshot.slides.Count
        if ($report.saved_slide_count -ne $ExpectedSlides) { throw 'Saved OOXML slide count changed.' }
        $savedSnapshot | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath (Join-Path $runDirectory 'saved-structure.json') -Encoding UTF8
        $report.checks['saved_package'] = Assert-SavedPackage $sourceSnapshot $savedSnapshot
        foreach ($reference in $allReferences) { Release-ComReference $reference }; $allReferences.Clear()
        $presentation.Close(); Release-ComReference $presentation; $presentation = $null
        $reopened = $presentations.Open($editedPath,-1,0,0)
        $report.reopened_slide_count = [int]$reopened.Slides.Count
        if ($report.reopened_slide_count -ne $ExpectedSlides) { throw 'Reopened slide count changed.' }
        $orderAfter = Get-SlideOrder $reopened
        if (($orderAfter -join ',') -cne ($orderBefore -join ',')) { throw 'Reopened slide order changed.' }
        $report.checks['save_reopen_order'] = [ordered]@{ status='PASS'; slide_order=$orderAfter }
        $cover = Require-Shape $reopened $CoverSlide $CoverTarget
        $body = Require-Shape $reopened $EditSlide $ParagraphTarget
        $icon = Require-Shape $reopened $EditSlide $IconTarget
        $card = Require-Shape $reopened $EditSlide $CardTarget
        $report.checks['cover_short_edit'] = [ordered]@{ status='PASS'; actual=(Assert-ParagraphEdit $cover $coverBefore $CoverReplacement) }
        $report.checks['paragraph_edit'] = [ordered]@{ status='PASS'; actual=(Assert-ParagraphEdit $body $bodyBefore $ParagraphReplacement) }
        if ([int]$icon.Type -ne 6 -or [math]::Abs([double]$icon.Left-$expectedLeft) -gt 0.05 -or
            [math]::Abs([double]$icon.Top-$expectedTop) -gt 0.05) { throw 'Reopened icon group differs from requested move.' }
        $report.checks['icon_move'] = [ordered]@{ status='PASS'; actual_left_pt=[double]$icon.Left; actual_top_pt=[double]$icon.Top }
        if ([int]$card.Type -ne 1 -or [int]$card.Fill.Type -ne 1 -or [int]$card.Fill.ForeColor.RGB -ne $fillRgb) {
            throw 'Reopened native card fill differs from requested color.'
        }
        $report.checks['card_fill'] = [ordered]@{ status='PASS'; actual_ole_rgb=[int]$card.Fill.ForeColor.RGB }
        foreach ($number in @(@($CoverSlide,$EditSlide) + $ReferenceSlides | Sort-Object -Unique)) {
            $report.exports += Export-CheckedSlide $reopened $number 'saved'
        }
        $report.application_edit_checked = $true
    }
}
catch {
    $report.errors += [ordered]@{ type=$_.Exception.GetType().FullName; message=$_.Exception.Message; hresult=$_.Exception.HResult }
    $report.status = 'REVIEW_REQUIRED'
}
finally {
    foreach ($reference in $allReferences) {
        try { Release-ComReference $reference } catch { $report.cleanup_errors += $_.Exception.Message }
    }
    foreach ($ownedPresentation in @($reopened,$presentation)) {
        if ($null -ne $ownedPresentation) {
            try { $ownedPresentation.Saved=-1; $ownedPresentation.Close() }
            catch { $report.cleanup_errors += $_.Exception.Message }
            try { Release-ComReference $ownedPresentation } catch { $report.cleanup_errors += $_.Exception.Message }
        }
    }
    if ($null -ne $app) {
        try {
            $report['presentations_after_cleanup'] = [int]$app.Presentations.Count
            # Be conservative: unknown ProtectedView support means leave the app
            # running. Never quit a preexisting process or an app with user docs.
            if ($initialProcesses.Count -eq 0 -and $report.presentations_before_open -eq 0 -and
                [int]$app.Presentations.Count -eq 0) {
                $protectedCount = $null
                try { $protectedCount = [int]$app.ProtectedViewWindows.Count }
                catch { $report['quit_skipped_reason']='protected_view_inventory_unavailable' }
                if ($null -ne $protectedCount -and $protectedCount -eq 0) { $app.Quit(); $report.application_quit=$true }
            }
            else { $report['quit_skipped_reason']='preexisting_process_or_remaining_presentations' }
        }
        catch { $report.cleanup_errors += $_.Exception.Message }
    }
    try { Release-ComReference $presentations } catch { $report.cleanup_errors += $_.Exception.Message }
    try { Release-ComReference $app } catch { $report.cleanup_errors += $_.Exception.Message }
    try {
        $report.input_sha256_after = (Get-FileHash -LiteralPath $inputFullPath -Algorithm SHA256).Hash
        $report.input_original_unchanged = $report.input_sha256_before -ceq $report.input_sha256_after
        if (-not $report.input_original_unchanged) { $report.errors += [ordered]@{ type='SourceHashCheck'; message='Original source changed.' } }
    }
    catch { $report.errors += [ordered]@{ type='SourceHashCheck'; message=$_.Exception.Message } }
    if ($report.application_open_checked -and $report.application_edit_checked -and $report.input_original_unchanged -and
        $report.errors.Count -eq 0 -and $report.cleanup_errors.Count -eq 0) {
        $report.status='PASS_APP_AUTOMATED'
        $report.powerpoint_checked=($Application -eq 'PowerPoint'); $report.wps_checked=($Application -eq 'WPS')
    }
    elseif ($applicationUnavailable -and $report.errors.Count -eq 0) { $report.status='NOT_RUN' }
    else { $report.status='REVIEW_REQUIRED' }
    $report['completed_at_utc'] = [DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText($reportPath,($report | ConvertTo-Json -Depth 18),[Text.UTF8Encoding]::new($false))
    # Keep business text and object snapshots in the local report, not stdout.
    Write-Output ([ordered]@{ status=$report.status; application=$Application; report=$reportPath
        original_unchanged=$report.input_original_unchanged; opened=$report.opened_slide_count
        saved=$report.saved_slide_count; reopened=$report.reopened_slide_count } | ConvertTo-Json)
}

if ($report.status -eq 'NOT_RUN') { exit 2 }
if ($report.status -ne 'PASS_APP_AUTOMATED') { exit 1 }
