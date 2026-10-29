# Layered quality loop

`quality-loop` is an opt-in pipeline based on `96571dc`. The default remains
`legacy`. It operates on generic image evidence; no document names, business
text, page-specific coordinates or page-number exceptions are in the pipeline.

## Run

Use a Python environment containing the existing vision dependencies (OpenCV,
NumPy, Pillow and the core package dependencies). The fixed compiler continues
to use the bundled Artifact Tool runtime.

```powershell
python -X utf8 scripts/layered.py --inputs <page1.png> <page2.png> `
  --worker-config <existing-worker.json> --model-config <existing-client.json> `
  --quality-policy quality-loop --baseline-job <same-source-baseline-job> `
  --max-extra-vision-calls 2 --max-repair-rounds 2 --output <new-output-root>
```

The PowerShell wrapper accepts `-QualityPolicy`, `-BaselineJob`,
`-MaxExtraVisionCalls` and `-MaxRepairRounds`. Model configuration is optional;
unavailable visual/OCR evidence is recorded as unresolved, never accepted as a
successful review. Remote work must use an authorized development worker and
authorized model service. The script does not deploy production containers.

The extra visual budget covers one pre-inpainting crop review and one
post-render crop review. The original object-box review and text paragraph
grouping remain separate base calls, disclosed in their existing records.
Every visual request is reserved on disk before dispatch, including failed
requests. There are no implicit retries. A maximum of eight regions fit each
batch; overflow is reported. Each repair attempt consumes its revision slot
before compiling; a failed attempt preserves any previous usable candidate.

## Evidence and selection

- Source input, scan, guards and original perception remain immutable. Object
  review uses `working-perception`; each `r00`/`r01`/`r02` has its own proposal,
  compiler evidence, image comparison, lineage and edit-probe records.
- Text corrections need source crop review and an independent crop OCR
  invocation. Conflicting evidence retains source pixels. OCR is explicitly
  labelled as recognition evidence rather than ground truth.
- Icon masks are checked before erasure. Detected damage receives a bounded
  local mask repair; unrecoverable icon pixels are retained. Presets with
  internal residual artwork fall back to the complete object crop.
- Retention closes over native text and complex objects, removes their duplicate
  nodes, punches intersecting raster alpha and clips plain rectangles. A single
  patch may cover at most 16% of the page; total retention is capped at 55%.
  Exceeding either bound is a review finding, not an automatic whole-slide fallback.
- Module grouping preserves overlap order. Actual PPTX copies are edited and
  reimported: all visible overlapping picture candidates are hidden in disjoint
  batches; one group is moved in the selected revision. The delivered bytes
  are hash-checked before and after probes.
- Candidates are ordered by blocking issues, confirmed defects, regional pixel
  error, then reliable native-object count. A new confirmed content defect is rejected.
  Newly exposed duplicate candidates still count as blocking, but do not prevent
  selection of a candidate with fewer blocking defects overall.
  A difference within 0.05 intensity units is treated as measurement equivalence;
  a lower native count cannot win a tie.
- Ordered composition reimports the final deck and binds each slide to the
  selected IR and source hash. A merged preview differing from its selected
  single-page render remains a separate review finding.

`LayeredJob/1` is preserved. Confirmed loss, structural failure and build failure
remain `FAILED`; otherwise the maximum status is `REVIEW_REQUIRED`.
`human_acceptance` and `office_acceptance` always remain false. Image objects
disclose `move_resize_only`; internal text and graphics in those images are
not native editable. Heuristic edge loss and OCR disagreement alone do not
establish a confirmed defect.

## Repeatable validation

```powershell
python -X utf8 scripts/validate_quality_loop.py freeze `
  --baseline-job <ten-page-baseline-job> --independent-input-dir <legal-input-dir> `
  --count 10 --output <new-freeze-directory>
python -X utf8 scripts/validate_quality_loop.py run `
  --manifest <new-freeze-directory>/frozen.json --dataset regression `
  --worker-config <existing-worker.json> --model-config <existing-client.json> `
  --output <new-run-root>
# Repeat run with --dataset independent after the rules have been frozen.
python -X utf8 scripts/check_automatic_delivery.py --job <new-job>
```

The holdout selection is ascending source SHA256, excluding regression hashes.
Source files, baseline PPTX/previews and local rule hashes must still match at
run time. Layout/background coverage requires visual inventory; different file
hashes alone do not establish layout diversity. Independent sample failures
must be reported without changing thresholds to hide them.

Mechanism checks:

```powershell
$env:PYTHONPATH='src;services/vision/src'
python -m pytest tests/test_layered_quality_loop.py services/vision/tests/test_layered.py services/vision/tests/test_layers.py -q --basetemp <new-test-directory>
./image2ppt.ps1 test
./image2ppt.ps1 demo
node scripts/check_demo.mjs <generated-synthetic-slideir.json>
```

Use a fresh test directory. Synthetic tests, successful file import and local
cached-perception integration do not establish fresh worker/model execution,
real-image accuracy, cross-sample acceptance or Office/human acceptance.
