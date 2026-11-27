# Saved plans before layered segmentation

This experimental path is opt-in and leaves the default route unchanged. It
consumes a previously completed `SegmentationPlan/1` job before the worker's OCR
and segmentation stages. Loading it makes **zero new model requests**. A normal
`--model-config` can still enable the separately budgeted quality-loop reviews.

Create a UTF-8 manifest:

```json
{
  "schema": "LayeredPlanInputs/1",
  "plans": ["absolute/or/manifest-relative/path/to/completed-plan-job"]
}
```

The manifest must cover exactly the input source hashes, without duplicate
plans. Each job requires `source.png`, `binding.json`, `request.json`,
`response.json`, `plan.json`, and `result.json` (benchmark) or `job.json`
(standalone planner). Request and response hashes must match their receipts;
the embedded request image must match the input bytes; parsed response and saved
plan must agree. Source bundles are copied into the new output job before any
worker invocation. No response or box is silently repaired in that bundle.

```powershell
python scripts/layered.py --inputs page.png --output outputs/experiment `
  --worker-config .build/layered-worker.json --quality-policy quality-loop `
  --segmentation-plans plans.json --max-extra-vision-calls 0 --max-repair-rounds 1
```

The wrapper accepts `./image2ppt.ps1 layered -InputPath page.png
-QualityPolicy quality-loop -SegmentationPlans plans.json`. Both entry points
reject planning with the legacy policy. Omitting model configuration keeps this
comparison free of model API calls.

Only childless `keep_pixels` regions guide raster segmentation. A module with
children is a logical container, never a second crop. A childless module marked
`keep_pixels` can represent a complex illustration. Text, native-shape, table,
and chart candidates continue through the existing measured detectors; the plan
does not rewrite OCR text or promise native chart reconstruction.

After OCR and text removal, candidate boundaries grow outward to nearby ink and
complete intersecting OCR lines. Growth beyond 24 pixels at an OCR boundary,
excessive area, intersections with measured tables, and overlapping raster plans
fall back to ordinary segmentation. Accepted regions are removed from the image
fed into SAM and candidate detectors, then installed as separate rectangular
cutouts from the text-free image. Detector candidates touching the temporary
filled surface are discarded: that synthetic surface is not source evidence
for a new native rectangle. Covered raster fragments have their alpha
punched out so the pixels have one owner. Enclosing native cards can remain
underneath. The normal residual-ink recovery remains active. A second object
planning model call is skipped when a saved plan was applied.

`perception/<page>/plan-execution.json` records raw and refined pixel boxes,
fallback reasons, and installed element IDs. `plan-boundaries.png` shows red raw
boxes and green accepted/orange fallback boxes. The saved plan does not guarantee
complete image edges; these are bounded heuristics requiring visual comparison.

Acceptance still uses actual exported PPTX re-import previews, source-bound
lineage, OCR comparison, and hide/move probes. A crop is editable only by moving
or resizing; its internal pixels are not vector objects. `REVIEW_REQUIRED` is
not Office/WPS or human acceptance. Compare the same sources, budgets and model
settings with and without the manifest, preserve both jobs, and run
`scripts/check_automatic_delivery.py --job <job>` on each.
