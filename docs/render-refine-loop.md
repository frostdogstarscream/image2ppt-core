# Render–compare–patch loop (PR-D)

`scripts/render_refine_loop.py` is the outer closed loop for high-confidence
text. It compiles a calibrated SlideIR with the pinned Artifact Tool, reimports
and renders the final PPTX, compares each eligible text node against the source
pixels, patches only its origin and font size within a small budget, compiles
again, and keeps a patch only when the re-rendered ink and the whole page agree.

It never edits the input IR, `submitted-plan.json` or the recorded authoring
revision. The result is a new directory with the patched IR, an authoring
sidecar bound to that IR, a final preview/PPTX and `render-loop.json`.

## Usage

```bash
python scripts/render_refine_loop.py \
  --ir runs/<page>/calibration/calibrated/slideir.json \
  --calibration runs/<page>/calibration/calibrated/calibration.json \
  --output runs/<page>/render-loop \
  --rounds 2 --max-shift 2 --max-size-change .02
```

- **Eligible nodes**: typography rows with `applied: true` and `font` in
  `applied_fields`. Without a calibration file, only text nodes with a
  `source_text_bbox` are eligible.
- **Patch search**: translations and scale factors are scored directly on the
  fixed source region by MAE, not by glyph centroids. The winning patch is
  applied to the IR and verified by another real compile/render.
- **Per-node acceptance**: ink MAE must improve by at least 0.2 and must not
  regress by more than 2%.
- **Page guard**: if the whole-page MAE after the candidate compile is worse
  than before by more than 0.05, the entire round is reverted. The page metric
  is a guard, not the optimization target.
- **Budget**: at most 3 rounds; a round with no patch, no accepted node or a
  failed page guard ends the loop immediately. No improvement is a valid result.

## What it changes

Only `bbox[0:2]` and `font.size_pt` of eligible text nodes. It does not change
colours, fills, runs, images or geometry of other nodes. `insets` and
`vertical_anchor` are available to upstream calibration but this loop leaves
them alone.

## Observed results

- Synthetic page with a deliberately offset text box: page MAE **0.575 →
  0.160**; the accepted patch was `dx=-8, dy=-4`, and the second round found
  nothing further.
- page05 calibrated IR: the page guard rejected every proposed round
  (15.31 → 15.62 local proxy), so the output is byte-identical in layout to the
  input. This is the intended safe outcome: the loop does not ship a change that
  only improves its own node proxy.

## Limits

The preview comes from reimporting the final PPTX with the Artifact Tool, not
from PowerPoint/WPS. `actual_render_verified` and `human_acceptance` stay false.
The loop does not prove editability in Office, and it does not replace visual
review. It is a bounded correction layer, not an acceptance gate.
