# Page scorecard baseline — 2026-11-17

Status: measurement baseline only. No product behavior changed. Not human, WPS or publication acceptance.

## Tool

`services/product/product/scorecard.py` (`PageScorecard/1`) and `scripts/scorecard.py`. The scorecard inspects the delivered PPTX package directly (OOXML, z order, picture alpha), independent of SlideIR/SceneGraph claims. Optional inputs: the actual Office render (`page-01.png`) and an independent local OCR inventory (same pinned RapidOCR/PP-OCRv4 manifest as production).

Visible-pixel ownership classes: `plate` (page-sized picture), `opaque_picture` (opaque picture smaller than the page: region crop or opaque container layer), `asset` (picture with transparency), `shape`, `table`, `chart`, `text` (frame area, not ink), `background`.

Metrics:
- `content_fraction`: share of source pixels with local structure (5x5 gray std > 6) owned by each class. `raster_plate_content_fraction` is the structured source content still delivered through the page raster; `editable_content_fraction` is everything else.
- `fidelity`: render vs source MAE, p95 absolute RGB error, SSIM (Gaussian 11x11), CIEDE2000 mean/p95/fraction > 10 (validated against Sharma 2005 reference pairs).
- `text`: OCR line/character recall by native text frames (overlap + wording), mean frame IoU. OCR is an independent automatic reference, not a human inventory; recall against complete page content remains unknown.

Synthetic tests: `services/product/tests/test_scorecard.py` (3 passed).

## Baseline on retained outputs

Sources: `outputs/benchmark-ai-workbench-8p/inputs/*` and `D:/Code/image2PPT/inputs/xuzhou-20261029/06.png`. Delivered files and renders: retained `result.pptx` / `page-01.png` in each output folder. Raw JSON: `outputs/scorecard-baseline/`.

| Run | Editable content | Plate content | Plate area | MAE | SSIM | ΔE00 p95 | OCR line recall |
|---|---|---|---|---|---|---|---|
| workbench-8p / p02 | 0.000 | 1.000 | 1.000 | 0.20 | 1.000 | 0.45 | 0.00 |
| workbench-8p-v2 / p02 | 0.363 | 0.637 | 0.898 | 4.34 | 0.926 | 3.06 | 0.36 |
| workbench-8p-v3 / p02 | 0.559 | 0.441 | 0.839 | 5.92 | 0.889 | 9.71 | 0.66 |
| workbench-8p-v3b / p02 | 0.616 | 0.384 | 0.817 | 6.99 | 0.874 | 15.31 | 0.73 |
| workbench-8p-v2 / p07 | 0.668 | 0.332 | 0.785 | 7.49 | 0.890 | 15.47 | 0.81 |
| workbench-8p-v3b / p08 | 0.436 | 0.564 | 0.877 | 4.25 | 0.923 | 3.30 | 0.88 |
| xuzhou 06 / m2 | 0.509 | 0.491 | 0.846 | 7.45 | 0.880 | 16.88 | 0.90 |
| xuzhou 06 / m3 (best) | 0.733 | 0.267 | 0.784 | 8.51 | 0.863 | 23.35 | 1.00 |
| xuzhou 06 / m4 | **0.933** | **0.068** | 0.414 | 8.85 | 0.842 | 23.13 | 1.00 |

## Observations

- Editability rose across revisions while render fidelity fell (MAE 0.2 for raster-only to 7–9 for the most editable runs). The largest ΔE00 contributions observed in xuzhou 06 come from native text rendering (title weight/family differs from source), not from backing residue; this is a visual observation, not a measured attribution.
- m4 xuzhou 06 leaves 6.8% of structured content in the page raster; the remaining plate area (41%) is mostly flat background.
- workbench p02/p08 still leave 38–56% of structured content in the plate; these are the primary targets for local region-crop fallback.
- The `opaque_picture` object in m4 is the award-card container layer (`a-c07`), not a region crop.
