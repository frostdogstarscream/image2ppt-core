# Region crop fallback and layout evidence — 2026-11-17

Status: mechanism verified on synthetic and real slides with actual PowerPoint checks. **Not** a full-pipeline acceptance: the real-slide runs below start from a raster-only baseline (no MoMA text/asset/surface reconstruction), so they prove crop mechanics, layout seeding, Office editing behaviour and visual equivalence only.

## What changed

- `product/recovery/region_crop.py` (shared r10/runtime): small-first opaque region crops. Seeds are objects that did not become editable plus pictorial layout regions. Seeds grow only to connected visible content (bounded 48 px) and never cut a remaining text line; continuing content (texture, page-wide rules) keeps the small seed.
  - Crop pixels equal the source appearance except inside committed foreground owners, where the verified backing is used, so no old ink is duplicated beneath edited text.
  - `MOVABLE`: isolated from committed objects and an independent lateral/surface backing fits the ring (error ≤ 4). The rectangle's pixels move into the crop, the plate receives the backing, and a normal owner receipt drives actual-deletion residual checks.
  - `MOVE_UNSAFE`: the plate keeps identical pixels. Nothing is erased. Residual status is `NOT_APPLICABLE` (accepted only for this explicit case via `residual_accepted`).
- `parallel_rebuild.adapt`: crops render above backings/containers and below editable foreground.
- r10: request `region_fallback: plate|crop` (default `plate`, unchanged behaviour).
- `product/layout_evidence.py` + `services/layout/worker.py`: out-of-process PP-DocLayoutV3 (PaddleOCR 3.7 / Paddle 3.3.1, weights from ModelScope, cached by source/model hash). Pictorial labels (`image`, `header_image`, `chart`, `seal`, `table`) that are still mostly in the plate become crop seeds. Labels are hints only.
- Scorecard: `duplicate_picture` class for opaque pictures whose pixels still exist unchanged in the plate (not counted as editable).

## Checks

- Product suite: 356 passed. Protocol suite (`./image2ppt.ps1 test`): 39 passed.
- Synthetic Office run: MOVABLE crops → PowerPoint PASS with residual PASS from actual deletion; MOVE_UNSAFE crops → PowerPoint PASS, residual NOT_APPLICABLE; render MAE 0.0 in both.

## Real slides (raster-only baseline → layout seeds → crops → compile → PowerPoint 16)

| Slide | Layout regions | Crops (MOVABLE / MOVE_UNSAFE) | Office | Render MAE | ΔE00 p95 | Movable crop content |
|---|---|---|---|---|---|---|
| workbench 08 | 33 | 10 / 2 | PASS | 0.069 | 0.00 | 42.4% |
| workbench 02 | 49 | 17 / 0 | PASS | 0.181 | 0.38 | 10.2% |
| China Mobile 06 | 103 | 29 / 5 | PASS | 0.183 | 0.42 | 11.7% |

PP-DocLayoutV3 on CPU: ~17 s load, ~2.6 s per 1672×941 slide. On these slides it found photos with burnt-in captions, the chart block, the brand logo (`header_image`) and most icons; it missed some icons/labels and does not detect cards (expected for a document model).

Local artifacts (outside Git): `D:/e2e-rc/{p08,p02,cm06}/` (pptx, Office report, owner previews), `outputs/layout-probe/v3/`.

## Not yet done

- Runtime `batch_reconstruct_page` integration (waits for the in-progress phase-0 commit on `feat/hybrid-opencode-pipeline` to avoid conflicting edits).
- Full r10 run with `region_fallback=crop` on real pages (needs MoMA/LayerD services).
- RMBG-2.0 matting candidates; 4090 deployment of the layout worker.
