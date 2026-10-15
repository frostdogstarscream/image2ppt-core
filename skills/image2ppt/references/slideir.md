# SlideIR 0.1

This protocol is a single-page reconstruction contract. It accepts `text`, `shape`, and `image` nodes, plus real groups; native `table` and `chart` nodes are accepted only in builds whose `doctor` reports `native_tables` / `native_charts` as supported, and are explicit image fallbacks everywhere else. The compiler rejects unsupported types rather than silently skipping them.

```json
{
  "schema_version": "0.1",
  "source_image": "source/page.png",
  "source_sha256": "<sha256 of source/page.png>",
  "canvas": {"source_px": [1920, 1080], "target_px": [1280, 720]},
  "background": {"color": "FFFFFF"},
  "review_status": "needs_review",
  "review_notes": ["字体为候选；请核对关键数值。"],
  "evidence": [
    {"id": "text-001", "kind": "text", "bbox": [120,80,1500,100], "text": "业务规划"}
  ],
  "assets": [],
  "groups": [],
  "nodes": [
    {
      "id": "title", "type": "text", "bbox": [120,80,1500,100], "z": 10,
      "evidence_ids": ["text-001"], "text": "业务规划",
      "font": {"latin": "Arial", "ea": "Microsoft YaHei", "size_pt": 32},
      "bold": true, "color": "222222", "align": "left", "wrap": false
    }
  ]
}
```

## Geometry and text

- `bbox` always means `[x, y, width, height]` in source-page pixels. All values must be finite; width and height positive. Nodes must stay within the page. For a horizontal/vertical line, use a thin positive rectangle rather than zero width/height.
- Source and target must have the same aspect ratio within one pixel of rounding. No silent stretching/cropping. Target pixels describe a 96-DPI presentation canvas; the resulting EMU size is `target_px * 9525`.
- Nodes require unique IDs and unique integer `z` values; ascending z means bottom to top. Source evidence can have overlapping bounding boxes.
- Text preserves `\n`. `font.size_pt`, `line_spacing_pt` are target points. `align` is `left`, `center`, or `right`. `wrap` defaults to false. Optional `runs` is a single-line array of `{text, bold?, color?}` with concatenated text exactly equal to `text`; multiline mixed runs are not supported yet.
- Each text node requires a text evidence item whose content equals `text`; changes to content require `text_override_reason`. An override is retained in the quality report and requires review.
- Shape `geometry` accepts `rect`, `roundRect`, `ellipse`, `line`, the flow-diagram presets `chevron`, `pentagon`, `rightArrow`, `leftArrow`, and the common presets `trapezoid`, `parallelogram`, `hexagon`, `diamond`, `triangle`. That is the whole list: the engine carries many more presets, but this pipeline only accepts what it has compiled and what `doctor` confirms the installed build still provides. `fill` is a six-digit RGB color or `none`, `line` is `{color,width_pt}` or omitted. Decorations recreated from a source are still evidence-backed shapes.
- Optional `corner_radius_px` is a source-pixel radius, converted with the page geometry. Optional `fill_gradient` is `{angle_deg,stops:[{offset:0,color:"FFF8FA"},{offset:100000,color:"FFF0F4"}]}`; offsets are strictly increasing from 0 to 100000. These are native editable shape properties.

## Tables

A `table` node is a native PowerPoint table. It is accepted only where `doctor` reports `native_tables`, and is an image fallback everywhere else.

- `{id,type:"table",bbox,z,evidence_ids,rows,font}` plus optional `column_tracks`, `header_row`, `banded_rows`. `rows` is the whole grid as a 2D array of strings or finite numbers; the row and column counts follow from it. A cell string is at most 2048 characters, the grid at most 2000 rows by 100 columns, and one page may hold at most 100 tables — the same bounds the delivered package is validated against, so an IR that validates here cannot produce a table the finalizer refuses.
- `font` is required and has the same shape as a text node's: `{latin,ea,size_pt}`. The engine would otherwise leave the cells on its own default family, which no build approves; declaring the font is the author's choice, and `ea` is the one applied.
- Optional `column_tracks` is one `{mode:"fr"|"fixed",value}` per column, matching the column count; `fixed` values are target pixels.
- A table node carries its source region as one `shape` evidence item. Cell text is not separately evidence-backed; `evidence` describes source content classes, and nothing compares a cell string to an evidence string.
- A table may be laid out taller than its `bbox`; that is reported as `native_table_overflow`, which is a review diagnostic and does not fail the build.

## Charts

A `chart` node is a native PowerPoint chart. It is accepted only where `doctor` reports `native_charts`, and is an image fallback everywhere else.

- `{id,type:"chart",bbox,z,evidence_ids,chart_type,font,categories,series}` plus optional `title`, `has_legend`, `legend_position`. `chart_type` is one of `bar`, `line`, `area`, `pie`, `doughnut`. `categories` is 1—1000 strings, `series` is 1—128 entries of `{name,values}`, and every series has exactly one value per category.
- `font` is required, for the same reason as a table: the chart's title, axis and legend text would otherwise inherit the theme's family, which no build approves. `ea` is the one applied.
- `title` is optional. An absent title produces no title element, which the delivered package accepts; but when a title is given it must be a real one — an empty string, or the engine's own placeholder, fails the build's title gate.
- Data is literal. The compiler declares the chart's owning slide to the finalizer and enables the literal-workbook snapshot, so the delivered file carries an engine-generated embedded workbook whose values reconcile exactly with the chart's caches. That workbook is a **snapshot of the values**: no source formulas are recovered or claimed, and the review notes should say so.
- Prefer `title` when the source chart shows one, and a review note when it does not — an untitled chart that came from a titled source is a silent fidelity loss.

## Evidence ownership and assets

- Inventory source content in `evidence`. Each item needs one owner: a node's `evidence_ids`, an image asset's `contains_evidence_ids` through its node, or `unresolved_evidence_ids`. Duplicate ownership and omitted evidence fail validation. An unresolved item requires `needs_review`; it is not permission to omit it from a delivered result.
- A text node cannot silently change source evidence wording. Native text plus an image that still contains that evidence is prohibited. This rule is not mechanically enforced: validation compares evidence IDs, never overlapping pixels, so a crop whose pixels still show a text a sibling native node re-emits will pass every check. Honouring it is the author's responsibility, and the quality report will not detect a violation.
- Image node: `{id,type:"image",bbox,z,asset_id,evidence_ids:[]}`. Asset: `{id,path,contains_evidence_ids:[],purpose:"module"|"cleaned_background",sha256}`. Paths are local relative paths confined to the run directory; URLs and absolute/traversal paths are rejected. SHA-256 is checked before embedding. A decomposed region is still `purpose:"module"`; its reduced extent is expressed by `contains_evidence_ids`, not by a new purpose.
- `background.asset_id` may reference a cleaned background. It is compiled beneath all nodes. Retained content ownership is validated. Declaring a background clean does not establish that it was successfully repaired; compare source and perturbation previews.
- Rebuilding a region natively is a trade, not a free improvement: a crop is pixel-exact and a rebuild is not. Measured across a 14-page restoration, the same regions scored 2.3—4.2 mean pixel difference as crops and 8.9—50.1 rebuilt, and no region did better rebuilt. Most of that gap is the text being re-set in a different face, so it is largely independent of how well the coordinates were measured. A region therefore keeps its native nodes only while its rebuilt drift stays at or below about 10 (`scripts/fidelity_gate.py`); above that it goes back to a crop, and a page legitimately ends up part native and part raster.
- Every asset must be used exactly once. A crop owns the evidence still visible in it, and `contains_evidence_ids` may be any non-empty subset of the evidence inside its `bbox` — it need not own everything the rectangle covers. That subset is how a decomposed region is expressed: the crop keeps only what must stay raster (a photo, an illustration, a logo, an icon, or a background that cannot be rebuilt) while sibling native text and shape nodes own the rest. One region may yield several crops, each `purpose:"module"`, each used once; a region is not one decision and need not produce one crop. No two owners may claim the same evidence ID. Crop from the preserved page; do not redraw logos or photos.
- Only PNG and JPEG are embedded. Convert other formats during preparation.

## Groups and review

- Group: `{id,member_ids:["card-background","card-title","card-body"]}`. Members must be consecutive in sorted z-order, each in at most one group. Nested groups are excluded from v0.1. Members retain absolute source coordinates; the PPTX postprocessor writes group transforms that preserve their positions.
- `review_status` is `needs_review` or `source_checked`. `source_checked` records an agent source comparison; it does not mean human acceptance or PowerPoint/WPS acceptance.
- A generated quality report records package checks, native object counts, image fallbacks, warnings, source hash, renderer identity and a separate `human_acceptance: false`.
- Do not store model-generated code in the IR. The compiler runs only the fixed project code and local asset processing.

