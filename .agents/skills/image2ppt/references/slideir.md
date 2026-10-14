# SlideIR 0.1

This protocol is a single-page reconstruction contract. It accepts only `text`, `shape`, and `image` nodes, plus real groups. Tables/charts are explicit image fallbacks in this version. The compiler rejects unsupported types rather than silently skipping them.

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
- Shape `geometry` supports `rect`, `roundRect`, `ellipse`, `line`; `fill` is a six-digit RGB color or `none`, `line` is `{color,width_pt}` or omitted. Decorations recreated from a source are still evidence-backed shapes.
- Optional `corner_radius_px` is a source-pixel radius, converted with the page geometry. Optional `fill_gradient` is `{angle_deg,stops:[{offset:0,color:"FFF8FA"},{offset:100000,color:"FFF0F4"}]}`; offsets are strictly increasing from 0 to 100000. These are native editable shape properties.

## Evidence ownership and assets

- Inventory source content in `evidence`. Each item needs one owner: a node's `evidence_ids`, an image asset's `contains_evidence_ids` through its node, or `unresolved_evidence_ids`. Duplicate ownership and omitted evidence fail validation. An unresolved item requires `needs_review`; it is not permission to omit it from a delivered result.
- A text node cannot silently change source evidence wording. Native text plus an image that still contains that evidence is rejected.
- Image node: `{id,type:"image",bbox,z,asset_id,evidence_ids:[]}`. Asset: `{id,path,contains_evidence_ids:[],purpose:"module"|"cleaned_background",sha256}`. Paths are local relative paths confined to the run directory; URLs and absolute/traversal paths are rejected. SHA-256 is checked before embedding.
- `background.asset_id` may reference a cleaned background. It is compiled beneath all nodes. Retained content ownership is validated. Declaring a background clean does not establish that it was successfully repaired; compare source and perturbation previews.
- Every asset must be used exactly once. A source crop of a complex module owns all evidence it still contains. Crop from the preserved page; do not redraw logos or photos.
- Only PNG and JPEG are embedded. Convert other formats during preparation.

## Groups and review

- Group: `{id,member_ids:["card-background","card-title","card-body"]}`. Members must be consecutive in sorted z-order, each in at most one group. Nested groups are excluded from v0.1. Members retain absolute source coordinates; the PPTX postprocessor writes group transforms that preserve their positions.
- `review_status` is `needs_review` or `source_checked`. `source_checked` records an agent source comparison; it does not mean human acceptance or PowerPoint/WPS acceptance.
- A generated quality report records package checks, native object counts, image fallbacks, warnings, source hash, renderer identity and a separate `human_acceptance: false`.
- Do not store model-generated code in the IR. The compiler runs only the fixed project code and local asset processing.

