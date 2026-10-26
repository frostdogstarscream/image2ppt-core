# Local candidate build and review

The conversation authors the candidate and reviews source versus final-file evidence.
The scripts never call a model, automatically approve a candidate, or replace a baseline.
Historical records remain immutable. A successful technical review is `VERIFIED` with
`human_acceptance=false`; Office/WPS editing and human acceptance are separate.

## Prepare, build, review

Run from the complete project checkout with the core and review dependencies installed:

```bash
python -m pip install -r requirements-review.lock
python scripts/candidate_workflow.py prepare --plans plans.json --human-decisions human.json --baseline-ir baseline/slideir.json --region P5-example-058-dashboard --output outputs/trials
python scripts/candidate_workflow.py build --candidate outputs/trials/candidate-ID --ir proposal/slideir.json --mapping proposal/mapping.json
python scripts/candidate_workflow.py review --candidate outputs/trials/candidate-ID --revision r000 --record review.json
```

`prepare` requires a complete authoritative scope annotation (the six fields in
`region-human-decisions/0.2`). It snapshots the baseline and binds the original input
files by hash. A changed original input requires a new candidate. Old three-field
decisions are still valid for shadow scoring, but must be annotated before trial builds.
`apply-llm → validate` now compares the full normalized human decision. An old plan
without the current authoritative scope reports **reapply saved responses**; no new
model call is required.

Only baseline nodes wholly contained in the selected region may change. An explicitly
approved adjacent title/icon can be included using repeated `--context-node NODE_ID`;
the resulting boundary and mutable IDs are recorded in the manifest. Other node
properties, relative stacking, owned evidence/assets, and groups remain frozen.
Insert a replacement region as a contiguous block without splitting frozen groups.

The proposal must have a current `authoring.json` (core `AuthoringRecord`, including
tool, model ID, timestamp and exact IR hash). The mapping has exactly two sections:

```json
{
  "native": {"section_title": ["title-node"]},
  "retained": {"dashboard_body": ["dashboard-asset"], "icons": ["title-icon-asset"]}
}
```

Native mappings reference text/shape/table/chart nodes; retained mappings reference
assets used by region image nodes. Every region node must be mapped, required edit
targets and retained parts must be present, and native targets must be permitted by
the human scope. Mapping checks references and types, not the semantic meaning of pixels.
An image icon is retained even if a historical report called it native.

Each build compiles both baseline and candidate with the same runtime using the shared
core, reimports the actual PPTX, and saves source/preview/cropped comparison/difference
images, object maps, coverage, metrics and content hashes. Source ROIs use source-image
pixels and stay fixed between revisions. Compiler work uses a short `.build/cw-*` path
for Windows; completed evidence is copied into the candidate revision. The short-path
core run is also retained. No proprietary runtime is bundled or downloaded.

Review JSON uses the `artifact_binding` returned by this exact build:

```json
{
  "schema": "CandidateReview/1",
  "artifact_binding": "COPY_THE_RETURNED_SHA256",
  "reviewer": "current Codex conversation",
  "issues": [{
    "object_ids": ["title-node"],
    "kind": "style",
    "source_evidence": "evidence/candidate-source.png",
    "rendered_evidence": "evidence/candidate-rendered.png",
    "suggestion": "Match the source title weight; leave the icon and dashboard unchanged."
  }],
  "unresolved": [],
  "summary": "Title needs a local weight correction.",
  "human_acceptance": false
}
```

Allowed issue kinds: `text`, `style`, `position`, `crop`, `scope`, `fidelity`.
Issues must name existing object IDs and images from the bound packet. Review records
are immutable, and cannot be reused for another revision even if its rendered pixels
are identical. Review the last successful build before submitting a correction.
Create a new proposal directory, edit only identified objects, refresh authoring, and
invoke `build` again. Two corrections are allowed after the initial successful build;
the workflow enforces the last review's object IDs as the correction allowlist, freezing
other nodes and their assets/evidence. Keep IDs stable; restructuring requires a new
candidate with a fresh scope record.
failed build attempts are retained but do not consume this visual-correction budget.
Remaining issues at the limit stay `REVIEW_REQUIRED`.

`report.md` summarizes attempts; `records.json` can be checked with the existing
`region_candidates.py validate/verify-files/report` commands using the candidate directory
as the artifact root. Failure-only runs have no trial records yet. Scores rank evidence;
they never cause automatic acceptance, replacement or whole-region fallback.

For a replay of a historically rejected proposal, keep the historical human file intact.
Use a separate, explicitly authorized trial annotation, label it `calibration`, and state
that it permits only the experiment. It must not change the delivery decision or claim
that a previous veto was reversed.

## ElementPlan/3 and text styles

New `/3` raster objects require both `source_crop_bbox` and `dest_bbox`, with no ambiguous
`bbox`. Both use source-page coordinate units. Cropping, source text ownership and duplicate
source-pixel checks use the first box; SlideIR image placement uses the second. Masks retain
their existing semantics. Shapes/text retain `bbox`, and `/3` inherits `/2`'s font contract.

Legacy `/1` and `/2` raster boxes are normalized **before patching** into both explicit
boxes. A legacy `ElementPatch/1` `bbox` edit is a destination-only edit; a recrop must name
`source_crop_bbox`. The saved new artifact retains its legacy schema/font contract, so
normalizing old coordinates does not invent font-identification evidence. Newly authored
`/3` plans must supply the font evidence required by `/2`. Unmentioned objects do not move.

The compiler resolves color and bold defaults per rich-text run. Global layout/font
properties no longer overwrite explicit run colors or `bold:false`. Plain text keeps its
existing defaults. `scripts/check_rich_text.mjs` verifies exported final OOXML and emits
a reimported preview; its synthetic source is not a fidelity benchmark.

## Verification boundaries

CI runs the selected planning, candidate, coordinate and workflow tests with no OCR model,
GPU, network calls or Artifact Tool. Local compiler changes additionally require:

```powershell
./image2ppt.ps1 test
./image2ppt.ps1 demo
node scripts/check_demo.mjs <returned-synthetic-ir>
node scripts/check_rich_text.mjs
```

Use the configured/bundled Node executable if `node` is not on PATH. Review real P5/P8/P12
previews and local crops separately. Private sources and generated trial packets remain
under ignored `outputs/` or `.build/`; they are not public test fixtures.
