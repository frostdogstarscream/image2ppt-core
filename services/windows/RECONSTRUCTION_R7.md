# r7 object restoration

This deterministic entry uses the existing fixed compiler. It makes no model API
calls and does not load skills. Inputs are a complete r6 TypographyBatch/1 file,
its sibling font-manifest.json, all referenced scenes and source-bound text masks.
The r6 source/wording/paragraph/runs/assets remain immutable.

```bat
image2ppt-local.cmd rebuild reconstruct-r7 --input <r6-batch.json> --output <new-run> --mode shadow
image2ppt-local.cmd rebuild reconstruct-r7 --input <r6-batch.json> --output <same-run> --mode gated --pages 02 08 03
image2ppt-local.cmd rebuild reconstruct-r7 --input <r6-batch.json> --output <same-run> --mode gated --resume <same-run>
image2ppt-local.cmd rebuild qa-r7 --run <same-run>
image2ppt-local.cmd rebuild qa-r7 --run <completed-parent-run> --output <new-independent-qa-run>
```

ObjectRegistry/2 explicitly adapts legacy xywh into xyxy. Unknown masks are
retained as UNKNOWN and never authorize edits. Reviewed corrections have distinct
visible, repair, protection and inspection masks; evidence ownership remains unique.
Only source-reviewed page 02/08 missing elements are regenerated. Other objects
remain in the accepted r6 subscene. Native candidates are rendered first; at most
one source-shaped transparent PNG alternative is tried for rejected elements.
Raster label repairs fill source text holes from blue-only samples and never embed
the original white text. All unchanged assets remain byte-identical.

Each transaction snapshots the complete parent/candidate scene. All required gates
must explicitly PASS before atomic publication. Failed attempts restore the complete
parent, not a generic source crop or r3 card. Source/code/fonts/runtime changes reject
resumption. Original delivery files and failed attempts remain available.

QA without `--output` remains strict: code and all frozen inputs must match the
run manifest. After correcting a validator, use an explicit, previously absent
`--output` directory outside the parent run. This permits only code fingerprint
changes; source/baseline data, scenes, assets, fonts, runtime, masks, relation specs,
previews and checkpoint receipts are still verified against the parent freeze.
The composed deck hash and all twelve ordered IR hashes must match its compiler
receipt. No candidate reconstruction or model call is repeated, and the original
PPTX, manifest and reports are not rewritten. The new QA run saves current validator
fingerprints, byte copies of the parent manifest/batch, and `parent-link.json`
with parent manifest/batch/deck hashes and original reconstruction code. Its
delivery copies the unchanged parent deck and includes this provenance chain.

Final QA separates export consistency, source-based text residue, key-element
relations and actual edit probes. Normal/contrast OCR absence alone is not proof.
Unreviewed pixel ownership and missing independent clean-plate evidence remain
REVIEW_REQUIRED. Automated probes do not establish Office/WPS/human acceptance.
Full deliveries always contain twelve pages; failed corrections retain their baseline
page and remain unresolved in the report. Outputs and business inputs stay out of Git.
