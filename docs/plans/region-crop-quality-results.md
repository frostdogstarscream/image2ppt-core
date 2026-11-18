# Region crop quality experiments

The reference implementation is 056a356. Results below distinguish a complete historical slide, a replay of the crop stage and an isolated PowerPoint edit test.

| Experiment | Observed result | Scope |
| --- | --- | --- |
| Historical p08 delivered PPTX, rescored | Six duplicate pictures own 28.27% of structured content. Native objects own 39.64%. | Same file, changed measurement. |
| p08 crop stage on the source-backed empty scene | Thirteen seeds, seven protected by text, two skipped for unsafe movement, four independently movable crops. Native fraction 0, duplicate fraction 0, verified movable fraction 3.05%. Office RGB MAE 0.102 on a 0–255 scale. | Four move/scale and delete operations saved and reopened. All four deletion residuals passed. This input scene differs from the historical delivered slide. |
| Child graphic inside a native solid card | Original foreground, card after deleting the child, and canvas after deleting the card all matched the expected sampled colors, maximum channel error zero. | Synthetic fixture, actual PowerPoint save and reopen. Complex overlapping parents still need separate evidence. |

The p08 replay PPTX SHA-256 is `3d1d68b69f9b3489023db26e1080b113971111236755d6ce226377fe08ed0e6f`. Supply the actual Office checks.json to the scorecard with `--checks` to measure the verified movable fraction. A score without these checks leaves this metric unknown. Independent non-logo line recall needs a reference inventory.

For the real-page model comparison, submit the same two source images through OpenCode using the baseline model, model-only switch and revised pipeline. Record terminal status and whether each job produced a deliverable before comparing editability. Early baseline OCR environment retries affect elapsed-time comparisons.

## Held-out crop-stage replay

Page 07 was held out from crop tuning. The 4090 PP-DocLayout worker returned 54 regions. Production local OCR supplied 37 lines and layout supplied 24 pictorial seeds. Starting from a source-backed scene, 21 movable crops passed actual Office move/scale, delete and residual checks. They own 16.57% of structured source content. Native text reconstruction was outside this stage. Source SHA-256 is `7eb253667f654d9de95e81629dc3dbdbb7aecf37989cc300295bdcdc90ecd225` and the compiled PPTX SHA-256 is `60ddbc660df902ad204d5fe4ae94179b11f356e04366c5132aa2ea36b3c56497`.
## Controlled comparison on p08

Both runs used the same empty input scene, source hash and PP-DocLayout result. On commit 056a356, the crop stage produced 12 pictures (10 movable, two in-place). Together they covered more than half of each of nine source OCR line boxes. Verified movable pictures covered 42.39% of structured content and duplicates covered 2.37%. The revised crop stage returned four movable pictures covering 3.05%, zero duplicates, and zero OCR lines more than half covered. Both runs had zero native text because they exercised only cropping. Four rejected image regions included captions near their lower edges. The experiment measures preservation of a later text repair opportunity, not restoration of those captions or extraction of their photos.

## Business editing reference

The accepted refined semi-automatic reference contains 93 native text fields on page 1 and 74 on page 2. The six predefined actions (KPI replacement, role box with label movement, software screenshot deletion, independent project name change, table field change and card recolor) each changed pixels inside its target area after a PowerPoint save and reopen; no pixel outside the padded target area changed by more than eight channel levels. These counts refer to the reference deck. The automated jobs are evaluated separately.
## Two-page OpenCode runs

All three controls used the same 64-turn, 2,700-second budget, two source pictures, local PowerPoint and 4090 LayerD access. The baseline needed an early local OCR environment correction. The model-only run reused the hash-bound production OCR results. The model calls were live, and execution overlapped, so these are case studies rather than a latency benchmark.

| Code and model | Terminal outcome | Page 1 committed | Page 2 committed |
| --- | --- | --- | --- |
| 056a356, qwen3.8-max | Time budget exhausted | 0 | 0 |
| 056a356, hy4 | Time budget exhausted | 28 native text, 23 pictures including 7 duplicates | 0 |
| 57037fb, hy4 | Time budget exhausted after two-page composition | 28 native text, 14 pictures, 0 duplicate pictures | 14 native text, 0 pictures |

The frozen Hy4 run composed a two-page PPTX with SHA-256 `3b34b23de2e3361136b7ed2eda645c1c0feda591d4b5556faa163d4f021de7c1`. OpenCode did not finalize the job. Office candidate edit checks and existence of a compiled file are separate from end-to-end acceptance. On page 1 the model-only run and frozen run each reproduced 28/93 independent reference fields. Duplicate pictures owned 4.07% of structured content in the model-only page and 0% in the frozen page. On page 2 the frozen run reproduced 8/74 reference fields. The complete source content can also be present in the backing; field recall measures native wording at its source location.

A separate development run experienced candidate code-hash drift and an interrupted model compaction. It was excluded from the controlled comparison. Its validated page 1 candidate had 27 native textboxes. Each frozen page had evidence of local OCR and LayerD decomposition. The tests do not demonstrate human or WPS acceptance.

An independent whole-deck PowerPoint recheck of those exact bytes passed: 112 move/scale or deletion records, 84 short or equal-length text replacements, and 56 object residual checks. The first whole-deck check had an infrastructure error. The independent rerun used a new directory after all experimental hosts ended. The Office manifest now excludes transient Office owner files.

| Frozen page | Native share of structured content | Verified movable picture share | RGB MAE / 255 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 30.55% | 1.99% | 8.230 | 0.8625 |
| 2 | 12.81% | 0% | 3.383 | 0.9507 |

All six predefined business edits were checked on the complete frozen sample. Each returned NO_TARGET_OBJECT. The reference passed all six. The next experiment should measure whether submitting high-priority editable fields and their parent cards on both pages before full-page reconstruction improves these six outcomes within the same budget.
