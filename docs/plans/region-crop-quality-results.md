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
