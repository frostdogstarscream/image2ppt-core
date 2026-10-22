# Region candidate trials

Metrics rank candidates only. A human decision, not a metric, accepts or rejects one.

| Candidate | Page | State | Baseline MAE | Candidate MAE | Delta | Outcome |
|---|---|---|---:|---:|---:|---|
| p5-title-native-dashboard-module | P5 | ACCEPTED | 13.817005984674758 | 13.140721847367914 | -0.676 | ACCEPTED: Title stays editable and the dashboard body keeps its original visual quality; the native KPI variant was rejected on fidelity. |
| p8-petition-hybrid-sample | P8 | REJECTED | 13.50081566828868 | 14.938482721391443 | 1.438 | REJECTED: Fidelity-first goal: the module image is visually closer than the hybrid; the sample is kept for a future editability-first goal. |
| p12-native-risk-bars | P12 | ACCEPTED | 16.097389522497192 | 17.19031405559595 | 1.093 | ACCEPTED: Titles, bars, categories and percentages are editable; values can be checked against the source at a small pixel-fidelity cost. |

## p5-title-native-dashboard-module

- region: P5-example-058-dashboard
- routing decision: hybrid
- baseline: previous native KPI decomposition
- candidate: native title + one dashboard module image
- edit goals: section_title
- target fidelity: pixel
- outcome: ACCEPTED — Title stays editable and the dashboard body keeps its original visual quality; the native KPI variant was rejected on fidelity.
- baseline note: The whole dashboard was split into native KPI cards, table and chart. It had more editable objects but lost light borders and distorted icons, so it was not the visual baseline for acceptance.
- contamination: The title strip was inpainted with background column interpolation. Manual review found no visible ghosting in the final preview.
- budget: nodes=124, text=56, images=27, shapes=96
- budget warning: 56 text nodes exceeds the 52-text-box warning threshold; the final slide still exported and rendered.

| Edit target | Coverage | Evidence |
|---|---|---|
| section_title | covered | Native title and title icon remain editable; calibrated to 14pt bold and the source ink box. |
| dashboard_body | retained | KPI cards, chart and table stay in one 557x401 module image, as requested by the human plan. |

History: PROPOSED -> VALIDATED_STRUCTURE -> TRIAL_BUILT -> REVIEW_REQUIRED -> ACCEPTED

## p8-petition-hybrid-sample

- region: P8-petition-example
- routing decision: hybrid
- baseline: retained petition module in the final deck
- candidate: hybrid single-page sample: native table and text, icon strip retained
- edit goals: section_title, table, values, status_chips
- target fidelity: pixel
- outcome: REJECTED — Fidelity-first goal: the module image is visually closer than the hybrid; the sample is kept for a future editability-first goal.
- baseline note: Current fidelity-first plan: the table, letter excerpt, icons and status pill remain one module image. Page MAE 13.501 is exact; region MAE 8.6 is the approximate region-crop comparison recorded in the archived report.
- candidate note: Page MAE 14.938 is exact from the single-page quality report; region MAE 17.7 is the approximate region-crop comparison recorded in the archived report.
- contamination: No crop contamination observed; the table value area is flatter than the source, and separators/icons differ by roughly 1-2px.
- budget: nodes=164, text=78, images=32, shapes=132
- budget warning: 78 text nodes exceeds the 52-text-box warning threshold.
- budget warning: 32 images sits at the observed export limit; the candidate used one icon-strip asset to avoid 33.

| Edit target | Coverage | Evidence |
|---|---|---|
| section_title | covered | Native title, non-bold and repositioned to match the source. |
| table | partial | Rows and labels are editable, but the value area is flatter than the source and separator positions differ by 1-2px. |
| values | partial | Five label/value rows are editable; remaining visual differences are in backgrounds and separators. |
| status_chips | covered | The orange medium-risk pill is a native rounded rectangle and editable text. |

History: PROPOSED -> VALIDATED_STRUCTURE -> TRIAL_BUILT -> REVIEW_REQUIRED -> REJECTED

## p12-native-risk-bars

- region: P12-industry-risk-bars
- routing decision: native
- baseline: retained horizontal-bar module image
- candidate: native title band, five gradient bars, labels and percentages
- edit goals: section_title, bars, category_labels, percentage_labels
- target fidelity: balanced
- outcome: ACCEPTED — Titles, bars, categories and percentages are editable; values can be checked against the source at a small pixel-fidelity cost.
- baseline note: The bar chart remained a module image before the native trial.
- contamination: No module crop remains; bar lengths and five percentage values are native and can be checked against the source.
- budget: nodes=121, text=49, images=33, shapes=88
- budget warning: Image count is 33, above the 32-image guidance; export and render succeeded for this page.
- budget warning: This is shape-level bar reconstruction, not a data-driven native chart with a workbook.

| Edit target | Coverage | Evidence |
|---|---|---|
| section_title | covered | Dark title band and title text are native gradient/shape and editable text. |
| bars | covered | Five horizontal bars are native rectangles with two-stop gradients; endpoints follow the source. |
| category_labels | covered | Five category labels are editable text. |
| percentage_labels | covered | Five percentage labels are editable text and can be checked against the source values. |

History: PROPOSED -> VALIDATED_STRUCTURE -> TRIAL_BUILT -> REVIEW_REQUIRED -> ACCEPTED
