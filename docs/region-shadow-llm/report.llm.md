# Region rebuild shadow mode

- regions: 8
- decided: 8
- agreement: 7/8 = 0.875

| Region | Page | Type | Agent | Human | Agree | Confidence | Evidence |
|---|---|---|---|---|---|---:|---|
| P5-example-058-dashboard | P5 | ui_mockup | hybrid | hybrid | yes | 0.85 | Dashboard UI mockup with rounded cards, soft shadows, and gradient-style status chips; Microtext fraction 0.26 with 231 text components and median height 12px; Custom icons (checkm |
| P6-funds-analysis-chart | P6 | dashboard | retain_module | retain_module | yes | 0.92 | Central radial network diagram with 7 connected nodes and glowing connector lines forming non-native geometry; Median text height of 4px with 87.76% microtext fraction across 49 te |
| P7-clue-network | P7 | network | retain_module | retain_module | yes | 0.95 | Complex radial network graph with 8 nodes connected by curved glowing edges to a central person icon; Numerous microtext labels including category names, relationship descriptions, |
| P8-petition-example | P8 | table | hybrid | retain_module | no | 0.85 | Blue header bar with white Chinese title text '示例：信访智能分析结果'; White card panel containing six labeled rows with icons, text values, and orange '中风险' status badge; Row labels include |
| P9-image-020-report-ui | P9 | ui_mockup | retain_module | retain_module | yes | 0.92 | Entire image is a UI mockup with sidebar navigation, document preview panel, and feature cards with icons; Contains 144 text components with 28.47% microtext fraction below readabl |
| P10-vehicle-risk-example | P10 | photo_map | retain_module | retain_module | yes | 0.95 | Map background with street grid and location markers showing route from company to restaurant/hotel; Red alert popup with warning icon and timestamp contains microtext-sized elemen |
| P11-rural-three-assets | P11 | photo_diagram | retain_module | retain_module | yes | 0.95 | Aerial photograph of rural village landscape fills entire background with continuous tonal imagery; Radial network diagram with six nodes connected by dashed and solid arrows to ce |
| P12-industry-risk-bars | P12 | chart | native | native | yes | 0.92 | Horizontal bar chart with five category labels and percentage values aligned to bar ends; Uniform bar styling with gradient fills reconstructable as native bars; Title text in dark |

## Disagreements

- P8-petition-example: agent=hybrid, human=retain_module

## Measured features

| Region | Area px | Aspect | Colors | Edge density | Median text px | Microtext |
|---|---:|---:|---:|---:|---:|---:|
| P5-example-058-dashboard | 223357 | 1.389 | 536 | 0.06996 | 12.0 | 0.2597 |
| P6-funds-analysis-chart | 394416 | 2.1134 | 749 | 0.07862 | 4.0 | 0.8776 |
| P7-clue-network | 501832 | 1.4128 | 917 | 0.06906 | 5.0 | 0.703 |
| P8-petition-example | 167475 | 1.1299 | 321 | 0.08684 | 14.0 | 0.3744 |
| P9-image-020-report-ui | 220896 | 1.2764 | 350 | 0.04302 | 13.0 | 0.2847 |
| P10-vehicle-risk-example | 249247 | 3.1566 | 1186 | 0.10395 | 4.0 | 0.8095 |
| P11-rural-three-assets | 577220 | 1.3053 | 2098 | 0.19038 | 5.0 | 0.7597 |
| P12-industry-risk-bars | 186420 | 2.7577 | 217 | 0.05287 | 16.0 | 0.3529 |

## Provenance

- prompt_version: region-decision/1
- capabilities_sha256: 88e09e23865ea83df7ba401021089aa4ae627e381c8af2e3250362bd87b7078e

| Region | Status | Transport | Response sha256 |
|---|---|---|---|
| P5-example-058-dashboard | VALIDATED | RECEIVED | c7b2564bc8181579 |
| P6-funds-analysis-chart | VALIDATED | RECEIVED | ca59dc05a4342ea5 |
| P7-clue-network | VALIDATED | RECEIVED | 2bc11575471561d1 |
| P8-petition-example | VALIDATED | RECEIVED | 6dc85f3ca2d9bf3e |
| P9-image-020-report-ui | VALIDATED | RECEIVED | 204561826a9d3390 |
| P10-vehicle-risk-example | VALIDATED | RECEIVED | 91e0a5a411942ee3 |
| P11-rural-three-assets | VALIDATED | RECEIVED | 2cc4df86e4875b5e |
| P12-industry-risk-bars | VALIDATED | RECEIVED | c587a9e14fc61865 |

## Validation

- valid: True

## Apply report

- status: VALIDATED
- validated: 8/8
- human_acceptance: false
- shadow_only: true
