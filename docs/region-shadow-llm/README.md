# LLM region planner shadow results

This directory records the sanitized 2026-10-22 shadow-mode run for the LLM-backed
region planner.

The reported 7/8 is a three-way routing-label agreement, not a concrete-plan
agreement. The human labels are six `retain_module`, one `hybrid`, one `native`,
so an always-retain baseline scores 6/8 on this sample. P5 has the same `hybrid`
label but a different native scope from the human plan; see the main report.

`human-decisions.json` uses `region-human-decisions/0.2`. Besides the routing
label, each explicitly annotated region records:

- `annotation_level`: `proxy`, `explicit`, `trial_veto`, or `calibration`;
- `retain_scope`: `none`, `partial`, or `whole_region`;
- `fidelity_target`: `pixel`, `balanced`, or `editability`;
- `required_edit_targets`, `allowed_native_parts`, `must_retain_parts`.

Regions without these fields remain valid legacy annotations and are skipped by
scope scoring. `score.llm.json` reports routing agreement and scope agreement
separately, with over-decomposition, under-decomposition, and retain-scope
violations. Scope agreement requires the routing label plus coherent part scope.

Committed:
- final plans, validation, score, apply report, provenance and human decisions
- the generated shadow report

Not committed:
- region crops and request/response bodies under `.build/region-shadow/llm-jobs-v2/`
- the P8 comparison image and hybrid sample PPTX

Those excluded files contain source pixels or model text that may include personal
case data. They remain local evidence and are not published to the repository.

`validation.llm.json` is bound to the exact file it checked. Its `checked.plans_sha256`
and `checked.human_decisions_sha256` must match the current files; a validation result
from before an edit is not evidence for the edited archive. The `raw_artifacts` note
lives here rather than in `plans.llm.json`, which is schema-validated with
`additionalProperties: false`.
