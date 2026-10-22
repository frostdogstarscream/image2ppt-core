# Region candidate trials

This directory records concrete page candidates, not routing labels. The state
chain is:

```text
PROPOSED -> VALIDATED_STRUCTURE -> TRIAL_BUILT
         -> VERIFIED | REVIEW_REQUIRED
         -> ACCEPTED | REJECTED
```

- `records.json`: candidate records with baseline/candidate artifacts, hashes,
  metrics, edit-target coverage, object budget, state history and human outcome.
- `validation.json`: state-machine and record validation.
- `artifact-verification.json`: local artifact hash verification. The records
  point at gitignored `outputs/` and `.build/` evidence; the verification was run
  against the local checkout with those artifacts present.
- `report.md`: generated summary; metrics only rank candidates.

Current candidates:

- P5: native title + one dashboard module. **ACCEPTED**; the previous native KPI
  decomposition is the measured baseline (13.817 to 13.141).
- P8: hybrid table/text sample. **REJECTED** by human veto; retained module is
  visually closer (region MAE 8.6 vs 17.7). The sample remains available for an
  editability-first goal.
- P12: native title/bars/labels/percentages. **ACCEPTED**; page MAE moved from
  16.097 to 17.190, but the data-bearing parts are editable and checkable.

Not measured here and deliberately left `null`: text correctness, per-target
coverage beyond the explicit annotations, and elapsed build time. Do not read a
lower MAE as an acceptance decision, and do not generalize these three samples
to other templates.
