# Candidate workflow verification — 2026-10-26

Implementation base: main `b44adabb448e560be8fbd5876b1456427c26bf1f`.
Work was isolated from the existing dirty checkout. No external model calls or uploads
were made, and historical plans, inputs, decks and human decisions were preserved.

## Automated and actual-file verification

- 59 planning/candidate/coordinate/workflow tests pass in an isolated Python 3.12
  environment with `requirements-review.lock`. These include full scoped
  apply/validate/score, stale bindings, immutable revisions, correction allowlists,
  explicit source/destination boxes and legacy migration.
- The runtime-free CI jobs are configured; GitHub Actions has not been run for these
  local changes. A source-only export was tested without private inputs or untracked
  helpers from the original checkout. 43 existing core tests and 37 protocol tests pass.
- Synthetic demo and `scripts/check_demo.mjs` pass. Its page difference is 1.57377,
  a fixture observation only, not real-image accuracy.
- Artifact Tool 2.8.99 from bundled runtime 26.1020.11618 compiled and reimported the
  style probe. Final OOXML preserves red/bold, green/normal, inherited-color/normal,
  and inherited-color/bold runs, including Chinese/English text. Preview was inspected.
- Both maintained skill flavors were independently updated and re-exported. ZIP CRC,
  manifests and source-to-Codex-mirror equality are checked.

Windows' legacy path limit affected initial deep-output builds and a long-path core
test extraction. Candidate compilation now uses a short project-local staging path,
then copies verified outputs into the revision. Clean core tests passed after extracting
the identical source archive to a short path. Initial failure records are retained.

## Private page trials

The current runtime compiled both sides of each comparison. These fixed source-region
metrics differ from historical measurements and must not be substituted for old reports.
All values below are mean absolute RGB differences on the fixed region, lower being closer.

| Page | Retained/current baseline | Initial native/hybrid trial | After scoped typography correction |
|---|---:|---:|---:|
| P5 | 4.4905 | 4.4905 | unchanged |
| P8 | 4.7872 | 17.1131 | 17.1231 |
| P12 | 3.2215 | 11.6243 | 11.5888 |

- **P5:** native title plus retained dashboard remains within the saved scope. The title
  icon is raster and is explicitly mapped as retained. A direct final-PPTX deletion
  probe removes the editable title but exposes faint title-shaped traces in the archived
  background. The latest review is `REVIEW_REQUIRED`; automatic background repair was
  outside this change. This is a retained-material limitation, not a new coordinate shift.
- **P8:** the authorized historical hybrid trial was replayed using a separate calibration
  annotation. Title/body/row-label typography was corrected in a new revision, but the
  regional metric did not improve and the icon halo, backgrounds and separators still
  differ. Keep the original full-module delivery veto; the trial remains `REVIEW_REQUIRED`.
- **P12:** five bars, labels and exact values (35%, 22%, 18%, 15%, 10%) remain native
  shapes/text. Removing excess label boldness slightly improves the regional comparison.
  Rounded bar ends and background geometry still differ. It remains `REVIEW_REQUIRED`
  for a new human decision under the balanced goal; it is not a workbook-backed chart.

Direct native-text deletion in copies of all three final PPTX files was inspected after
reimport. Original candidate hashes stayed unchanged. This verifies those edit operations;
it does not establish PowerPoint/WPS application acceptance or eliminate all pixel residue.

Private evidence is retained under the isolated workspace's
`outputs/review-validation-20261026/`: immutable build attempts, review records, cropped
comparisons, full previews and edit probes. The `delivery/` directory contains the latest
three PPTX files and readable report. Private source pixels are excluded from source/skill
exports. Every new review and deliverable has `human_acceptance=false`.
