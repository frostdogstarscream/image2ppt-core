# Independent typography refinement on Windows

This is a command-driven workflow. It does not load skills or change their files.
Use the existing project-local core Python environment, Node and Artifact runtime.
Install the additional dependencies from `requirements-typography.txt` into the core environment.

## Credentials

`credentials.py install-stdin` accepts a JSON object with `moma` and `opencode_go`
through stdin only. Supply it using a non-echoing prompt or secret manager; never
put values in command arguments, scripts, source control or logs. Storage is under
`%LOCALAPPDATA%/image2ppt/credentials/providers.dpapi`, protected by current-user
DPAPI and a user/SYSTEM directory ACL. Run as the same Windows account used to
configure credentials. `rotate-moma-stdin` replaces only MoMA. `status` reports
presence booleans. OpenCode Go is stored separately and is not an automatic model
fallback; configuration does not establish successful invocation.

## Run and resume

```bat
image2ppt-local.cmd refine-typography --baseline <r5-surface-batch.json> --output <new-run-directory> --stage pilot
image2ppt-local.cmd refine-typography --baseline <same-batch.json> --output <same-run-directory> --stage all
```

`--stage plan` performs planning only. `--reviews` overrides the existing surface
review directory. Pilot pages are 01, 03, 04 and 10. Run directories are immutable
inputs plus versioned attempts: completed responses are reused only when their
source and request hashes match. Changed inputs require a new run. Never run two
writers against the same run directory.

Default call policy is 26 total requests and two attempts per planning item,
including failures. `api-budget.json` records every HTTP attempt before sending;
the provider client's hidden retries are disabled. Explicit user-authorized
budget changes are recorded in that ledger (`limit: null` means no total cap;
`default_job_limit` still bounds corrective attempts). Do not reset the ledger to
bypass a failed attempt. Auth and transport failures stop the invocation; invalid
plans receive precise validation feedback within the applicable attempt bound.

## Contracts and behavior

The existing `FontPlan/1` is unchanged. `FontPlan/2` binds every existing paragraph
and explicit run exactly once to shared styles; text and coordinates cannot be
returned as mutations. `font_size_px` is source-image em size, not OCR ink height.
`DeckStyles/1` partitions page style IDs into shared groups. Structure findings
are advisory strings only. Raw responses, original suggestions, local fits and
final decisions remain distinct evidence.

The font manifest checks the installed file, TTC index, exact weight, hashes and
required characters. Proposals are calibrated against source pixels, then checked
using the fixed compiler's actual PPTX rendering. Selection requires regional
improvement plus glyph coverage, stroke-weight, extent and integrity gates;
retained baseline objects are never counted as corrected.

Background/card rebuilding runs before typography fitting so both baseline and
candidate use the same clean rendered background. Its masks continue using the
frozen source text evidence, not newly proposed colors. The selected deck is
rendered with all native text deleted and checked against that background, plus
normal/contrast OCR. Representative text and card edits are rendered separately.

Delivery includes normal/deleted decks, original/r5/r6 previews, text crops,
per-object decisions, font catalog, model responses and usage. Automated PASS
does not establish original font identity or Office/WPS/human acceptance.
