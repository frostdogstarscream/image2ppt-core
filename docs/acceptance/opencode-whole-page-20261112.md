# OpenCode whole-page deployment test — 2026-11-12

Status: CANCELLED after operator cost-control stop; final QA UNKNOWN. Deployment completed; whole-page editable reconstruction failed acceptance. No automatic publication.

## Scope

Deployment to the existing RTX 4090 host at 192.168.1.103, application F:/i2oc/app. Existing algorithm deployments preserved. OpenCode 2.1.4, main Agent OpenCode Go qwen3.8-flash. One new autonomous whole-page task, source PPT图片/06.png (AI+专项监督, 1672x941), SHA256 f39a7c200e789625333425cfe12a1153042de639a6005c1c09855fa4dcf075b5. Job f7611179379f46e5bce54c7bce03ec69. Runtime budget 7200 seconds, 128 Agent steps. No hand-authored SlideIR or manual candidate selection.

## Runtime repairs

Deployment initially used 0b41aaa. Incremental update_scene lost shared styles before first regional commit. Fix 312a2b9 merges drafts before rebuilding. Repeated create_scene could then erase the complete uncommitted draft; fix 3ccd774 preserves existing drafts. A versioned recovery operation restored the Agent-produced 124-object snapshot from scene-v4-body-text. Failed candidates and original snapshots were retained. These interventions mean this run is an assisted deployment test, not proof of uninterrupted unattended delivery.

Focused runtime tests: 32 passed. Complete product suite after both fixes: 329 passed in 40.76 seconds. Fixed-core protocol checks from the exploration revision: 39 passed. No compiler changes.

## Observed limitations

The later Agent full-scene proposal changed some object identifiers: final draft at version 23 contains 152 text objects, 92 RESOLVED and 60 AMBIGUOUS. At least 18 pairs have identical wording and bbox but different IDs. Draft counts are not native editable counts. No asset/surface drafts were produced at this checkpoint.

The only retained reconstruction candidate is BLOCKED / MASK_INCOMPLETE. Script execution reported SANDBOX_UNAVAILABLE. Agent also used absent object/evidence references and submitted a contradictory visual-review payload (passed=true with nonempty findings; required hard-check evidence was omitted). These failures must not be summarized as model service or Office outages.

## Actual compiled artifact

PPTX SHA256 c92780f0b56fd8c32726d4d1a545acfb6ad7e6ecbf4bfaf03c76be1d77ef04a6. Independent ZIP/XML inspection: zero native text runs, one full-page raster picture. Zero recovered independent icons.

PowerPoint 16.0 returned PASS_APP_AUTOMATED for the source-backed file. Object edit records are empty; text replacement is NOT_APPLICABLE because no native text objects exist. Therefore this result does not prove text replacement, independent icon movement, or editable reconstruction. Actual Office preview was inspected and remains visually source-backed. No pixel-equivalence claim is made from visual observation alone.

Local evidence: outputs/opencode-whole-page/, including deployment hashes, recovery receipts, state snapshot, runtime audit, actual-output/raster-only.pptx, office-preview.png, office-report.json and structure-check.json. Remote immutable task evidence remains under F:/i2oc/jobs/f7611179379f46e5bce54c7bce03ec69.

## Final run accounting

70 model messages; 69 recorded tool calls ({'completed': 49, 'error': 20}); input 231672 tokens, output 56942 tokens. Event span 4320.7 seconds; collection elapsed 4651 seconds. One Office request and response. Zero independent Go review calls. Provider reported cost 0 is not verified monetary pricing.

The Agent did not finalize autonomously. The operator cancelled further calls after repeated identity/protocol failures and no accepted native object. The controller reached terminal CANCELLED and downloaded the terminal result. The deployed service remains available.
