# OpenCode identity and evidence repair — 2026-11-12

Status: protocol/deployment fixes tested and deployed; real reconstruction acceptance NOT achieved. No full-page acceptance claim.

## Changes

- create_scene initializes once; existing pages return SCENE_EXISTS. update_scene explicitly revises drafts and can remove uncommitted IDs. Accepted objects still require dependency-local rollback.
- Exact duplicate wording/bbox with another ID returns IDENTITY_CONFLICT and existing IDs, without silently changing the scene or pixel ownership.
- list_evidence returns task-bound references; get_progress distinguishes drafts, candidates, committed native text and independent images. Context recovery includes both.
- Visual review separates positive observations from unresolved findings and accepted deviations. Raster-only finalization explicitly reports reconstruction_succeeded=false.
- Scene text-mask input rejects a source/background image used as a binary original Mask. Scene mode can derive strokes without original_mask_id.
- SAM object_evidence exposes the actual ClosedObject contract. Task context reports missing sandbox prerequisites before execution.

## Checks

Product suite at 03a5b65: 329 passed in 41.30 seconds. Core protocol: 39 passed. Product suite after the submission-budget and diagnostic changes: 333 passed in 39.91 seconds. Targeted runtime plus Mask diagnostics: 48 passed in 8.86 seconds. Latest runtime check after missing-evidence diagnostics: 36 passed in 8.54 seconds.

Frozen replay of job f7611179379f46e5bce54c7bce03ec69 used zero new model calls. Incremental drafts progressed 15 -> 25 -> 43 -> 124. The old destructive create calls returned SCENE_EXISTS; explicit update of those payloads returned the prior-object identity conflicts. Local receipt: outputs/opencode-identity-replay/replay.json.

## Real run

Job 2f5bb613c20c4141bf9284bff6f4414d uses fresh page 06 input, OpenCode Go qwen3.8-flash, all tools, 64 steps and 2400 seconds. The goal is one meaningful native text object, independently chosen by the Agent, followed by real Office checks and visual review. It is not whole-page acceptance.

During the run the operator deployed Mask input diagnostics while all existing candidates were BLOCKED, retaining old candidate records. An interface-only clarification was sent because the Agent repeatedly treated optional original_mask_id as mandatory. No object ID, proposed coordinates, Mask or SlideIR was supplied by the operator. This run is assisted debugging, not proof of uninterrupted unattended operation.

Observed: Agent chose the long intro line; changed fitting input after poor background evidence; accidentally supplied the background as Mask, then diagnosed it using list_evidence; attempted unavailable sandbox; generated scene-mode mask; found extensive unknown pixels and chose the bottom caption instead. It split an ambiguous combined caption using explicit draft removal and addition, preserving unrelated drafts. Native submission and final QA remain pending until terminal evidence is recorded.

## First debugging run terminal result

{
  "job": "2f5bb613c20c4141bf9284bff6f4414d",
  "result": {
    "status": "REVIEW_REQUIRED",
    "orchestrator": "opencode",
    "visual_review_backend": "opencode",
    "reason": "Agent time budget exhausted before verified finalization",
    "automatic_publication": false,
    "final_qa": "UNKNOWN",
    "failure": "TIME_BUDGET_EXHAUSTED"
  },
  "model_messages": 35,
  "tokens": {
    "input": 259329,
    "output": 8501,
    "reasoning": 0
  },
  "office_requests": 0
}

The operator found an unsafe Agent attempt to remove the neighboring body draft to avoid protection. Source-bound OCR protection is now retained independently of draft presence. Independent recomputation found 127 protected pixels in derived title seeds. Automatic seeds now exclude known protected peers; explicit supplied masks remain checked. A later submission with an old broader Mask was rejected with 362 unauthorized pixels and 11 unassigned contrast pixels. These numbers do not prove an antialiasing cause. Submission rejection now returns actual difference/protection/reference Mask evidence.

Clean test e2547e26e4524324a67bd641e59e7153 starts at deployment e46bf78, basic tools, 3600 seconds, autonomous short meaningful label selection. No intermediate code changes planned.

## Deployment configuration correction

During the second test, LaMa returned HTTPError. Read-only listener/OpenAPI inspection proved port 18848 belongs to a task API (/v1/jobs), not the existing LaMa /inpaint API. The original deployment helper had incorrectly configured that port. The actual existing LaMa server is at loopback 18841; /health reports alive, CUDA, big-lama. Its model bytes match SHA256 344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9. Only the Image2PPT client URL and its local deployment helper were corrected. GPU deployments were not changed, and the unknown request was not resent by the operator. This means the second run has an infrastructure configuration intervention, although Agent tools/strategy and deployed code remained unchanged.

## Second test findings

Agent independently switched from an OCR-ambiguous heading to the meaningful label 平台应用示例, combined OCR, scene drafting, Mask completion, LayerD background extraction and native layout, and compiled a genuine exploratory TextBox. No operator supplied the target, Mask, SlideIR or tool sequence. The preview PPTX hash is 2d6c54e1afc429d280da720a42a3ed6225694e301f4305e3f1b4996b6a153238. This is an unaccepted candidate, not a delivered native object.

Two formal submissions failed. The first had 2904 authorized changed pixels, zero protected changes, zero changes outside the Mask and zero estimated source contrast outside the Mask. Its independent lateral reference had validation p95 217.6393, so the reported 82 residual pixels do not prove actual residuals or missing antialias pixels. The Agent nevertheless misread that number as outside-Mask antialiasing and switched to a smaller 2690-pixel Mask. The second submission retained the same invalid reference and failed again. Raw LayerD recomposition quality does not prove that the concealed background is correct.

The third proposed route hit RESOURCE_LIMIT because exploratory candidates and their formal-check records were both counted. Local code now tags formal records with submission_of and counts each exploration route once. It also labels reference-dependent estimates UNKNOWN when the reference fails, without changing numeric acceptance thresholds, ownership or neighbor checks. These later fixes were not hot-deployed into this test.

The final composed raster PPTX has SHA256 a2d9b2ae3aa40978934256271bdc8c7c5b5e52028862eb2f0121ad9d2037132d. Actual PowerPoint returned PASS_APP_AUTOMATED with an empty edit-record list and text replacements NOT_APPLICABLE. This proves raster opening/rendering, not native text editing. Agent attempted a failed visual review that explicitly recognized zero native texts; an unviewed evidence reference initially blocked recording, and it then retrieved further evidence.

Remaining root issue: formal submission still depends on a narrow gradient-reference proof even when the generated backing comes from LayerD or LaMa. This reference can be contaminated by nearby graphics. No threshold was relaxed and no source ownership was granted to resolve that failure. A trustworthy alternative reference/visual-evidence acceptance path remains unverified. Full-page reconstruction and independent PNG recovery are not accepted by this test.

## Terminal evidence and deployment

The second test ended REVIEW_REQUIRED / TIME_BUDGET_EXHAUSTED with final QA UNKNOWN. The final recorded OpenCode visual review is REVIEW_REQUIRED, explicitly acknowledging that the native-editability goal failed. There are zero committed objects, two exploratory candidates and two rejected submission records. One Office request/response completed; no native replacement or movement was proved. Reported model messages: 56; tokens: 177639 input and 13195 output; independent Go review calls: zero. Event elapsed time: 3598.37 seconds. These counts include failed attempts and compaction, not a guaranteed provider billing total.

Latest deployed code: dfc89fc520c21bd3e78193b309951ef8d04ad935; 371 package files verified after extraction. Previous code backup: F:/i2oc/rollback-identity-1794498122.zip. Deployment occurred only after the real job ended. Later fixes therefore have local regression coverage, not a claim of a second successful unattended run.

Retained local evidence: outputs/opencode-identity-clean/audit.json, result.json, deployment.json, final-evidence.zip and candidate-preview/exploratory.pptx. The evidence archive contains the raster fallback PPTX, actual Office preview/report, two failed candidate diagnostics, final failed self-review and the redacted Agent transcript. These artifacts remain outside Git source commits. The raster PPTX is explicitly a fallback, not the requested reconstructed deliverable.
