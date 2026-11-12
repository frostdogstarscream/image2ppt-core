# OpenCode identity and evidence repair — 2026-11-12

Status: integration testing in progress. No full-page acceptance claim.

## Changes

- create_scene initializes once; existing pages return SCENE_EXISTS. update_scene explicitly revises drafts and can remove uncommitted IDs. Accepted objects still require dependency-local rollback.
- Exact duplicate wording/bbox with another ID returns IDENTITY_CONFLICT and existing IDs, without silently changing the scene or pixel ownership.
- list_evidence returns task-bound references; get_progress distinguishes drafts, candidates, committed native text and independent images. Context recovery includes both.
- Visual review separates positive observations from unresolved findings and accepted deviations. Raster-only finalization explicitly reports reconstruction_succeeded=false.
- Scene text-mask input rejects a source/background image used as a binary original Mask. Scene mode can derive strokes without original_mask_id.
- SAM object_evidence exposes the actual ClosedObject contract. Task context reports missing sandbox prerequisites before execution.

## Checks

Product suite at 03a5b65: 329 passed in 41.30 seconds. Core protocol: 39 passed. Latest targeted runtime suite: 33 passed in 8.93 seconds.

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
