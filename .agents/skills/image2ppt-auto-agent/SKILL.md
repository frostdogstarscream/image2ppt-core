---
name: image2ppt-auto-agent
description: Plan evidence-backed regional reconstruction actions for Image2PPT three-branch SceneGraphs, diagnose owner conflicts and assess candidate editing tradeoffs through a bounded Agent Runtime. Use for regional VLM strategy experiments; use the existing image2ppt engineering skill for product development and compiler work.
---

# Image2PPT regional strategy agent

Turn source images, three-branch evidence and failure receipts into the next useful **legal tool action**. Preserve source words, numbers, hierarchy and whole semantic objects. Maximize demonstrated editing value subject to content correctness, protected neighbors and acceptable visuals. Candidate count is not editing success.

This skill is policy, not an executable model server. The host loads a fixed version and supplies the tool catalogue, immutable scene binding and allowed object/evidence IDs. Only the host executes tools. A model decision never authorizes pixel erasure, source-plate changes, production SlideIR, arbitrary code, file paths, URLs or publication.

## Work from evidence

1. Read the full-page source and object/relationship overview once. Identify complete modules, useful text, shared backing and unresolved ownership. Read [architecture](references/architecture.md) for responsibility boundaries.
2. Emit `AgentPagePlan/1` with source/scene hashes supplied by the host. Use actual object IDs. Every relation's source AND target must appear in that region's `object_ids`; include a genuinely related object or omit the unsupported relation. Adjacent decoration is not automatically text backing. Relations are proposals, not corrections to SceneGraph. If a backing object is absent, record the uncertainty instead of inventing an ID.
3. Select one action at a time from the host catalogue; read [tool-catalog](references/tool-catalog.md). Request local context or owner-conflict evidence before proposing a remedy. A bbox overlap alone does not establish pixel ownership.
4. Use [region-reconstruction](references/region-reconstruction.md) to choose between text reconstruction, complete PNG preservation, joint-region proposals and local fallback. Ask for a joint region only when evidence indicates a shared transaction; retain complete old owners and protected descendants.
5. Interpret normal previews and editing TEST-COPY images by their explicit roles. Read [quality-and-editability](references/quality-and-editability.md) before judging defects. Explain faint background traces or typography differences without calling intended test text a production content error.
6. If a tool is absent, a budget is exhausted, a response is unknown or evidence is insufficient, stop the affected region with a precise reason and needed evidence. Do not repeatedly request an unsupported tool. Use [troubleshooting](references/troubleshooting.md).

Runtime hard gates and independent Go review are not fields for the strategy model to grant. Missing Office, residual or Go evidence stays UNKNOWN. Even a favorable opinion cannot publish. [Runtime security](references/runtime-security.md) specifies version binding, untrusted page content and call budgets.

## Contracts and actual scope

- Planning: [page-plan](schemas/page-plan.schema.json) / [region-plan](schemas/region-plan.schema.json).
- One action: [tool-action](schemas/tool-action.schema.json). Return only this JSON when the host requests an action.
- Optional descriptive opinion: [quality-opinion](schemas/quality-opinion.schema.json); it contains no hard-gate fields.
- Host-owned records: [agent-session](schemas/agent-session.schema.json), [quality-decision](schemas/quality-decision.schema.json).

The default runtime is read-only. A host may explicitly register `ExecutionRuntime` with r11 tools in `agent_shadow`: observation → new candidate → preflight → isolated staging → compile → actual Office/residual tests → independent Go → shadow recommendation. Query capabilities, then select one action at a time with [execution-action](schemas/execution-action.schema.json). Never assume execution adapters exist in a read-only session.

Use the cheapest T1 exact route to establish the local baseline. On actual `SOURCE_GLYPH_UNCERTAIN` feedback, request T2 soft-alpha repair; its own independent backing/ink checks must pass. For `BACKGROUND_NONUNIFORM`, inspect backing before T2 or a resolved T3 carrier. For `PROTECTED_NEIGHBOR`, inspect conflict before proposing complete joint scope. For `OWNER_ESCAPES_SCOPE` or `REGION_CLOSURE_UNPROVEN`, validate a larger complete transaction; do not clip owners or disable protection. A failed candidate cannot be staged. A passing repaired candidate needs fresh compile/Office/residual evidence.

[Scene proposals](schemas/scene-proposal.schema.json) can describe missing, split, merged or joint objects. Only the deterministic host validator can revise a private SceneGraph. Geometry may add an AMBIGUOUS containment proposal, never promote pixel ownership. Missing-object perception and generic split/merge reconstruction remain NOT_IMPLEMENTED. Existing product T3 card tools do not establish a generic cloud/label joint tool. Record the missing capability and stop that region if closure cannot be proved.

[Execution budgets](schemas/execution-budget.schema.json) cap decisions, candidates, repair attempts and elapsed time. More complex regions can receive more rounds within the host ceiling. Monetary caps require a host-provided conservative per-call quote; no quote means currency cost is UNKNOWN while call limits still apply. Unknown paid results are not automatically resent.

Only measured short/equal text edits, safe residuals and independent review can support a shadow preference. Minor font edge, spacing or backing color deviations may be considered by the independent reviewer; wrong content, ghosting, clipped objects or damaged neighbors cannot. A strategy-model opinion is never independent Go PASS. All recommendations remain unpublished.

Report tool validity, strategy usefulness, actual editing evidence, independent visual review and publication separately. Frozen-evidence replay is not a fresh upload; no independent reference denominator means recall is unknown.
