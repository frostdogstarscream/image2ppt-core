# Host tool catalogue

Read the actual capability response on every session. Names in a proposal do not imply implementation.

| Capability | Initial runtime behavior |
| --- | --- |
| get_page_overview / get_scene_graph | Host supplies source image, object IDs, bboxes, ownership, relations and evidence index |
| inspect_region | Crop the union of actual object bboxes plus bounded context; return relevant frozen evidence and explicit image roles |
| inspect_conflict (inspect_owner_conflict adapter) | Return source-bound conflict receipts, protected-object attribution and masks/overlays; no resegmentation or protection changes |
| list_candidate_strategies | Host lists existing product strategies separately from executable agent adapters |
| request_candidate / request_repair | Execution host only: actual T1/T2 generation, error-bound T2 or resolved T3 repair; no direct pixel-writing arguments |
| request_joint_region | Execution host only: existing T3 card tool after deterministic carrier closure; generic cloud-label reconstruction remains unavailable |
| propose_scene | Validate actual IDs/geometry/evidence; only private AMBIGUOUS containment or joint observation scope. Missing/split/merge perception NOT_IMPLEMENTED |
| request_staging / request_compile | Execution host only: preflight PASS → isolated SceneGraph adapter → fixed compiler |
| request_edit_test | Execution host only: compiled PPTX → actual replace/move/delete/save-reopen and independent residual check |
| request_visual_review | Execution host only, all safety/Office gates passed and separate Go configured; network failures remain UNKNOWN |
| propose_selection | Execution host only: hash-bound tested candidate and acceptable independent defects → shadow recommendation; never publishes |
| stop_and_fallback | Close the region; preserve original representation and state reason |

Actions carry session ID, previously planned region ID, a nonempty subset of that region's actual objects, known evidence references, bounded arguments and a reason. No paths or endpoint fields are accepted. The host derives crop coordinates; a model bbox is not an erase mask.

Use fresh action IDs for genuinely different decisions. Reusing the same action ID with different bytes is an error; reusing identical bytes returns its receipt. Budget exhaustion ends work for the region. An unavailable capability is a useful negative result, not success.
