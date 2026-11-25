# OpenCode autonomous runtime

The automatic entry point is `POST /api/v1/jobs`. On a new upload the host builds an Office-checked baseline, then an OpenCode Agent addresses open issues using the tools listed by `get_job_context`. The runtime uses OpenCode 2.1.4, MCP 1.3.0 and Pydantic 2.10.x. MCP 1.3 imports an internal Pydantic helper, so the product dependency range is constrained to the compatible minor version.

## Region-based text and context-assisted reading

The admission host freezes two experiment settings in each new OpenCode job. By default, new jobs reconstruct text on a common title, band or card
surface before falling back to individual lines. With a configured MoMA reading model the host can prepare a local reading draft after the baseline. Add `IMAGE2PPT_PAGE_READING=1` to enable the full-page reading comparison.
The upload API is unchanged. Existing jobs keep their recorded settings. Set `IMAGE2PPT_REGION_TEXT=0` for a line-only comparison. Page reading remains opt-in.

The page reader sees a numbered source image. An independent local reader sees
only crops. They use fresh calls to the same model. Matching ordinary wording
is recorded as model-confirmed, separately from human confirmation. Disputed
readings and fields retained in raster content are recorded in the delivery
report. Changes to quantities, units and names require additional source evidence.

For a doubtful word, pass its object ID from the current page's numbered inventory to `review_numbered_text`. Explicit requests can include a high-confidence OCR line outside the issue blocks. Each requested ID gets an entry in `items` with a status (`observed`, `skipped` or `unavailable`) and a reason. When the source is bound, the entry identifies the target and OCR row. A single-view result records the original wording, transcription and legibility. In a two-view comparison, individual transcriptions appear in `readings` and missing answers in `unavailable_views`. `checked` counts distinct targets for which at least one model reading was nonempty. A split field must map unambiguously to the requested part of the source line. An OCR-only field or a text-layout mark can yield a reading while it still lacks a writable native field. Such an observation supplies evidence for a later rebuild. Office verification requires an actual editable object.

For the frozen comparison, run `services/experiments/run_native_benchmark.py`
against three pinned hosts (control, region, combined) with the same input roots,
`--manifest services/experiments/holdout/manifest.json --probe-selected`, and a
separate output directory for each. `services/experiments/compare_region_reading.py
CONTROL REGION COMBINED REPORT --project-root PROJECT --input-root INPUTS`
checks the inputs, downloaded PPTX hashes and actual edit receipts before comparing
the three sets. Page 08 uses `services/experiments/probe_region_text.py SOURCE PPTX OUTPUT`
to check its two new editable labels and three previously working actions.

## Host-precomputed uploads

New OpenCode uploads run deterministic perception on up to two pages concurrently, then batch reconstruction, candidate validation, commit, compilation and actual Office checks. `host-baseline/manifest.json` identifies the immutable control PPTX, scorecard and state. The host records source bindings before opening an Agent session. The 4090 detector configuration uses `IMAGE2PPT_EXPERIMENT_CONFIG`. The runtime records unavailable detectors in page evidence.

Object-scoped uploads checkpoint verified text first. The next checkpoint fits the measured native panels even when they have no badge. Inside a panel, a resolved or explicitly detected icon can be separated from its circular base or from sufficiently uniform paper. A retry on an existing native face checks its backing and any retained foreground ownership. A complex picture that already owns the pixels stays intact. The page source and measured geometry identify shapes across retries. Symbol trials record detector IDs, the measured bounds and the selected representation. Each checkpoint and the final page go through the Office worker.

A measured card can also go underneath an existing transparent icon or separator. The host checks the earlier Office-validated scene, the image and its ownership mask, and the exposed source behind the transparent pixels. A line or picture crossing the card edge prevents that replacement. The new card supplies the delete background for its contained children, and the Office worker repeats their edit checks on the combined page.

When the source contains several separate cards inside a section, the host processes the cards first. Both host and Agent surface routes defer the section while any of those child frames lacks a corresponding native shape. This preserves each child's editable scope when another card is difficult to recover.

The host retains the original OCR file, line IDs, full line bounds and wording. It binds whole native fields, complete rows of source-bound paragraphs and adjacent native fields in reading order. Punctuation participates in the match. In object-scoped reconstruction, a compact layout detection at either end of a line is separated when an empty gutter and independently measured character count support it. The derived field bounds and the graphic's detection ID are retained alongside the original OCR bounds. Repeated, distinct source ink colours in full-width text produce native formatting runs and per-run erase masks. An independently measured field separator can use the parent text colour when the child colour measurement is inconclusive.

The local retry queue includes source text still visible when an existing native field is hidden in Office. Each trial restores the original field bounds on a private copy. Only an independent text owner with a verified, isolated rollback may be reopened. Each batch contains at most 30 targets. Adoption requires an additional clean, exact native field in the checked PPTX and preserves earlier verified edits. The scorecard identifies retained artwork that visibly carries an OCR reading through its bound glyph pixels. Numeric fields and units may fit their native frames to observed ink and measured font bearings while keeping the neighboring ink and the erase mask fixed.

For an isolated single-line field, the runtime retains the actual typeface and size used by the failing PowerPoint replacement. When that edit stayed inside its own region and the full-size frame cannot fit, it may use a smaller source-supported size and scale the measured frame accordingly. The retry must pass the same PowerPoint save, reopen, replacement and deletion checks. Multiline or mixed-style fields retain their own layout.

The runtime Office worker exports a second PNG per slide with editable text cleared in an unsaved copy of the presentation. Pictures, card fills, groups and their stacking order remain visible. The receipt binds this view to the checked PPTX, its hidden field names and the PNG hash. The host runs local OCR on that view and records readings from the bare plate separately. A partial native match, including a missing digit or punctuation mark, does not count as a complete field.

`get_job_context` returns two indexes per page. `issue_blocks` supplies the mark numbers for `plan_marked_page` and the scope for local rebuilds. `reading_blocks` covers the page's numbered inventory in blocks of up to 30 marks. Each entry identifies the block and its `image` and `inventory` evidence. Fetch the relevant block with `get_evidence` to find a reading's object ID or inspect a high-confidence line missing from the issue list.

The plan accepts a route, role and `move_with` partner numbers. A `surface` route may include the outline hint `roundRect`, `homePlate` or `chevron`. The host measures the proposed geometry. To move a card and its label together, plan the card and the movement relationship. Check the resulting group with a single-selection move in PowerPoint. Source text and placement come from the job inventory. Private evaluation references are read only by the offline evaluator.

Local rebuilds accept an issue block or up to 30 current object IDs. The host counts batch attempts per participating object. One attempt is available initially. A second requires the preceding batch to have made progress through a candidate commit. When some requested objects remain eligible the host processes that subset. An all-ineligible request returns `local repair limit reached for the selected objects`. A refreshed issue list and a new plan do not reset the history.

Both arms use the same immutable source. The host compares page identities, Office verification, previously passed object edits, editable object counts, OCR lines matched by native fields (including recorded reading revisions), native field counts and background text findings. It also compares the Office render with the source over each marked region and the whole page. RGB mean error is recorded. The regression decision uses ΔE00 p95 and SSIM. For each region the allowed maximum color difference is the larger of the baseline value and the configured hard limit. The minimum similarity is the smaller of the baseline value and the configured hard limit. A reviewed reading draft matching the candidate file may use the configured margins in its changed fields. Soft deviations and hard limits are defined in `configs/fidelity_tolerance.json`.

The host keeps the immutable internal PPTX under `host-baseline/`. A locally checked Agent composition is also an internal candidate until selection. If the Agent fails, an eligible baseline can be downloaded with `REVIEW_REQUIRED`. Strict object-scoped uploads apply the same eligibility check on fallback. When both choices are ineligible, the API exposes the summary, scorecard and diagnostic preview. `runtime/delivery-selection.json` records `selected_sha256: null`. Read the selection and the public artifact list together. A file listed for download may still require content confirmation. Report the terminal status, the actual downloadable files and their hashes, and the independent review receipt separately from human or WPS acceptance.

Run `services/experiments/evaluate_two_pages.py` twice with the same `--job`, private `--references` and `--cases`. Use separate output directories and `--arm baseline` and `--arm final`. Each invocation performs the six actual PowerPoint edits on copies and compares the exported slide to the original. Keep these artifacts outside Git.

## Launch

Install `services/product/requirements.txt` in a dedicated environment. Provide the existing OCR manifest and GPU client configuration through `.runtime/parallel-services.json`. The fixed compiler needs its verified Node/artifact-tool runtime. Use a short jobs root on Windows.

```powershell
$env:PYTHONPATH="$PWD/services/product"
$env:IMAGE2PPT_PRODUCT_TOKEN=[Guid]::NewGuid().ToString('N')
python -m product.runtime.launch --root F:/i2oc/jobs --port 8771 --prompt-key
```

The hidden prompt supplies the selected execution credential. Configure the independent review provider and its credentials separately. The launcher accepts `IMAGE2PPT_OPENCODE_BIN`, verifies the exact server version and redacts process credentials from its server log. Production credentials should be injected by the service host. Keep the product token available to the authenticated business client.

Select Go inference by setting `IMAGE2PPT_AGENT_PROVIDER=opencode_go` and `IMAGE2PPT_AGENT_MODEL=qwen3.8-flash` before launch. Supply `IMAGE2PPT_AGENT_API_KEY` through the host or `--prompt-key`. This selects `https://opencode.ai/inference/go/openai/v1` for both the OpenCode Session and semantic/planning tool calls. The Agent and tool transports identify their client and share the job's stable session header. Visual self-review stays in the current Session.

The Windows bootstrap accepts `agent_provider`, `agent_api_key` and `agent_model` for this configuration. Its generated access profile continues to hold business/worker credentials only. `agent_inference_provider` in health and job receipts identifies the active transport. `opencode_go_configured` refers to the legacy independent review credential.

For official DeepSeek V4.1 Flash, set `IMAGE2PPT_AGENT_PROVIDER=deepseek` and
`IMAGE2PPT_AGENT_MODEL=deepseek-flash`. The host uses `https://api.deepseek.com/v1` and
takes the execution credential from `IMAGE2PPT_AGENT_API_KEY`. For independent MoMA
reading and review, provide the separate `MOMA_API_KEY`, set `IMAGE2PPT_REVIEW_PROVIDER=moma`
and set `IMAGE2PPT_REVIEW_MODEL` and `IMAGE2PPT_MOMA_READING_MODEL` to `qwen/qwen3.8-max`.
Use `--agent-model deepseek-flash` with the fresh benchmark client. Use a new output
directory when changing model. Run all three arms of a page pair with the same execution model,
and report previously completed OpenCode Go pairs in a separate cohort.

Submit ordered multipart `files` to `POST /api/v1/jobs`. Query the resulting job, SSE events, cancellation and artifact list under `/api/v1/jobs/{id}`. The legacy `/api/jobs` keeps its existing default route. The request ledger records `orchestrator=opencode` and `visual_review_backend=opencode`.

## Agent and evidence

In the host-precomputed upload, the available tools are `get_job_context`, `get_progress`, `get_evidence`, `prepare_page_evidence`, `plan_marked_page`, `review_numbered_text`, `batch_reconstruct_page`, `commit_candidate`, `compile_pptx`, `validate_office`, `record_visual_review` and `finalize_job`. This list takes precedence over `tool_profile`. Fetch up to eight evidence IDs per call. Use current issue blocks after a commit. A successful tool call may return zero readings, no usable candidate or a negative quality verdict. Inspect the per-object result and the candidate or selection receipt.

The following low-level tools describe jobs using the exploratory catalog. `create_candidate` uses the existing scene adapter to create a private native preview, including candidates whose mask or background remains unapproved. `compile_pptx(candidate_id)` compiles that preview without changing the committed page or delivery composition. `validate_office(candidate_id)` returns edit diagnostics for observation. `commit_candidate` separately invokes the existing submission checks and actual Office verification.

`validate_mask` and `validate_rgba` return measurements and uncertainty. RGBA measurement accepts image evidence directly. `unmix_foreground` computes from explicit source, background and mask evidence. Scene-mode `complete_text_mask` requires a chosen background. Inpainting accepts source/mask evidence without requiring text-mask approval. Layout proposals include conflict measurements. LayerD exploration returns prefix hypotheses with their actual opacity coverage and recomposition error. Stable production callers retain strict staging behavior.

Agent local findings use `record_observation`, which has no Office prerequisite. Final delivery review remains hash-bound through `record_visual_review`. Existing formal submission thresholds are retained in the submission path. This revision removes their effect on exploratory generation and preview; it does not establish new real-slide visual acceptance.

The runtime Skill is `skills/image2ppt-runtime/SKILL.md`. A job receives a private MCP context, source attachments and the delivery goal. Each tool records its input, implementation/candidate bindings and output artifacts. Candidates may combine separately generated masks and backgrounds. The source mask, private staging adapter and actual Office checks validate submissions.

In a job without host precomputation, `tool_profile=basic` filters out `run_perception`, `run_parallel_pipeline`, `rebuild_region`, `repair_region`, `prepare_page_evidence` and `batch_reconstruct_page`. Numbered planning and inspection tools in that catalog also depend on the experiment profile. Refer to the job's actual MCP tool list.

Visual reviews bind the final package, scene, Office previews and edit evidence. `get_evidence` records which source, render and editing panels the Agent received. A missing required check or a changed file blocks finalization. The Agent self-review and independent visual audit have different receipts. Report the independent provider and actual model from the audit, including whether the response was obtained live, reused from cache or unavailable. A missing verified independent verdict leaves the result under review. Complete recall remains unknown without an independent source inventory.

Scene proposals may grow incrementally. Updates preserve accepted object definitions and shared styles. A changed accepted object requires a local rollback. Candidate commits bind both the parent private scene and public IR, so an unchanged raster IR cannot conceal changed ownership evidence.

The optional `fit_background.context_width` changes the independent source sampling strips. The host recomputes that evidence at staging with the original fit, content, residual and owner thresholds. Custom masks must remain inside independently reconstructed semantic text pixels and pass the whole-scope residual checks. Neither parameter grants new erase authority.

`agent_steps` bounds OpenCode reasoning turns. `model_calls` bounds model calls inside the Image2PPT tool adapters. Both budgets are recorded separately. `runtime_seconds` is a persisted task deadline. Agent usage reports distinguish reported tokens and provider cost from a verified monetary price. A zero provider-reported cost has no established currency value here.

The driver waits for the job's MCP connection before admitting the Agent prompt. The prompt and tool receipts expose the remaining wall-time budget. Image evidence references include filenames so the Agent can request local comparisons selectively. The Office adapter can return an unverified native-frame size proposal after replacement overflow. That proposal requires a new candidate and the complete existing checks.

Tool outcomes, region commits, final QA and publication qualification have separate receipts. A successful Agent composition also requires a passing independent audit and compliance with the final scorecard gate. The host then compares the eligible result to its baseline and applies the object-scoped delivery rules. Session admission with an uncertain response is reconciled through the saved Session and message IDs before another prompt is considered.

## Office worker

The OpenCode Office worker checks a presentation's full path before editing it and after reopening it. If PowerPoint returns an incomplete automation handle, the worker looks up the already-opened test copy by name and verifies the same full path. After closing, it confirms that the exact file has left the presentation collection. A transient close error retries only that close. The returned packet contains measured results and previews. An Office lock on a disposable PPTX cannot suppress an already measured failure receipt.

The local mode runs pywin32/PowerPoint on the executing Windows host. Set `IMAGE2PPT_OFFICE_MODE=remote` and a separate `IMAGE2PPT_OFFICE_WORKER_TOKEN` on the business host to use an outbound worker.

```powershell
$env:PYTHONPATH="$PWD/services/product"
python -m product.runtime.office_bridge --url http://127.0.0.1:8771 --root C:/i2oc-office
```

Supply the same worker token to this process. For a remote business host, use the established authenticated SSH tunnel or a protected HTTPS endpoint. Requests and results carry hash manifests. A leased but missing response stays unknown and is not automatically reassigned. Inspect retained packets before recovery.

The worker registers a process and installed-font binding. Identical Office inputs reuse checked evidence while that binding remains current. Uncertain results retain their original operation and packet. The worker continues independent requests after a failed leased packet. Font installation changes require restarting the worker.

## Windows host and client

`product.runtime.windows_host --root F:/i2oc` starts an independent current-user scheduled task using the existing dedicated `app` and `venv` directories. Bootstrap JSON arrives on standard input with `moma`, `business_token` and `office_token`. Its user-ACL named pipe transfers credentials in memory to the host. Model credentials require host reinjection after a restart. This bootstrap creates an interactive-user task with no reboot trigger.

The validated deployment uses business port 8772 and OpenCode port 14100 on the 4090 host loopback. A local SSH tunnel exposes business port 18772 to the current desktop. `windows_host` uses the previously verified fixed Node runtime on that host. OpenCode children belong to a kill-on-close Windows Job Object, which bounds their lifetime to the independent host.

`product.runtime.client.save_profile` encrypts generated business/worker access configuration with Windows DPAPI. Model credentials are outside that profile. Use the client with a profile or inject `IMAGE2PPT_PRODUCT_TOKEN` into its environment.

For a normal job, express the editing operations appropriate to the input page. Save, close and reopen the copy in actual PowerPoint when measuring whether those operations work. Keep benchmark-specific answers in the evaluator.

```powershell
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi submit input.png --goal "Restore business text and numbers as independent editable fields. Preserve card labels and icons through single-selection movement and recoloring. Keep separate photos removable. Verify applicable edits in PowerPoint."
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi status JOB_ID
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi artifacts JOB_ID --output outputs/downloaded-job
```

For a connectivity smoke test, a narrow goal such as “restore at least one complete editable object” can exercise upload and editing. Its pass criterion covers only that operation.

## Optional script sandbox

`run_image_script` requires a pre-provisioned Docker image containing Python, Pillow, NumPy, OpenCV and libseccomp. Set `IMAGE2PPT_SANDBOX_IMAGE` to a local digest-pinned image reference. The runner restricts network, privileges, process count, memory, execution time and output size, and mounts only explicit inputs and task output. The trusted entry installs a kernel syscall filter that denies execution, process creation and sockets before the Agent script runs. Missing isolation returns an unavailable result. Script outputs remain unverified evidence.

## Checks

Optional RMBG-2.0 graphic extraction in an OpenCode operation runs the configured model in a separate process. GPU queue time and inference share `matting.timeout_seconds` (default 120 seconds). A timed-out crop can still use the LayerD, SAM or fitted-background candidates. The job records the crop fingerprint, dimensions and model binding in `runtime/rmbg-timeouts.json` and skips the same failed input on retries. A second distinct timeout exhausts that job's RMBG retry allowance. Other jobs start with a separate allowance. An interrupted comparison does not populate the shared pixel cache.

```powershell
python -m pytest services/product/tests/test_opencode_runtime.py -q
./image2ppt.ps1 test
```

Synthetic checks cover contracts and recovery behavior. Real image/Agent/Office acceptance is recorded separately under `docs/acceptance` when executed.

For object-scoped uploads each Office ticket makes one physical check. On an infrastructure failure the host may reserve one extra Office validation and submit the same package a second time. Each ticket uploads its own `output/check` evidence. Retry evidence lives under the original operation's retry directory. Object failures are handled separately by the batch isolation path. The compact batch response includes `office_attempts` and `isolation_history`.

The strict delivery check in `product/runtime/host_baseline.py` requires verified editable objects on every page, a measured source-plate audit, no audit rows classified as retained, residual, plate-only or unresolved, and a single verified object owner for every fallback crop. Use the reported rows when investigating a rejection. Inspect whether a flagged mark is business wording, an intrinsic brand label or a decorative symbol. If fallback also fails these conditions, the API lists diagnostic artifacts only. When a PPTX is selected, the strict artifact list exposes the selected `result.pptx`.
