# OpenCode autonomous runtime

The runtime uses OpenCode 2.1.4, MCP 1.3.0 and Pydantic 2.10.x. MCP 1.3 imports an internal Pydantic helper, so the product dependency range is constrained to the compatible minor version. Its job-scoped MCP server exposes basic image/evidence tools together with optional reconstruction shortcuts. Original pipeline defaults remain available.

## Launch

Install `services/product/requirements.txt` in a dedicated environment. Provide the existing OCR manifest and GPU client configuration through `.runtime/parallel-services.json`. The fixed compiler needs its verified Node/artifact-tool runtime. Use a short jobs root on Windows.

```powershell
$env:PYTHONPATH="$PWD/services/product"
$env:IMAGE2PPT_PRODUCT_TOKEN=[Guid]::NewGuid().ToString('N')
python -m product.runtime.launch --root F:/i2oc/jobs --port 8771 --prompt-key
```

The hidden prompt supplies MoMA credentials to the process and its OpenCode child. The launcher accepts `IMAGE2PPT_OPENCODE_BIN`, verifies the exact server version and redacts process credentials from its server log. Production credentials should be injected by the service host. Keep the product token available to the authenticated business client.

Select Go inference by setting `IMAGE2PPT_AGENT_PROVIDER=opencode_go` and `IMAGE2PPT_AGENT_MODEL=qwen3.8-flash` before launch. Supply `IMAGE2PPT_AGENT_API_KEY` through the host or `--prompt-key`. This selects `https://opencode.ai/inference/go/openai/v1` for both the OpenCode Session and semantic/planning tool calls. The Agent and tool transports identify their client and share the job's stable session header. Visual self-review stays in the current Session.

The Windows bootstrap accepts `agent_provider`, `agent_api_key` and `agent_model` for this configuration. Its generated access profile continues to hold business/worker credentials only. `agent_inference_provider` in health and job receipts identifies the active transport. `opencode_go_configured` refers to the legacy independent review credential.

Submit ordered multipart `files` to `POST /api/v1/jobs`. Query the resulting job, SSE events, cancellation and artifact list under `/api/v1/jobs/{id}`. The legacy `/api/jobs` keeps its existing default route. The request ledger records `orchestrator=opencode` and `visual_review_backend=opencode`.

## Agent and evidence

The runtime Skill is `skills/image2ppt-runtime/SKILL.md`. A job receives a private MCP context, source attachments and the delivery goal. Each tool records its input, implementation/candidate bindings and output artifacts. Candidates may combine separately generated masks and backgrounds. The source mask, private staging adapter and actual Office checks validate submissions.

`tool_profile=basic` in a task request exposes the basic tool catalog for autonomous-composition acceptance. It omits `run_perception`, `run_parallel_pipeline`, `rebuild_region` and `repair_region`. It does not select an object or prescribe a tool sequence.

Visual reviews bind the final package, scene, Office previews and edit evidence. `get_evidence` returns actual MCP image content and records provision. This is a transport receipt. Missing required checks or a changed file blocks finalization. Complete recall remains unknown without an independent source inventory.

Scene proposals may grow incrementally. Updates preserve accepted object definitions and shared styles. A changed accepted object requires a local rollback. Candidate commits bind both the parent private scene and public IR, so an unchanged raster IR cannot conceal changed ownership evidence.

The optional `fit_background.context_width` changes the independent source sampling strips. The host recomputes that evidence at staging with the original fit, content, residual and owner thresholds. Custom masks must remain inside independently reconstructed semantic text pixels and pass the whole-scope residual checks. Neither parameter grants new erase authority.

`agent_steps` bounds OpenCode reasoning turns. `model_calls` bounds model calls inside the Image2PPT tool adapters. Both budgets are recorded separately. `runtime_seconds` is a persisted task deadline. Agent usage reports distinguish reported tokens and provider cost from a verified monetary price. A zero provider-reported cost has no established currency value here.

The driver waits for the job's MCP connection before admitting the Agent prompt. The prompt and tool receipts expose the remaining wall-time budget. Image evidence references include filenames so the Agent can request local comparisons selectively. The Office adapter can return an unverified native-frame size proposal after replacement overflow. That proposal requires a new candidate and the complete existing checks.

Tool outcomes, region commits, final QA and publication qualification have separate receipts. A source-only raster package remains `REVIEW_REQUIRED`. A hash-bound, Office-checked and self-reviewed partial editable package can finish successfully. Session admission with an uncertain response is reconciled through the saved Session and message IDs before another prompt is considered.

## Office worker

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

```powershell
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi submit input.png --tool-profile basic --seconds 1800 --goal "Restore at least one complete editable object and verify the partial delivery."
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi status JOB_ID
python services/product/opencode_client.py --profile .runtime/opencode-client.dpapi artifacts JOB_ID --output outputs/downloaded-job
```

## Optional script sandbox

`run_image_script` requires a pre-provisioned Docker image containing Python, Pillow, NumPy, OpenCV and libseccomp. Set `IMAGE2PPT_SANDBOX_IMAGE` to a local digest-pinned image reference. The runner restricts network, privileges, process count, memory, execution time and output size, and mounts only explicit inputs and task output. The trusted entry installs a kernel syscall filter that denies execution, process creation and sockets before the Agent script runs. Missing isolation returns an unavailable result. Script outputs remain unverified evidence.

## Checks

```powershell
python -m pytest services/product/tests/test_opencode_runtime.py -q
./image2ppt.ps1 test
```

Synthetic checks cover contracts and recovery behavior. Real image/Agent/Office acceptance is recorded separately under `docs/acceptance` when executed.
