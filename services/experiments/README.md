# Six-model comparison over OpenSSH

All six experimental inference paths execute on the RTX 4090 host. The local Windows machine submits page/crop PNGs on SSH stdin, verifies input and artifact hashes, and runs PowerPoint. Go hy4 is the external planning model. CPU inference (including VTracer) also runs on the remote host. LayerD uses the existing authenticated remote loopback API; model credentials stay with that service.

The tested experiment root is `F:/i2oc/six-model-comparison`. Install the pinned artifacts with `prepare_models.py <root>/weights` and copy this directory's `worker.py` to that root. The tested isolated packages are doclayout-yolo 0.0.4, pymatting 1.1.16 and vtracer 0.6.15 under `deps`, with numpy 1.26.4 under `compat` for DocLayout-YOLO and ViTMatte/PyMatting. Existing PyTorch, Transformers, Ultralytics, PaddleOCR, and LayerD installations supply the underlying runtimes. Keep experiment dependencies isolated from the live services.

Set `IMAGE2PPT_EXPERIMENT_CONFIG` on the experiment product host to a JSON object mapping `v3`, `pp-l`, `doclayout`, `omni`, `vitmatte`, `local-layerd` and `vtracer` to OpenSSH argv arrays. For example:

```json
{
  "omni": ["ssh", "-o", "BatchMode=yes", "research-4090", "F:/i2oc/venv/Scripts/python.exe F:/i2oc/six-model-comparison/worker.py F:/i2oc/six-model-comparison --device cuda:0"],
  "timeout": 1250
}
```

Use the same command for doclayout, vitmatte, local-layerd and vtracer. For v3 and pp-l use the existing layout-venv interpreter with `--device cpu` and the remote PaddleX cache. Configure the SSH host alias with the correct wired interface and existing identity. Layout-L was tested with oneDNN disabled. Weight file size and upstream SHA for LFS files are verified on acquisition; per-file SHA-256 and source revision are persisted.

Submit `experiment_profile` through `POST /api/v1/jobs` or `python services/product/opencode_client.py ... submit --experiment-profile <name>`. Use OpenCode Go hy4. `run_comparison.py --url <local-api> --jobs <job-root> --out <run-output> --images page1.png page2.png --holdout p08.png --holdout p07.png` runs the matched matrix using the service access token in the environment. It preserves admissions, the 45-minute snapshot for still-running jobs, and terminal stage results. A missing result or absent model receipt is unknown. Count only independently tested PPTX edits as successful.

## Deployment smoke test

OpenSSH inference on real source pixels produced 50 PP-DocLayout-L boxes and 55 DocLayout-YOLO boxes on the first page. OmniParser ran on the full page and cropped windows; counts before deduplication depend on the selected windows. A local crop from that page passed through the remote LayerD service, ViTMatte plus PyMatting, and CPU VTracer, returning layer PNGs, an RGBA image and SVG. These receipts verify model execution, not editability.

## Reproduce the two-page evaluation

On the Windows PowerPoint host, use the product Python environment with pywin32. Keep the job directory and its original artifact paths available. Supply the two independent field inventories (`reference-fields-1.json` and `reference-fields-2.json`) and a private UTF-8 JSON array of edit cases. Each case contains the page number, target label, `[left, top, width, height]` on the 1672 × 941 source, action (`text`, `move`, `delete`, or `fill`), and replacement text. For example, a synthetic entry is `[1, "Target", [100, 80, 64, 28], "text", "Changed"]`. Use fresh output directories outside the checkout.

```powershell
python services/experiments/evaluate_two_pages.py --job $combinedJob --references $references --cases $editCases --out $combinedOut
python services/experiments/evaluate_two_pages.py --job $baselineJob --references $references --cases $editCases --out $baselineOut
python services/experiments/compare_deliveries.py $combinedOut $baselineOut
```

The evaluator checks the composition digest, locates the corresponding Office checks, and probes the six specified actions on private copies of the deck. If a terminal run has only committed scenes, the evaluator labels the independently compiled deck as diagnostic. The two evaluation reports contain field matches and model decisions. The comparison includes the number of differing PowerPoint render pixels for each page.

## Paired business-page result

On implementation `6ced62b`, the combined and baseline runs each scored **0/6** prescribed edits. Their PowerPoint exports at source resolution matched pixel for pixel on both pages. The six target operations covered a numeric field, a labeled role box, a software screenshot, an independently editable project name, a table field and a card fill.

| Page | Matched native fields | Native share of structured content | Verified movable picture share | Duplicate share | RGB MAE vs. source | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 28/93 | 30.55% | 1.99% | 0% | 8.230 | 0.8625 |
| 2 | 9/74 | 18.30% | 0% | 0% | 5.168 | 0.9242 |

All six remote workers returned model receipts. Hy4 selected a software screenshot and two tables. The batch reconstruction tool checked that both marked plans existed, then processed the existing SceneGraph without receiving the selected object IDs in either batch call. No independently deletable screenshot or native table survived in the deck. The backing-fit strategy ranked first on all eight ranked asset trials. All eight SVG candidates failed the contrast check.

At 45 minutes the combined job had composed a two-page deck with 42 and 15 committed objects. Its SHA-256 was `3befe29bb8dfff9f3c917b477762959b9aae82e89d2a281997625457887d2051`. Office passed 114 move/scale or delete records, 86 text replacement records and 57 residual checks on those bytes. Both agents subsequently ended in `REVIEW_REQUIRED` after `compaction.failed`. Human and WPS acceptance are unverified. This comparison covers the baseline and combined configuration on these two pages. It does not rank the six individual groups.

