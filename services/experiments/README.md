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

