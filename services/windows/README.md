# Windows isolated deployment

The Windows deployment keeps model inference, OCR/reconstruction and the fixed Artifact Tool compiler in project-local environments. Existing Conda environments and system Python remain separate.

From the checkout:

```powershell
.\image2ppt-local.cmd status
.\image2ppt-local.cmd doctor
.\image2ppt-local.cmd test
.\image2ppt-local.cmd rebuild baseline
.\image2ppt-local.cmd layerd --input C:\input\page.png --output outputs\new-layer-pack --max-iterations 8
.\image2ppt-local.cmd compile --input C:\input\slideir.json
```

`rebuild` forwards to `ppt-rebuild/rebuild.py`, using the isolated OCR environment. Its `repair-text` command defaults to the local LaMa endpoint at `127.0.0.1:18841`; an explicit `--inpaint-url` takes precedence. `layerd` uses the private token file without printing the token. Always use new output directories.

Model services run as the `image2ppt-layerd-4090` and `image2ppt-lama-4090` scheduled tasks. Both bind only to loopback. The tasks start at boot, have no execution-time limit, and retry failures up to three times. Startup registration is configuration evidence; a full Windows reboot must be verified separately.

Logs: `.runtime/logs/layerd.log` and `.runtime/logs/lama.log`. New LayerD jobs: `.runtime/layerd/data/jobs`. Imported server evidence is preserved under `.runtime/server38-snapshot`. The application token is protected by a file ACL and must not enter Git.

The bundled compiler, Node and helper Python live under `.runtime/bundle`; `runtime.local.json` selects them. Native Skia requires a compatible Microsoft C++ runtime. On the migrated machine its system `msvcp140.dll` was version 14.00; Microsoft-signed 14.51 runtime DLLs were deployed beside the private Skia addon. System DLLs were not replaced.

LayerD keeps its fixed upstream revision and `bounded_components_v2` policy. The Windows launch profile uses 1024 processing resolution and accepts up to 12 iterations; requests still default to 3. Extra iterations and higher resolution do not guarantee more meaningful layers. Keep resolution experiments separate from the production baseline and inspect actual alpha masks and recomposition.

Migration evidence, real inference comparisons and rebuilt PPTX previews are saved under `outputs/migration-4090`. Compiler checks and direct OOXML editing tests do not establish PowerPoint/WPS or human acceptance.
