# image2PPT project

This project implements a Codex-assisted screenshot-to-editable-PPT prototype.

- Use the `image2ppt` skill at `.agents/skills/image2ppt/SKILL.md` for restoration requests. The maintained source lives at `skills/image2ppt/`; keep installed copies synchronized when changing the skill.
- Treat the source page as the design reference. Preserve its wording, numbers, layout and proportions. Do not apply generic presentation redesign defaults to restoration.
- Use the bundled Artifact Tool runtime and fixed compiler. The model step runs in the current Codex conversation; project scripts do not call a model API.
- Preserve originals and all completed output versions. Synthetic fixtures establish only tool-chain behavior, not real-image accuracy or human acceptance.
- Run `./image2ppt.ps1 test` for protocol changes. For compiler or OOXML adapter changes, run the synthetic demo and `scripts/check_demo.mjs` on its IR, inspect actual-file previews, and retain verification reports.
- PDF, OCR, inpainting, native tables/charts and batch conversion are outside v0.1. Extend them only when requested and with representative source evidence.

