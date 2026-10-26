# Shared image2ppt-core M1

The model writes saved SlideIR 0.1 in the current conversation. Python coordinates deterministic local stages; it does not call a model API. The Node/Artifact Tool compiler remains shared with legacy commands. Use the bundled Python with Pydantic 2.13.5 and Pillow 12.3.0.

From the checkout use `python scripts/core_cli.py ...`. After installation use either `python -m image2ppt ...` or `image2ppt-core ...`. An installed package needs `IMAGE2PPT_PROJECT_ROOT` set to the complete checkout containing the Node scripts. Use `doctor` to inspect actual runtime paths and versions.

```text
python -m image2ppt doctor
python -m image2ppt replay --ir path/slideir.json --output outputs/core
python -m image2ppt replay --build path/build.json --output outputs/core
python -m image2ppt inspect --pptx path/result.pptx
python -m image2ppt probe --build path/build.json --cases path/cases.json
python -m image2ppt regress --suite configs/regression-m1.json --output outputs/regression
python -m image2ppt resume --run outputs/core/run-ID
```

Global `--timeout 300` and `--project-root PATH` precede the subcommand. Resume uses the recorded timeout and verifies cached content hashes. Each replay creates an independent run; its completed `revisions/rNNN` contains all replay inputs, result, actual-file preview, build/review/object map and legacy validation evidence. Probe copies live separately in `run-ID/probes/`.

Cases are JSON, never executable scripts:

```json
{"format_version":"1","cases":[
  {"id":"delete-caption","action":"delete_text","targets":["text-072"],"roi":[100,100,200,80]},
  {"id":"move-card","action":"move_group","targets":["card-group-015"],"delta_px":[24.0,16.0]},
  {"id":"dark-card","action":"change_fill","targets":["shape-011"],"fill":"172B4D","related_image":"image-012"}
]}
```

IDs must exist in the final PPTX object map. ROI and move deltas use **target preview pixels** (96 dpi), unlike IR bboxes which use source pixels. Delete only native text, move only real groups and change fills only on native shapes. The original PPTX hash must remain unchanged.

`PASS` checks only the stated method; `FAIL` is a detected discrepancy; `REVIEW_REQUIRED` needs review; `NOT_RUN` was not checked. Geometric intersections, opaque images or successful direct edits do not automatically establish a visual defect or semantic success. Only labelled synthetic solid-color ROIs support automatic residual-pixel assertions. PowerPoint/WPS editing and human acceptance are separate and remain unverified until actually performed.

Historical fixtures F01/F02 include cropped text/overlap risk; F03 has a background-compatibility risk; F04 records the old lost-run-style defect. The compiler now preserves run color and bold defaults; `scripts/check_rich_text.mjs` checks a new final export independently of those historical files. The real suite requires preserved private project outputs and is excluded from the portable source archive. Scoped candidate replay and conversation review are documented in `docs/candidate-workflow.md`; technical verification does not grant human acceptance.
