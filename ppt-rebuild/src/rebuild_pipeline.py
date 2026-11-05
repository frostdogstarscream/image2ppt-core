"""Versioned page/batch orchestration using the project's fixed PPTX compiler."""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))

from image2ppt.compiler import ArtifactBackend, execute
from image2ppt.contracts import ProbeCase
from image2ppt.quality import edit_copy, inspect_pptx
from layerd_loader import load_page, to_record, verify
from reconstruction import materialize, prepare_candidates, save_working_layers
from slideir_adapter import export_ir, sha256, write_json


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def resolve_page(pages_root, page):
    matches = [p for p in Path(pages_root).iterdir() if p.is_dir() and p.name.startswith(page)]
    if len(matches) != 1:
        raise ValueError(f"expected one page matching {page}, got {len(matches)}")
    return matches[0]


def baseline(pages_root):
    records = []
    for directory in sorted(Path(pages_root).iterdir()):
        if not directory.is_dir() or not (directory / "manifest.json").exists():
            continue
        page = load_page(directory)
        report = verify(page)
        if report.get("mae", float("inf")) > 3.0:
            raise ValueError(f"LayerD composite mismatch: {directory.name}")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        records.append({"page": directory.name, "source_sha256": sha256(page.source_path),
                        "manifest_sha256": page.manifest_sha256,
                        "layers": [{"file": l.file, "z": l.z, "sha256": l.sha256,
                                    "alpha_mode": l.alpha_mode} for l in page.layers],
                        "upstream_revision": manifest.get("upstream_revision"),
                        "refinement_policy": manifest.get("refinement_policy"),
                        "matting_process_size": manifest.get("matting_process_size"),
                        "max_iterations": manifest.get("max_iterations"), "recomposition": report})
    if not records:
        raise ValueError("no LayerD packs found")
    return {"schema": "LayerDBaseline/1", "created_at": datetime.now(timezone.utc).isoformat(),
            "page_count": len(records), "pages": records}


def compile_ir(backend, ir):
    return execute([backend.node, ROOT / "compiler_bridge.mjs", "compile", ir], REPO, backend.timeout)


def pixels(file, size):
    with Image.open(file) as image:
        if image.size != tuple(size):
            raise ValueError(f"actual PPTX preview size {image.size} != source {size}")
        return np.array(image.convert("RGB"))


def measurements(source, rendered, registry, active):
    error = np.abs(source.astype(np.int16) - rendered.astype(np.int16))
    regions = []
    for entry in registry.objects:
        x, y, w, h = entry["bbox"]
        regions.append({"id": entry["id"], "kind": entry["kind"], "bbox": entry["bbox"],
                        "active": entry["id"] in active,
                        "mae": round(float(error[y:y + h, x:x + w].mean()), 4)})
    return {"full_page_mae": round(float(error.mean()), 4), "regions": regions}


def native_text_adjustments(rendered, source, registry, replacements, active):
    """One measured alignment correction, kept separate from committed revisions."""
    adjustments = []
    for entry in registry.objects:
        if entry["id"] not in active or entry["kind"] != "text":
            continue
        node = replacements[entry["id"]]["node"]
        x, y, w, h = entry["bbox"]
        # Measure source and rendered ink against this replacement's clean plate.
        bg = replacements[entry["id"]]["repair"].astype(np.int16)
        src_ink = np.abs(source[y:y+h, x:x+w].astype(np.int16) - bg).max(axis=2) > 45
        out_ink = np.abs(rendered[y:y+h, x:x+w].astype(np.int16) - bg).max(axis=2) > 45
        if src_ink.sum() < 8 or out_ink.sum() < 8:
            continue
        sy, sx = np.nonzero(src_ink)
        ry, rx = np.nonzero(out_ink)
        dx, dy = int(sx.min() - rx.min()), int(sy.min() - ry.min())
        # Keep the source evidence box fixed. Only the text frame can shift.
        if abs(dx) <= 5 and abs(dy) <= 6 and (dx or dy):
            bx, by, bw, bh = node["bbox"]
            shifted = [bx + dx, by + dy, bw, bh]
            if min(shifted[:2]) >= 0 and shifted[0] + bw <= registry.size[0] and shifted[1] + bh <= registry.size[1]:
                node["bbox"] = shifted
                adjustments.append({"id": entry["id"], "delta_px": [dx, dy], "basis": "actual_pptx_ink_alignment"})
    return adjustments


def edit_probes(result, registry, replacements, active, directory, backend):
    from edit_checks import check_deleted_visuals
    directory.mkdir()
    original_hash = sha256(result["pptx"])
    probes = []
    selected_text = next((e for e in registry.objects if e["id"] in active and e["kind"] == "text"), None)
    selected_shape = next((e for e in registry.objects if e["id"] in active and e["kind"] == "shape"), None)
    cases = []
    if selected_text:
        cases.append(ProbeCase(id="delete-text", action="delete_text", targets=[selected_text["id"]]))
    for entry in registry.objects:
        if entry["id"] in active and entry["kind"] == "image":
            cases.append(ProbeCase(id="move-" + entry["id"], action="move_group",
                                   targets=[entry["id"] + "-group"], delta_px=[80, 0]))
    if selected_shape:
        cases.append(ProbeCase(id="change-fill", action="change_fill", targets=[selected_shape["id"]], fill="E03030"))
    before = pixels(result["preview"], registry.size)
    for case in cases:
        target = directory / f"{case.id}.pptx"
        edit_copy(result["pptx"], target, case)
        rendering = backend.render(target, directory / f"{case.id}-render")
        after = pixels(rendering["pages"][0]["path"], registry.size)
        changed = np.any(before != after, axis=2)
        record = {"case": case.model_dump(), "edited_pptx": str(target),
                  "render": rendering, "changed_pixels": int(changed.sum()),
                  "status": "PASS" if changed.any() else "FAIL", "method": "direct_final_pptx_edit_and_reimport"}
        if case.action == "move_group":
            entry = next(e for e in registry.objects if e["id"] + "-group" == case.targets[0])
            x, y, w, h = entry["bbox"]
            mask = replacements[entry["id"]]["local_mask"]
            repair = replacements[entry["id"]]["repair"]
            # The move is wider than the icon in the pilots. Only check pixels not
            # covered by the moved icon's new rectangle.
            check_mask = mask.copy()
            if w > 80:
                check_mask[:, 80:] = False
            residue = np.abs(after[y:y+h, x:x+w].astype(np.int16) - repair.astype(np.int16))
            record["old_site_mae"] = round(float(residue[check_mask].mean()), 4) if check_mask.any() else None
            if record["old_site_mae"] is not None and record["old_site_mae"] > 6:
                record["status"] = "FAIL"
        probes.append(record)
    deleted = check_deleted_visuals(result, registry, replacements, active, directory, backend)
    if deleted:
        probes.append(deleted)
    if sha256(result["pptx"]) != original_hash:
        raise ValueError("direct edit probe modified the original PPTX")
    write_json(directory / "report.json", {"original_sha256": original_hash, "probes": probes,
                                            "powerPoint_checked": False, "wps_checked": False,
                                            "human_acceptance": False})
    return probes


def run_page(pages_root, page_name, output_root, *, review_file=None, ocr_cache=None,
             mae_limit=10.0, timeout=600, probes=True, align=True):
    page = load_page(resolve_page(pages_root, page_name))
    check = verify(page)
    if check.get("mae", float("inf")) > 3:
        raise ValueError("LayerD recomposition failed baseline gate")
    directory = Path(output_root).resolve() / f"{page.directory.name}-{stamp()}"
    directory.mkdir(parents=True)
    write_json(directory / "input.json", {"pack": str(page.directory), **to_record(page), "recomposition": check})
    write_json(directory / "status-started.json", {"status": "RUNNING", "source_sha256": sha256(page.source_path)})
    print(f"[{page.directory.name}] candidates -> {directory}", flush=True)
    backend = ArtifactBackend(REPO, timeout)
    registry, replacements, source, layers = prepare_candidates(page, directory, review_file=review_file, ocr_cache=ocr_cache)
    active = set(replacements)
    attempts = []
    notes = [f"LayerD three-branch page {page.directory.name}; source {page.directory.name[:2]}.png",
             "Source wording/layout retained; uncommitted evidence remains in the source-backed plate",
             "Fonts are fitted candidates; final-PPTX ROI gate applied; human and Office acceptance pending"]
    result = None
    for index in range(4):
        revision = directory / f"revisions/r{index:03d}"
        (revision / "source").mkdir(parents=True)
        shutil.copy2(page.source_path, revision / "source/page.png")
        plate, nodes, assets = materialize(source, replacements, active)
        ir = export_ir(revision, registry, nodes, plate, assets, notes=notes)
        write_json(revision / "registry-trial.json", registry.record())
        print(f"[{page.directory.name}] compile r{index:03d} objects={len(nodes)}", flush=True)
        result = compile_ir(backend, ir)
        rendered = pixels(result["preview"], page.size)
        metrics = measurements(source, rendered, registry, active)
        failed = {r["id"] for r in metrics["regions"] if r["active"] and r["mae"] > mae_limit}
        attempts.append({"revision": str(revision), "ir": str(ir), "build": result, "metrics": metrics})
        write_json(revision / "evaluation.json", metrics)
        if index == 0 and align:
            adjustments = native_text_adjustments(rendered, source, registry, replacements, active)
            if adjustments:
                write_json(revision / "alignment-adjustments.json", adjustments)
                continue
        for entry in registry.objects:
            if entry["state"] == "matched":
                registry.transition(entry, "trial_rendered", revision=str(revision), pptx_sha256=sha256(result["pptx"]))
        if not failed:
            break
        for entry in registry.objects:
            if entry["id"] in failed:
                registry.transition(entry, "reverted", reason="actual_pptx_roi_mae_exceeded", mae_limit=mae_limit)
        active -= failed
    else:
        raise RuntimeError("ROI gate did not converge; no committed delivery was produced")
    for entry in registry.objects:
        if entry["id"] in active:
            registry.transition(entry, "verified", revision=str(revision), mae_limit=mae_limit)
    probe_results = edit_probes(result, registry, replacements, active, directory / "edit-probes", backend) if probes else []
    if any(p["status"] == "FAIL" for p in probe_results):
        write_json(directory / "registry-uncommitted.json", registry.record())
        raise RuntimeError("direct PPTX edit probe failed; trial retained without commit")
    for entry in registry.objects:
        if entry["id"] in active:
            registry.transition(entry, "committed", revision=str(revision), pptx_sha256=sha256(result["pptx"]))
    registry.validate_owners(json.loads(ir.read_text(encoding="utf-8")))
    save_working_layers(directory, layers, source, replacements, active)
    write_json(directory / "registry.json", registry.record())
    write_json(directory / "operations.json", registry.operations)
    counts = {kind: sum(e["kind"] == kind and e["id"] in active for e in registry.objects) for kind in ("text", "shape", "image")}
    text_entries = [e for e in registry.objects if e["kind"] == "text"]
    summary = {"schema": "LayerDPPTDelivery/1", "page": page.directory.name, "directory": str(directory),
               "ir": str(ir), "pptx": result["pptx"], "preview": result["preview"],
               "quality": result["quality"], "source_sha256": registry.source_sha256,
               "pptx_sha256": sha256(result["pptx"]), "mae_limit": mae_limit,
               "counts": counts, "candidate_count": len(registry.objects),
               "raster_retained": [{"id": e["id"], "kind": e["kind"], "reason": e.get("reason")} for e in registry.objects if e["id"] not in active],
               "text_review": {"lines": len(text_entries), "source_checked": sum(bool(e["content_checked"]) for e in text_entries),
                               "character_accuracy": None, "critical_text_acceptance": "REVIEW_REQUIRED"},
               "metrics": metrics, "baseline": check, "attempts": attempts,
               "edit_probes": probe_results, "status": "REVIEW_REQUIRED",
               "powerPoint_checked": False, "wps_checked": False, "human_acceptance": False}
    # This exclusive final record is the commit marker. Interrupted runs have none.
    write_json(directory / "delivery.json", summary)
    print(f"[{page.directory.name}] committed {counts} MAE={metrics['full_page_mae']}", flush=True)
    return summary


def batch(pages_root, output_root, *, reviews=None, pages=None, resume=None, timeout=600, probes=True):
    directory = Path(output_root).resolve() / f"batch-{stamp()}"
    directory.mkdir(parents=True)
    write_json(directory / "baseline.json", baseline(pages_root))
    names = [p.name for p in sorted(Path(pages_root).iterdir()) if p.is_dir() and (p / "manifest.json").exists()]
    if pages:
        names = [resolve_page(pages_root, p).name for p in pages]
    prior = {}
    if resume:
        previous = json.loads(Path(resume).read_text(encoding="utf-8"))
        for old in previous.get("pages", []):
            prior[old["page"]] = old
    results, failures = [], []
    for name in names:
        review_file = Path(reviews) / f"{name[:2]}.json" if reviews else None
        if review_file and not review_file.exists():
            review_file = None
        try:
            old = prior.get(name)
            if old:
                source = Path(pages_root) / name / "source.png"
                if old["source_sha256"] != sha256(source) or old["pptx_sha256"] != sha256(old["pptx"]):
                    raise ValueError("resume input or completed output hash changed")
                saved_input = json.loads((Path(old["directory"]) / "input.json").read_text(encoding="utf-8"))
                if saved_input["manifest_sha256"] != sha256(Path(pages_root) / name / "manifest.json"):
                    raise ValueError("resume LayerD pack changed")
                # Reuse only when the original review decision also matches.
                candidate_file = Path(old["directory"]) / "candidates.json"
                prior_review = json.loads(candidate_file.read_text(encoding="utf-8"))["review"]
                new_review = json.loads(review_file.read_text(encoding="utf-8")) if review_file else {"objects": [], "text_corrections": {}}
                if prior_review != new_review:
                    old = None
            result = old or run_page(pages_root, name, directory / "pages", review_file=review_file, timeout=timeout, probes=probes)
            results.append(result)
            write_json(directory / f"progress-{len(results):02d}.json", {"pages": results, "failures": failures})
        except Exception as error:
            failures.append({"page": name, "error": str(error)})
            print(f"[{name}] FAILED {error}", flush=True)
    summary = {"schema": "LayerDPPTBatch/1", "directory": str(directory), "pages": results,
               "failures": failures, "status": "REVIEW_REQUIRED", "human_acceptance": False}
    if not failures:
        backend = ArtifactBackend(REPO, timeout)
        summary["deck"] = execute([backend.node, ROOT / "compiler_bridge.mjs", "compose", directory / "deck",
                                   *[r["ir"] for r in results]], REPO, timeout)
        # Gate the composed deck again; individual-page checks do not cover assembly.
        deck = summary["deck"]
        previews = sorted(Path(deck["preview"]).glob("*.png"))
        if len(previews) != len(results):
            raise ValueError("composed deck preview count mismatch")
        summary["compose_checks"] = []
        for item, preview in zip(results, previews):
            single = pixels(item["preview"], Image.open(preview).size)
            composed = pixels(preview, (single.shape[1], single.shape[0]))
            difference = int(np.abs(single.astype(np.int16) - composed.astype(np.int16)).max())
            summary["compose_checks"].append({"page": item["page"], "max_channel_error": difference})
            if difference:
                raise ValueError("composed page differs from verified single-page preview")
    write_json(directory / "batch.json", summary)
    return summary


def verify_deliveries(input_file, output_root, *, timeout=600):
    """Supplement older completed pages with direct group deletion and text edits."""
    from edit_checks import check_deleted_visuals, edit_final_copy
    from object_registry import ObjectRegistry
    from reconstruction import background_model, rectangle

    payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
    deliveries = payload.get("pages", [payload])
    output = Path(output_root).resolve() / f"edit-checks-{stamp()}"
    output.mkdir(parents=True)
    backend = ArtifactBackend(REPO, timeout)
    checks = []
    for delivery in deliveries:
        directory = Path(delivery["directory"])
        if sha256(delivery["pptx"]) != delivery["pptx_sha256"]:
            raise ValueError("completed delivery PPTX changed")
        registry_record = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
        registry = ObjectRegistry(registry_record["canvas_px"], registry_record["source_sha256"])
        registry.objects = registry_record["objects"]
        input_record = json.loads((directory / "input.json").read_text(encoding="utf-8"))
        page = load_page(input_record["pack"])
        if sha256(page.source_path) != registry.source_sha256:
            raise ValueError("completed delivery source changed")
        rgb = pixels(page.source_path, registry.size)
        layer = np.array(Image.open(page.layers[0].path).convert("RGBA"))
        regions = np.zeros(registry.size[::-1], np.uint16)
        for entry in registry.objects:
            regions += rectangle(entry["bbox"], registry.size)
        active = {e["id"] for e in registry.objects if e["state"] == "committed"}
        replacements = {}
        for entry in registry.objects:
            if entry["id"] not in active or entry["kind"] != "image":
                continue
            x, y, w, h = entry["bbox"]
            excluded = regions > rectangle(entry["bbox"], registry.size).astype(np.uint16)
            estimate, _ = background_model(rgb, entry["bbox"], excluded)
            if entry["repair_method"] == "validated_layerd_background":
                estimate = layer[y:y+h, x:x+w, :3]
            replacements[entry["id"]] = {"repair": estimate}
        target = output / delivery["page"]
        target.mkdir()
        visual = check_deleted_visuals(delivery, registry, replacements, active, target, backend)
        # Exercise text replacement on the retained native trial even when its
        # visual differences caused all business text to revert in the delivery.
        trial = delivery["attempts"][min(1, len(delivery["attempts"])-1)]["build"]
        objects = inspect_pptx(trial["pptx"])["slides"][0]["objects"]
        native = next((e for e in objects if e.get("text") and len(e["text"]) > 2), None)
        text_check = {"status": "NOT_RUN"}
        if native:
            edited = target / "replace-native-trial-text.pptx"
            original_hash = sha256(trial["pptx"])
            edit_final_copy(trial["pptx"], edited, replace_text=(native["name"], "编辑验证"))
            record = inspect_pptx(edited)
            current = next(e for e in record["slides"][0]["objects"] if e["name"] == native["name"])
            if current["text"] != "编辑验证" or sha256(trial["pptx"]) != original_hash:
                raise ValueError("native text replacement failed")
            rendered = backend.render(edited, target / "replace-native-trial-text-render")
            changed = bool(np.any(pixels(rendered["pages"][0]["path"], registry.size) != pixels(trial["preview"], registry.size)))
            text_check = {"status": "PASS" if changed else "FAIL", "object_id": native["name"],
                          "original_text": native["text"], "replacement_text": "编辑验证",
                          "source": "uncommitted_native_trial", "edited_pptx": str(edited),
                          "render": rendered, "fidelity_acceptance": "REVIEW_REQUIRED"}
        checks.append({"page": delivery["page"], "delivery_pptx_sha256": delivery["pptx_sha256"],
                       "delete_visuals": visual, "replace_trial_text": text_check})
        print(f"[edit-check] {delivery['page']} delete={visual and visual['status']} text={text_check['status']}", flush=True)
    failed = any((c["delete_visuals"] and c["delete_visuals"]["status"] == "FAIL") or
                 c["replace_trial_text"]["status"] == "FAIL" for c in checks)
    result = {"schema": "LayerDDirectEditChecks/1", "input": str(Path(input_file).resolve()),
              "checks": checks, "status": "FAIL" if failed else "PASS", "human_acceptance": False}
    write_json(output / "report.json", result)
    return {"directory": str(output), "report": str(output / "report.json"),
            "pages": len(checks), "status": result["status"]}
