"""Read-only, hash-bound r6/r7 baseline and evidence-chain adapter.

The only inferred filenames are format contracts: a batch's run-manifest.json
and a TypographyBatch's font-manifest.json. Parent runs are never discovered by
walking neighboring folders. Returned references are absolute and immutable.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from reconstruction_transaction import scene_fingerprint


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value, directory):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(directory) / path).resolve()


def _ref(path, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Baseline evidence file missing: " + str(path))
    actual = _hash(path)
    if expected is not None and actual != expected:
        raise ValueError("Baseline evidence hash mismatch: " + str(path))
    return {"path": str(path), "sha256": actual}


def _page_ref(page):
    return {"ir": {"path": page["ir"], "sha256": page["ir_sha256"]},
            "pptx": {"path": page["pptx"], "sha256": page["pptx_sha256"]},
            "preview": _ref(page["preview"]) if page.get("preview") else None,
            "scene_sha256": scene_fingerprint(page["ir"])["sha256"]}


def _verify_pages(batch, directory, evidence):
    pages = copy.deepcopy(batch.get("pages", []))
    ids = [page["page"] for page in pages]
    if not pages or len(ids) != len(set(ids)):
        raise ValueError("Baseline pages must be nonempty and unique")
    for page in pages:
        for field in ("ir", "pptx"):
            path = _path(page[field], directory)
            ref = _ref(path, page[field + "_sha256"])
            page[field] = ref["path"]; evidence[ref["path"]] = ref["sha256"]
        page["directory"] = str(_path(page["directory"], directory))
        ir = _read(page["ir"])
        fingerprint = scene_fingerprint(page["ir"])
        if page.get("scene_sha256") and page["scene_sha256"] != fingerprint["sha256"]:
            raise ValueError("Baseline scene fingerprint mismatch: " + page["page"])
        for relative, sha in fingerprint["files"].items():
            evidence[str((Path(page["ir"]).parent / relative).resolve())] = sha
        source = _path(ir["source_image"], Path(page["ir"]).parent)
        source_ref = _ref(source, page["source_sha256"])
        if ir.get("source_sha256") and source_ref["sha256"] != ir["source_sha256"]:
            raise ValueError("Baseline IR source fingerprint mismatch")
        page["source"] = source_ref["path"]
        for field in ("preview", "deleted_text_preview", "deleted_text_pptx"):
            if page.get(field):
                path = _path(page[field], directory)
                ref = _ref(path)
                page[field] = ref["path"]; evidence[ref["path"]] = ref["sha256"]
        for path, sha in page.get("cache_dependency_hashes", {}).items():
            ref = _ref(_path(path, directory), sha)
            evidence[ref["path"]] = ref["sha256"]
    return pages


def _text_evidence(page):
    root = Path(page["directory"]) / "text-masks"
    checks = page.get("text_checks", [])
    ids = [row["id"] for row in checks]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Frozen source text check inventory missing or duplicated: " + page["page"])
    masks = {str(path.resolve()): _hash(path) for path in sorted(root.glob("*.png"))}
    expected_glyphs = {identifier + "-glyph.png" for identifier in ids}
    if {Path(path).name for path in masks if path.endswith("-glyph.png")} != expected_glyphs:
        raise ValueError("Frozen source glyph inventory differs from text checks: " + page["page"])
    for identifier in ids:
        if str((root / (identifier + "-inspection.png")).resolve()) not in masks:
            raise ValueError("Frozen source inspection mask missing: " + identifier)
    return masks


def _probe_inventory(specs):
    """Validate a frozen inventory without silently repairing or deduplicating it."""
    if not isinstance(specs, (list, tuple)) or any(
            not isinstance(spec, (list, tuple)) or len(spec) != 2
            or any(not isinstance(value, str) or not value for value in spec) for spec in specs):
        raise ValueError("Invalid frozen edit probe inventory")
    pairs = [tuple(spec) for spec in specs]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate frozen edit probe inventory")
    return pairs


def load_baseline(input_path):
    """Return a copied normalized batch, never modify any historical artifact.

    Added page fields: evidence_baseline, baseline_refs, expected_probes, source.
    Added top field: _baseline_context (font_manifest, lineage, evidence_files).
    """
    result = _load(Path(input_path).resolve(), set())
    # Detect input changes during chain traversal, rather than freezing a mix of
    # different revisions into a superficially consistent new run.
    for path, sha in result["_baseline_context"]["evidence_files"].items():
        _ref(path, sha)
    return result


def _load(path, visiting):
    if str(path) in visiting:
        raise ValueError("Cyclic baseline provenance chain")
    visiting = visiting | {str(path)}
    batch_ref = _ref(path)
    batch = _read(path)
    schema = batch.get("schema")
    if schema not in {"TypographyBatch/1", "R7Batch/1", "R7Batch/2"}:
        raise ValueError("Unsupported baseline batch schema: " + str(schema))
    evidence = {batch_ref["path"]: batch_ref["sha256"]}
    pages = _verify_pages(batch, path.parent, evidence)
    parent_probe_floors = {}
    lineage_entry = {**batch_ref, "schema": schema}
    if schema == "TypographyBatch/1":
        fontfile = path.parent / "font-manifest.json"
        font_ref = _ref(fontfile)
        fonts = _read(fontfile)
        if not fonts.get("fonts"):
            raise ValueError("Baseline verified font manifest required")
        evidence[font_ref["path"]] = font_ref["sha256"]
        for font in fonts["fonts"]:
            ref = _ref(_path(font["file"], fontfile.parent), font["sha256"])
            evidence[ref["path"]] = ref["sha256"]
        font_ref["fonts"] = fonts
        lineage = [lineage_entry]
        for page in pages:
            evidence.update(_text_evidence(page))
            page["evidence_baseline"] = copy.deepcopy(page)
            current = _page_ref(page)
            page["baseline_refs"] = {"source": _ref(page["source"]), "r6": current,
                                     "r7": None, "current": current}
    else:
        manifest_path = path.parent / "run-manifest.json"
        manifest_ref = _ref(manifest_path)
        manifest = _read(manifest_path)
        if manifest.get("schema") not in {"R7RunInputs/1", "R7RunInputs/2"}:
            raise ValueError("R7 baseline requires its declared run manifest")
        parent = _path(batch["input"], path.parent)
        declared_parent = _path(manifest["baseline"], manifest_path.parent)
        if parent != declared_parent:
            raise ValueError("R7 batch input differs from frozen parent reference")
        _ref(parent, manifest["baseline_sha256"])
        ancestor = _load(parent, visiting)
        context = ancestor["_baseline_context"]
        font_ref = context["font_manifest"]
        if manifest.get("fonts") != font_ref["fonts"]:
            raise ValueError("R7 frozen font manifest differs from ancestor evidence")
        evidence.update(context["evidence_files"])
        evidence[manifest_ref["path"]] = manifest_ref["sha256"]
        lineage_entry["manifest"] = manifest_ref
        lineage = [*context["lineage"], lineage_entry]
        parents = {page["page"]: page for page in ancestor["pages"]}
        frozen_pages = {page["page"]: page for page in manifest.get("pages", [])}
        if ([p["page"] for p in pages] != [p["page"] for p in ancestor["pages"]]
                or len(frozen_pages) != len(manifest.get("pages", []))
                or set(frozen_pages) != set(parents)):
            raise ValueError("R7 parent page order or frozen page inventory mismatch")
        for page in pages:
            prior = parents[page["page"]]
            frozen = frozen_pages[page["page"]]
            # The immutable run manifest describes the parent's expectations.
            # A mutable result-page list cannot lower either that floor or the
            # expectations recovered recursively from source-backed ancestors.
            parent_probe_floors[page["page"]] = [
                *_probe_inventory(prior["expected_probes"]),
                *(_probe_inventory(frozen["expected_probes"]) if "expected_probes" in frozen else [])]
            if (frozen.get("scene") != scene_fingerprint(prior["ir"])
                    or frozen.get("pptx_sha256") != prior["pptx_sha256"]):
                raise ValueError("R7 frozen parent scene mismatch: " + page["page"])
            if page["source_sha256"] != prior["source_sha256"]:
                raise ValueError("R7 source differs from evidence ancestor: " + page["page"])
            if _read(page["ir"])["canvas"]["source_px"] != _read(prior["ir"])["canvas"]["source_px"]:
                raise ValueError("R7 canvas differs from evidence ancestor")
            evidence_page = copy.deepcopy(prior["evidence_baseline"])
            masks = _text_evidence(evidence_page)
            frozen_masks = {str(_path(p, manifest_path.parent)): sha
                            for p, sha in frozen.get("source_masks", {}).items()}
            if not frozen_masks or masks != frozen_masks:
                raise ValueError("R7 frozen source mask inventory or hashes changed: " + page["page"])
            evidence.update(masks)
            page["evidence_baseline"] = evidence_page
            refs = copy.deepcopy(prior["baseline_refs"])
            # First reconstructed version remains the r7 visual reference across
            # r7.1 re-runs; current always identifies the immediate actual parent.
            current = _page_ref(page)
            refs.update(source=_ref(page["source"]), current=current)
            if refs.get("r7") is None:
                refs["r7"] = current
            page["baseline_refs"] = refs
    from r7_edit_checks import probe_specs
    for page in pages:
        # Derive the mandatory floor from original source-backed r6 evidence,
        # hash-bound parent requirements, and actual current mandatory additions.
        # A disappeared target never removes the original named probe.
        required = list(dict.fromkeys([
            *probe_specs(_read(page["evidence_baseline"]["ir"]), page["page"][:2]),
            *parent_probe_floors.get(page["page"], []),
            *probe_specs(_read(page["ir"]), page["page"][:2])]))
        if "expected_probes" in page:
            specs = _probe_inventory(page["expected_probes"])
            missing = set(required) - set(specs)
            if missing:
                raise ValueError("Frozen edit probe inventory dropped mandatory requirements for " +
                                 page["page"] + ": " + repr(sorted(missing)))
        else:
            specs = required
        page["expected_probes"] = [list(pair) for pair in specs]
    result = copy.deepcopy(batch)
    result["pages"] = pages
    result["_baseline_context"] = {"input": batch_ref, "font_manifest": font_ref,
                                    "lineage": lineage, "evidence_files": evidence}
    return result
