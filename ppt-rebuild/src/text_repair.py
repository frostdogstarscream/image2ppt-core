"""Text-first reconstruction from saved, source-backed LayerD page deliveries.

Original images and completed versions stay immutable. Each run has a text
inventory, pixel masks, model receipts, paragraph mapping and direct PPTX probes.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from font_matcher import fit_visual
from layerd_loader import load_page
from ocr_engine import detect
from reconstruction import background_model, intersect, padded, rectangle
from slideir_adapter import sha256, write_json
from text_mask import composite, grow, remote_inpaint
from rebuild_pipeline import ArtifactBackend, REPO, ROOT, compile_ir, pixels, stamp

NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def load_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def union_bbox(boxes):
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return [left, top, right-left, bottom-top]


def source_inventory(delivery, source, decisions):
    directory = Path(delivery["directory"])
    original = load_json(directory / "ocr.json")
    review = load_json(directory / "candidates.json")["review"]
    trial = load_json(delivery["attempts"][min(1, len(delivery["attempts"])-1)]["ir"])
    styles = {n["id"].removeprefix("text-"): n for n in trial["nodes"] if n["type"] == "text"}
    images = review["objects"]
    records = []
    corrections = copy.deepcopy(review.get("text_corrections", {}))
    known_ids={line["id"] for line in original["lines"]}
    if set(decisions.get("lines", {}))-known_ids:
        raise ValueError("text repair review references unknown source lines")
    for line_id,change in decisions.get("lines", {}).items():
        corrections[line_id]={**corrections.get(line_id,{}),**change}
    for line in original["lines"]:
        correction = corrections.get(line["id"], {})
        bbox = correction.get("bbox", line["bbox"])
        text = correction.get("text", line["text"])
        if correction.get("skip") or text.strip() in {">>", "》》", "→", "»"}:
            continue
        if any(intersect(bbox, image["bbox"]) >= bbox[2]*bbox[3]*0.95 for image in images if image["kind"] == "image"):
            continue
        # Source corrections remove decorative chevrons only when the saved
        # review supplies both the corrected text and the narrowed source box.
        node = styles.get(line["id"])
        if node:
            style = {key: copy.deepcopy(node[key]) for key in ("font", "color", "bold", "align") if key in node}
        else:
            estimate, _ = background_model(source, bbox, np.zeros(source.shape[:2], bool))
            fit = fit_visual(source, estimate, text, bbox, dpi=96)
            crop = source[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            diff = np.abs(crop.astype(np.int16)-estimate.astype(np.int16)).max(axis=2)
            foreground = crop[diff >= max(20, float(np.percentile(diff, 80)))]
            color = np.median(foreground, axis=0).round().astype(int) if len(foreground) else [20,20,20]
            style = {"font": {"latin": "Arial", "ea": fit.get("family", "微软雅黑"),
                              "size_pt": fit.get("size_pt") or bbox[3]*0.75},
                     "bold": bool(fit.get("bold")), "color": "".join(f"{v:02X}" for v in color), "align": "left"}
        style.setdefault("bold", False)
        style.setdefault("align", "left")
        style.update(correction.get("style", {}))
        records.append({"id": line["id"], "text": text, "bbox": bbox,
                        "ocr_polygon": line.get("polygon"), "ocr_text": line["text"],
                        "style": style, "style_origin": "existing_native_object" if node else "existing_local_font_matcher",
                        "runs": correction.get("runs"), "source_checked": line["id"] in review.get("checked_lines", []),
                        "mask_padding":correction.get("mask_padding",3),
                        "mask_dilation":correction.get("mask_dilation",2),
                        "frame_bbox":correction.get("frame_bbox",bbox if "bbox" in correction else node["bbox"] if node else bbox)})
    for item in decisions.get("additional_lines", []):
        if item["id"] in known_ids:
            raise ValueError("additional text line ID already exists")
        known_ids.add(item["id"])
        record = copy.deepcopy(item)
        if "style" not in record:
            style_from = record.pop("style_from")
            parent = next(r for r in records if r["id"] == style_from)
            record["style"] = copy.deepcopy(parent["style"])
        record.update(style_origin="saved_source_review", source_checked=True)
        record.setdefault("frame_bbox",record["bbox"])
        records.append(record)
    return sorted(records, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def paragraph_groups(lines, explicit):
    by_id = {line["id"]: line for line in lines}
    used = set()
    groups = []
    for group in explicit:
        if any(i not in by_id or i in used for i in group["lines"]):
            raise ValueError("paragraph review references absent or duplicate source lines")
        groups.append({"id": group["id"], "lines": [by_id[i] for i in group["lines"]]})
        used.update(group["lines"])
    for first in lines:
        if first["id"] in used:
            continue
        members = [first]
        used.add(first["id"])
        while True:
            previous = members[-1]
            x, y, w, h = previous["bbox"]
            candidates = []
            for line in lines:
                if line["id"] in used:
                    continue
                bx, by, bw, bh = line["bbox"]
                gap = by-(y+h)
                if not (-2 <= gap <= max(h,bh)*0.75 and 0.7 <= bh/h <= 1.35 and abs(bx-x) <= 22):
                    continue
                if min(x+w,bx+bw)-max(x,bx) < min(w,bw)*0.6:
                    continue
                a = np.array([int(previous["style"]["color"][i:i+2],16) for i in (0,2,4)])
                b = np.array([int(line["style"]["color"][i:i+2],16) for i in (0,2,4)])
                if np.abs(a-b).max() > 65:
                    continue
                candidates.append(line)
            if not candidates:
                break
            following = min(candidates, key=lambda r: r["bbox"][1])
            members.append(following)
            used.add(following["id"])
        groups.append({"id": "paragraph-"+first["id"], "lines": members})
    return groups


def glyph_mask(source, line, protection):
    """Use bounded source-color stroke growth shared with surface rebuilding."""
    from background_models import precise_text_masks

    return precise_text_masks(source, line, protection)["glyph"]


def independent_residue(source, cleaned, mask):
    """Measure remaining glyph-shaped high frequency signal in the actual patch."""
    gray = cv2.cvtColor(cleaned,cv2.COLOR_RGB2GRAY).astype(np.float32)
    high = np.abs(gray-cv2.GaussianBlur(gray,(0,0),2.0))
    original = cv2.cvtColor(source,cv2.COLOR_RGB2GRAY).astype(np.float32)
    original_high = np.abs(original-cv2.GaussianBlur(original,(0,0),2.0))
    core = mask & (original_high > 12)
    if not core.any():
        return {"original_glyph_pixels":0,"residual_fraction":1.0,"residual_p95":0.0,
                "status":"REVIEW_REQUIRED","reason":"no_original_glyph_evidence"}
    return {"original_glyph_pixels":int(core.sum()),
            "residual_fraction":round(float((high[core]>6).mean()),5),
            "residual_p95":round(float(np.percentile(high[core],95)),3)}


def residue_inspection_mask(source, line, mask):
    """Check the complete source text box and foreground halos outside it.

    Mask dilation can touch a card edge. That edge is background evidence,
    whereas the text box and its foreground-colored fringe are glyph evidence.
    """
    foreground=np.array([int(line["style"]["color"][i:i+2],16) for i in (0,2,4)])
    fringe=np.abs(source.astype(np.int16)-foreground).max(axis=2)<=70
    box=rectangle(line["bbox"],(source.shape[1],source.shape[0]))
    _,labels=cv2.connectedComponents((fringe&mask).astype(np.uint8),8)
    selected=np.unique(labels[fringe&box&mask]);selected=selected[selected!=0]
    return mask & (box | np.isin(labels,selected))


def clean_text(source, layers, mask, *, inpaint_url, output, regions=None):
    """Use clean lower pixels if possible; model repair handles the remaining mask."""
    lower = composite([(0,layers[0])], (source.shape[1],source.shape[0]))
    lower_check = independent_residue(source,lower,mask)
    if not regions and lower_check["residual_fraction"] <= 0.01:
        repaired = lower
        method = "verified_layerd_lower_composite"
    else:
        repaired = source.copy()
        calls=[]
        model_mask=np.zeros(mask.shape,bool)
        if regions:
            for region in regions:
                target=region["mask"]
                if not target.any():
                    continue
                yy,xx=np.nonzero(target)
                box=[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
                bx,by,bw,bh=box
                local_target=target[by:by+bh,bx:bx+bw]
                background_error=None
                try:
                    expected,_=background_model(source,box,mask)
                    delta=np.abs(lower[by:by+bh,bx:bx+bw].astype(np.int16)-expected.astype(np.int16))[local_target]
                    background_error=float(delta.mean())
                    lower_valid=background_error<=3 and float(np.percentile(delta,95))<=8
                except ValueError:
                    lower_valid=False
                lower_glyph=independent_residue(source,lower,target)
                if lower_valid and lower_glyph["residual_fraction"]<=0.01:
                    repaired[target]=lower[target]
                    calls.append({"paragraph":region["id"],"method":"verified_layerd_lower_composite",
                                  "background_mae":round(background_error,3),"mask_pixels":int(target.sum())})
                    continue
                model_mask|=target
                calls.append({"paragraph":region["id"],"method":"big_lama_full_context_masked_inpainting",
                              "rejected_lower_background_mae":round(background_error,3) if background_error is not None else None,
                              "context_bbox":[0,0,source.shape[1],source.shape[0]],"mask_pixels":int(target.sum())})
            if model_mask.any():
                patch=remote_inpaint(source,mask,inpaint_url,timeout=300)
                repaired[model_mask]=patch[model_mask]
        else:
            repaired = remote_inpaint(source,mask,inpaint_url,timeout=300)
        write_json(output/"inpaint-local-calls.json",calls if regions else [{"context":"full_source"}])
        method = "per_paragraph_verified_lower_or_big_lama"
    if repaired.shape != source.shape:
        raise ValueError("inpainting response changed source dimensions")
    clean = source.copy()
    clean[mask] = repaired[mask]
    if np.any(clean[~mask] != source[~mask]):
        raise ValueError("inpainting changed protected/outside pixels")
    write_json(output / "inpaint-receipt.json", {
        "method":method,"mask_pixels":int(mask.sum()),"input_size":[source.shape[1],source.shape[0]],
        "lower_layer_check":lower_check,"result_check":independent_residue(source,clean,mask),
        "outside_mask_changed_pixels":0,"endpoint":inpaint_url})
    return clean


def merged_nodes(groups):
    nodes = []
    for group in groups:
        lines = group["lines"]
        bbox = union_bbox([r["frame_bbox"] for r in lines])
        paragraphs=[]
        for index,line in enumerate(lines):
            spec = {"text":line["text"], **copy.deepcopy(line["style"]),
                    "space_before_pt":0,"space_after_pt":0,
                    "left_indent_px":max(0,line["frame_bbox"][0]-bbox[0])}
            if index+1<len(lines):
                spec["line_spacing_pt"]=(lines[index+1]["frame_bbox"][1]-line["frame_bbox"][1])*0.75
            if line.get("runs"):
                spec["runs"]=line["runs"]
            paragraphs.append(spec)
        # A little descender space prevents multiline frames clipping on import.
        node={"id":group["id"],"type":"text","bbox":bbox,"z":0,
              "evidence_ids":["text-evidence-"+r["id"] for r in lines],
              "text":"\n".join(r["text"] for r in lines), **copy.deepcopy(lines[0]["style"]),
              "paragraphs":paragraphs,"wrap":False,"vertical_anchor":"middle"}
        nodes.append(node)
    return nodes


def write_revision(delivery, source_file, cleaned_plate, lines, groups, directory):
    base_file=Path(delivery["ir"])
    base=load_json(base_file)
    directory.mkdir(parents=True)
    (directory/"source").mkdir()
    (directory/"assets").mkdir()
    shutil.copy2(source_file,directory/"source/page.png")
    ir=copy.deepcopy(base)
    old_text={e["id"] for e in ir["evidence"] if e["kind"]=="text"}
    ir["evidence"]=[e for e in ir["evidence"] if e["id"] not in old_text]
    ir["nodes"]=[n for n in ir["nodes"] if n["type"]!="text"]
    for asset in ir["assets"]:
        relative="assets/"+asset["id"]+".png"
        if asset["id"]=="plate":
            cleaned_plate.save(directory/relative)
        else:
            shutil.copy2(base_file.parent/asset["path"],directory/relative)
        asset["path"]=relative
        asset["sha256"]=sha256(directory/relative)
        asset["contains_evidence_ids"]=[i for i in asset["contains_evidence_ids"] if i not in old_text]
    for line in lines:
        ir["evidence"].append({"id":"text-evidence-"+line["id"],"kind":"text","bbox":line["bbox"],"text":line["text"]})
    native_lines={line["id"] for group in groups for line in group["lines"]}
    plate_asset=next(asset for asset in ir["assets"] if asset["id"]=="plate")
    plate_asset["contains_evidence_ids"].extend("text-evidence-"+line["id"] for line in lines if line["id"] not in native_lines)
    text_nodes=merged_nodes(groups)
    z=max(n["z"] for n in ir["nodes"])+1
    for node in text_nodes:
        node["z"]=z;z+=1
        # Frame padding is allowed only inside the actual page bounds.
        node["bbox"][3]=min(ir["canvas"]["source_px"][1]-node["bbox"][1],node["bbox"][3])
        ir["nodes"].append(node)
    ir["review_status"]="needs_review"
    ir["review_notes"]=["Text-first repair; original per-line font attributes preserved",
                         "Source glyphs removed using pixel masks and verified lower layers / Big-LaMa",
                         "Visual font differences require review; text editability is verified separately"]
    target=directory/"slideir.json"
    write_json(target,ir)
    write_json(directory/"authoring.json",{"tool":"Codex image2ppt text-repair","model_id":"current-conversation",
        "created_at":datetime.now(timezone.utc).isoformat(),"ir_sha256":sha256(target),
        "decisions":ir["review_notes"]})
    return target


def delete_text_copy(source,target,names):
    with ZipFile(source) as original:
        root=ET.fromstring(original.read("ppt/slides/slide1.xml"))
        tree=root.find("p:cSld/p:spTree",NS)
        removed=[]
        for element in list(tree):
            nv=element.find(".//p:cNvPr",NS)
            if nv is not None and nv.get("name") in names:
                tree.remove(element);removed.append(nv.get("name"))
        if set(removed)!=set(names):
            raise ValueError("text deletion probe targets missing")
        with ZipFile(target,"x",compression=ZIP_DEFLATED) as output:
            for entry in original.infolist():
                output.writestr(entry, ET.tostring(root,encoding="utf-8",xml_declaration=True)
                                if entry.filename=="ppt/slides/slide1.xml" else original.read(entry.filename))


def repair_page(delivery, output_root, *, inpaint_url, decisions=None, timeout=600):
    decisions=decisions or {}
    directory=Path(output_root)/f"{delivery['page']}-{stamp()}"
    directory.mkdir(parents=True)
    old_directory=Path(delivery["directory"])
    pack=load_page(load_json(old_directory/"input.json")["pack"])
    source=pixels(pack.source_path,pack.size)
    base_ir=load_json(delivery["ir"])
    base_plate_asset=next(a for a in base_ir["assets"] if a["id"]=="plate")
    base_plate=pixels(Path(delivery["ir"]).parent/base_plate_asset["path"],pack.size)
    if decisions.get("source_sha256") and decisions["source_sha256"]!=sha256(pack.source_path):
        raise ValueError("text repair review source hash mismatch")
    print(f"[text-repair] {delivery['page']} inventory",flush=True)
    lines=source_inventory(delivery,source,decisions)
    groups=paragraph_groups(lines,decisions.get("paragraphs",[]))
    protection=np.zeros(source.shape[:2],bool)
    for node in base_ir["nodes"]:
        if node["type"]=="image" and node["id"]!="background-image":
            protection |= rectangle(node["bbox"],pack.size)
        elif node["type"]=="shape" and node.get("fill")!="none":
            protection |= rectangle(node["bbox"],pack.size)
    from background_models import precise_text_masks
    mask_details={line["id"]:precise_text_masks(source,line,protection) for line in lines}
    # Keep stroke evidence separate from the authorized repair footprint.
    # Passing only glyph pixels to LaMa leaves antialiased edges and shadows
    # outside its hole, even though the bounded repair mask already covers them.
    masks={key:value["glyph"] for key,value in mask_details.items()}
    repair_masks={key:value["repair"] for key,value in mask_details.items()}
    inspection={line["id"]:residue_inspection_mask(source,line,masks[line["id"]]) for line in lines}
    union=np.logical_or.reduce(list(repair_masks.values()))
    (directory/"masks").mkdir()
    for line in lines:
        Image.fromarray(masks[line["id"]].astype(np.uint8)*255).save(directory/"masks"/(line["id"]+".png"))
        Image.fromarray(repair_masks[line["id"]].astype(np.uint8)*255).save(directory/"masks"/(line["id"]+"-repair.png"))
        Image.fromarray(inspection[line["id"]].astype(np.uint8)*255).save(directory/"masks"/(line["id"]+"-inspection.png"))
    Image.fromarray(union.astype(np.uint8)*255).save(directory/"text-mask.png")
    write_json(directory/"text-inventory.json",{"source_sha256":sha256(pack.source_path),"lines":lines})
    write_json(directory/"paragraph-map.json",[{"id":g["id"],"source_line_ids":[r["id"] for r in g["lines"]]} for g in groups])
    layer_arrays=[np.array(Image.open(layer.path).convert("RGBA")) for layer in pack.layers]
    print(f"[text-repair] {delivery['page']} inpaint lines={len(lines)} paragraphs={len(groups)}",flush=True)
    regions=[]
    for group in groups:
        selected=np.logical_or.reduce([repair_masks[r["id"]] for r in group["lines"]])
        color=group["lines"][0]["style"]["color"]
        foreground=np.mean([int(color[i:i+2],16) for i in (0,2,4)])
        regions.append({"id":group["id"],"mask":selected,"bright":foreground>210})
    cleaned=clean_text(source,layer_arrays,union,inpaint_url=inpaint_url,output=directory,regions=regions)
    committed=[]
    pending=[]
    for group in groups:
        target=np.logical_or.reduce([masks[line["id"]] for line in group["lines"]])
        inspected=np.logical_or.reduce([inspection[line["id"]] for line in group["lines"]])
        check=independent_residue(source,cleaned,inspected)
        mask_density=max(float(masks[line["id"]][rectangle(line["bbox"],pack.size)].mean()) for line in group["lines"])
        if target.any() and mask_density<=0.8 and check["residual_fraction"]<=0.01:
            committed.append(group)
        else:
            pending.append({"id":group["id"],"reason":"source_patch_residue_or_empty_mask",**check})
    # The existing plate has the already-tested icons and geometry removed.
    # Only repaired text pixels are changed, keeping those owners independent.
    plate=base_plate.copy();plate[union]=cleaned[union]
    for group in groups:
        if group not in committed:
            region=rectangle(union_bbox([line["bbox"] for line in group["lines"]]),pack.size)&~protection
            plate[region]=source[region]
    Image.fromarray(plate).save(directory/"cleaned-plate.png")
    ImageEnhance.Contrast(Image.fromarray(plate)).enhance(3).save(directory/"cleaned-plate-contrast.png")
    working=[array.copy() for array in layer_arrays]
    working[0][union,:3]=cleaned[union];working[0][union,3]=255
    for layer in working[1:]:layer[union,3]=0
    (directory/"working-layers").mkdir()
    for z,layer in enumerate(working):Image.fromarray(layer).save(directory/"working-layers"/f"layer-{z:02d}.png")
    ir=write_revision(delivery,pack.source_path,Image.fromarray(plate),lines,committed,directory/"revision")
    backend=ArtifactBackend(REPO,timeout)
    print(f"[text-repair] {delivery['page']} compile",flush=True)
    result=compile_ir(backend,ir)
    no_text=directory/"all-native-text-deleted.pptx"
    names=[g["id"] for g in committed]
    delete_text_copy(result["pptx"],no_text,names)
    rendering=backend.render(no_text,directory/"all-native-text-deleted-render")
    naked=pixels(rendering["pages"][0]["path"],pack.size)
    # Pixel comparison against the actual repaired output detects any hidden
    # extra text emitted by the compiler, independent of source-vs-font drift.
    glyph_reports=[]
    for group in groups:
        mask=np.logical_or.reduce([inspection[r["id"]] for r in group["lines"]])
        check=independent_residue(source,naked,mask)
        glyph_reports.append({"id":group["id"],"source_line_ids":[r["id"] for r in group["lines"]],**check,
                              "status":"PASS" if group in committed and check["residual_fraction"]<=0.01 else "REVIEW_REQUIRED"})
    # A final-render failure invalidates the whole native replacement. Retain
    # this version as diagnostics and stop before it can enter the batch deck.
    final_failures=[r["id"] for r in glyph_reports if r["status"]!="PASS" and r["id"] in names]
    if final_failures:
        write_json(directory/"final-render-failure.json",{"paragraphs":final_failures,"checks":glyph_reports})
        raise ValueError("final rendered text removal failed: "+", ".join(final_failures))
    # OCR is evidence in addition to source-stroke checks, never the sole gate.
    remaining=detect(rendering["pages"][0]["path"],min_score=0.45)
    ocr_hits=[]
    for hit in remaining:
        if any(intersect(hit.bbox,line["bbox"])>hit.bbox[2]*hit.bbox[3]*0.5 for line in lines):
            ocr_hits.append({"text":hit.text,"bbox":hit.bbox,"score":hit.score})
    if ocr_hits:
        write_json(directory/"ocr-residue-failure.json",{"hits":ocr_hits,"status":"FAIL"})
        raise ValueError("OCR detected residual text in native replacement regions")
    coverage={"source_lines":len(lines),"native_textboxes":len(committed),"merged_paragraphs":sum(len(g["lines"])>1 for g in committed),
              "pending_paragraphs":sum(r["status"]!="PASS" for r in glyph_reports),"ocr_residue_hits":ocr_hits}
    rendered=pixels(result["preview"],pack.size)
    visual_mae=float(np.abs(rendered.astype(np.int16)-source.astype(np.int16)).mean())
    report={"schema":"TextRepairDelivery/1","page":delivery["page"],"directory":str(directory.resolve()),
            "source_sha256":sha256(pack.source_path),"ir":str(ir.resolve()),**result,
            "coverage":coverage,"paragraph_checks":glyph_reports,"pending":pending,"visual_mae":round(visual_mae,4),
            "deleted_text_pptx":str(no_text.resolve()),"deleted_text_preview":rendering["pages"][0]["path"],
            "font_policy":"preserve_existing_line_attributes","status":"REVIEW_REQUIRED",
            "powerPoint_checked":False,"wps_checked":False,"human_acceptance":False}
    write_json(directory/"delivery.json",report)
    print(f"[text-repair] {delivery['page']} {coverage}",flush=True)
    return report


def repair_batch(input_file, output_root, *, pages=None, reviews=None, inpaint_url="http://127.0.0.1:18842/inpaint", timeout=600):
    original=load_json(input_file)
    deliveries=original.get("pages",[original])
    if pages:deliveries=[next(d for d in deliveries if d["page"].startswith(prefix)) for prefix in pages]
    directory=Path(output_root).resolve()/f"text-repair-{stamp()}";directory.mkdir(parents=True)
    results=[]
    for delivery in deliveries:
        review_file=Path(reviews)/(delivery["page"][:2]+".json") if reviews else None
        decisions=load_json(review_file) if review_file and review_file.exists() else {}
        results.append(repair_page(delivery,directory/"pages",inpaint_url=inpaint_url,decisions=decisions,timeout=timeout))
        write_json(directory/f"progress-{len(results):02d}.json",{"pages":results})
    backend=ArtifactBackend(REPO,timeout)
    from image2ppt.compiler import execute
    deck=execute([backend.node,ROOT/"compiler_bridge.mjs","compose",directory/"deck",*[r["ir"] for r in results]],REPO,timeout)
    summary={"schema":"TextRepairBatch/1","directory":str(directory),"pages":results,"deck":deck,"status":"REVIEW_REQUIRED"}
    write_json(directory/"batch.json",summary)
    return summary
