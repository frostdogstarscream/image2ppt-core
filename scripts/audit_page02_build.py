"""Audit the page-02 diagnostic PPTX against its IR and retained assets."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from model_slideir_loop import digest, write_json

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def audit(ir_path: Path, pptx: Path, parent_assets: Path, output: Path,
          asset_review: Path | None = None,
          reviewed_ir: Path | None = None) -> dict:
    ir_path = ir_path.resolve(strict=True)
    pptx = pptx.resolve(strict=True)
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    with ZipFile(pptx) as package:
        slide = ET.fromstring(package.read("ppt/slides/slide1.xml"))
    pictures = {}
    texts = {}
    for pic in slide.findall("./p:cSld/p:spTree/p:pic", NS):
        name = pic.find("p:nvPicPr/p:cNvPr", NS).get("name")
        pictures[name] = pic
    for shape in slide.findall("./p:cSld/p:spTree/p:sp", NS):
        name = shape.find("p:nvSpPr/p:cNvPr", NS).get("name")
        texts[name] = shape
    required = (["left-panel", "bottom-banner"] +
                [f"right-panel-{i:02d}" for i in range(1, 6)] +
                [f"checkbox-{i:02d}" for i in range(1, 4)])
    assert all(name in pictures for name in required)
    assert all(node["id"] in texts for node in ir["nodes"] if node["type"] == "text")
    for node in ir["nodes"]:
        if node["type"] == "text":
            actual = "".join(t.text or "" for t in texts[node["id"]].findall(".//a:t", NS))
            assert actual == node["text"], node["id"]
    assets = {a["id"]: a for a in ir["assets"]}
    extracted = []
    for name in required:
        asset = assets["plane-" + name]
        path = ir_path.parent / asset["path"]
        assert digest(path) == asset["sha256"]
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            alpha = image.getchannel("A")
            assert alpha.getextrema() == (0, 255), name
        extracted.append({"id": name, "sha256": asset["sha256"],
                          "alpha_range": [0, 255]})
    stable = []
    for name in ["logo.png"] + [f"graphic-{i:02d}.png" for i in range(1, 9)]:
        current = ir_path.parent / "assets" / name
        prior = parent_assets / name
        assert digest(current) == digest(prior), name
        stable.append({"asset": name, "sha256": digest(current)})
    review_status = "PENDING"
    if asset_review is not None:
        if reviewed_ir is None:
            raise ValueError("asset review requires the model-reviewed parent IR")
        reviewed_ir = reviewed_ir.resolve(strict=True)
        answer = json.loads(asset_review.read_text(encoding="utf-8"))
        extraction = json.loads((reviewed_ir.parent / "extraction.json").read_text(encoding="utf-8"))
        authoring = json.loads((reviewed_ir.parent / "authoring.json").read_text(encoding="utf-8"))
        if reviewed_ir != ir_path:
            correction = json.loads((ir_path.parent / "authoring.json").read_text(encoding="utf-8"))
            assert correction["parent_ir_sha256"] == digest(reviewed_ir)
        assert answer.get("schema") == "asset-review/1"
        assert answer.get("source_sha256") == ir["source_sha256"]
        assert answer.get("plate_sha256") == extraction["old_plate_sha256"]
        assert authoring["model_review_sha256"] == digest(asset_review)
        assert {item["candidate_id"] for item in answer["objects"]
                if item.get("accept") is True} == set(required)
        review_status = "HASH_BOUND_ACCEPTED"
    result = {"status": "PASS", "source_sha256": ir["source_sha256"],
              "pptx_sha256": digest(pptx), "picture_count": len(pictures),
              "text_node_count": len(texts), "independent_planes": extracted,
              "preserved_assets": stable,
              "model_asset_review": review_status, "human_visual_acceptance": False}
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--pptx", type=Path, required=True)
    parser.add_argument("--parent-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-review", type=Path)
    parser.add_argument("--reviewed-ir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.ir, args.pptx, args.parent_assets, args.output,
                           args.asset_review, args.reviewed_ir),
                     ensure_ascii=False, indent=2))
