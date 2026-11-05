"""Audit actual paragraph styles and edit diagnostic copies of final PPTX files."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

from text_repair import NS, load_json
from rebuild_pipeline import ArtifactBackend, REPO, pixels, stamp
from slideir_adapter import write_json, sha256
import numpy as np


def verify_styles(pptx, ir, slide=1):
    with ZipFile(pptx) as package:
        root=ET.fromstring(package.read(f"ppt/slides/slide{slide}.xml"))
    shapes={element.find("p:nvSpPr/p:cNvPr",NS).get("name"):element
            for element in root.findall("p:cSld/p:spTree/p:sp",NS)}
    checked=[]
    for node in ir["nodes"]:
        if node["type"]!="text":
            continue
        shape=shapes[node["id"]]
        paragraphs=shape.findall("p:txBody/a:p",NS)
        expected=node["paragraphs"]
        assert len(paragraphs)==len(expected),node["id"]
        for paragraph,spec in zip(paragraphs,expected):
            runs=paragraph.findall("a:r",NS)
            planned=spec.get("runs",[{"text":spec["text"]}])
            assert len(runs)==len(planned),node["id"]
            for run,wanted in zip(runs,planned):
                style={**spec,**wanted}
                assert run.find("a:t",NS).text==wanted["text"],node["id"]
                props=run.find("a:rPr",NS)
                assert props.get("sz")==str(round(style["font"]["size_pt"]*100)),node["id"]
                assert props.get("b")==str(int(style["bold"])),node["id"]
                assert props.find("a:solidFill/a:srgbClr",NS).get("val")==style["color"],node["id"]
                for role in ("latin","ea"):
                    assert props.find("a:"+role,NS).get("typeface")==style["font"][role],node["id"]
            for field,tag in (("line_spacing_pt","lnSpc"),("space_before_pt","spcBef"),("space_after_pt","spcAft")):
                if field in spec:
                    actual=paragraph.find("a:pPr/a:"+tag+"/a:spcPts",NS)
                    assert actual is not None and actual.get("val")==str(round(spec[field]*100)),node["id"]
        checked.append({"id":node["id"],"paragraph_count":len(expected),"status":"PASS"})
    return checked


def delete_deck_text(source, target, pages):
    rewritten={}
    with ZipFile(source) as original:
        for index,delivery in enumerate(pages,1):
            ir=load_json(delivery["ir"])
            names={n["id"] for n in ir["nodes"] if n["type"]=="text"}
            part=f"ppt/slides/slide{index}.xml"
            root=ET.fromstring(original.read(part))
            tree=root.find("p:cSld/p:spTree",NS)
            removed=set()
            for element in list(tree):
                name=element.find(".//p:cNvPr",NS)
                if name is not None and name.get("name") in names:
                    tree.remove(element);removed.add(name.get("name"))
            assert removed==names,part
            rewritten[part]=ET.tostring(root,encoding="utf-8",xml_declaration=True)
        with ZipFile(target,"x",compression=ZIP_DEFLATED) as output:
            for entry in original.infolist():
                output.writestr(entry,rewritten.get(entry.filename,original.read(entry.filename)))


def edit_text_copy(source, target, node_id):
    with ZipFile(source) as original:
        root=ET.fromstring(original.read("ppt/slides/slide1.xml"))
        shape=next(s for s in root.findall("p:cSld/p:spTree/p:sp",NS)
                   if s.find("p:nvSpPr/p:cNvPr",NS).get("name")==node_id)
        position=shape.find("p:spPr/a:xfrm/a:off",NS)
        for axis,delta in (("x",20),("y",12)):
            position.set(axis,str(int(position.get(axis))+delta*9525))
        for number,paragraph in enumerate(shape.findall("p:txBody/a:p",NS),1):
            texts=paragraph.findall("a:r/a:t",NS)
            texts[0].text=f"第{number}行已替换为连续编辑验证内容"
            for text in texts[1:]:
                text.text=""
        with ZipFile(target,"x",compression=ZIP_DEFLATED) as output:
            for entry in original.infolist():
                output.writestr(entry,ET.tostring(root,encoding="utf-8",xml_declaration=True)
                                if entry.filename=="ppt/slides/slide1.xml" else original.read(entry.filename))
    with ZipFile(source) as before,ZipFile(target) as after:
        assert before.namelist()==after.namelist()
        assert all(before.read(name)==after.read(name) for name in before.namelist() if name!="ppt/slides/slide1.xml")


def audit_batch(file, timeout=600):
    batch=load_json(file)
    directory=Path(batch["directory"])/("acceptance-"+stamp())
    directory.mkdir()
    backend=ArtifactBackend(REPO,timeout)
    deleted=directory/"all-native-text-deleted.pptx"
    delete_deck_text(batch["deck"]["pptx"],deleted,batch["pages"])
    deleted_render=backend.render(deleted,directory/"all-native-text-deleted-render")
    records=[]
    for index,delivery in enumerate(batch["pages"],1):
        ir=load_json(delivery["ir"])
        styles=verify_styles(batch["deck"]["pptx"],ir,index)
        record={"page":delivery["page"],"style_checks":styles,"pptx_sha256":sha256(delivery["pptx"])}
        size=ir["canvas"]["source_px"]
        naked=deleted_render["pages"][index-1]["path"]
        difference=float(np.abs(pixels(naked,size).astype(np.int16)-pixels(delivery["deleted_text_preview"],size).astype(np.int16)).mean())
        assert difference<=0.1,"composed naked background changed"
        record["deleted_text_preview"]=naked
        record["composed_background_mae_vs_page_check"]=difference
        if delivery["page"][:2] in {"03","04","10"}:
            node=next(n for n in ir["nodes"] if n["type"]=="text" and len(n["paragraphs"])>1)
            target=directory/(delivery["page"][:2]+"-moved-replaced.pptx")
            edit_text_copy(delivery["pptx"],target,node["id"])
            rendered=backend.render(target,directory/(delivery["page"][:2]+"-edited-render"))
            size=ir["canvas"]["source_px"]
            difference=float(np.abs(pixels(rendered["pages"][0]["path"],size).astype(np.int16)-pixels(delivery["preview"],size).astype(np.int16)).mean())
            assert difference>0,"edit did not change actual rendering"
            record["edit_probe"]={"target":node["id"],"move_px":[20,12],"replacement":True,
                                  "actual_render_difference":difference,"pptx":str(target.resolve()),
                                  "preview":rendered["pages"][0]["path"],"status":"PASS"}
        records.append(record)
    result={"schema":"TextRepairAcceptance/1","pages":records,"automated_styles":"PASS",
            "deleted_text_pptx":str(deleted.resolve()),"directory":str(directory.resolve()),
            "powerpoint_checked":False,"wps_checked":False,"human_acceptance":False}
    write_json(directory/"report.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("batch")
    args=parser.parse_args()
    result=audit_batch(args.batch)
    print(f"Audited {len(result['pages'])} pages in {result['directory']}")
