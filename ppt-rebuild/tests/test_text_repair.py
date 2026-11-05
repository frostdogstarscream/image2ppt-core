"""Text masks, paragraph ownership and actual-pixel residue regressions."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _helpers import SRC
from text_repair import glyph_mask, independent_residue, paragraph_groups, merged_nodes, clean_text, residue_inspection_mask


def test_white_glyph_mask_preserves_blue_banner_surface():
    image=Image.new("RGB",(300,100),"white")
    draw=ImageDraw.Draw(image)
    draw.rectangle((10,20,290,78),fill=(12,100,180))
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",32)
    draw.text((30,26),"WHITE",font=font,fill="white")
    source=np.array(image)
    line={"bbox":[28,28,118,45],"style":{"color":"FFFFFF"}}
    mask=glyph_mask(source,line,np.zeros(source.shape[:2],bool))
    assert mask.any()
    assert mask.sum()<118*45*.65
    assert not mask[25,140]
    protected=np.zeros(source.shape[:2],bool);protected[:,35:45]=True
    assert not glyph_mask(source,line,protected)[protected].any()


def test_residue_is_measured_from_result_pixels():
    source=np.full((60,120,3),240,np.uint8)
    source[20:40,30:34]=10;source[20:24,30:70]=10
    mask=np.zeros(source.shape[:2],bool);mask[17:43,27:73]=True
    unchanged=independent_residue(source,source,mask)
    erased=independent_residue(source,np.full_like(source,240),mask)
    assert unchanged["residual_fraction"]>.1
    assert erased["residual_fraction"]==0


def test_merged_textbox_keeps_each_line_style_and_evidence():
    lines=[{"id":"a","text":"第一行","bbox":[10,20,100,24],"frame_bbox":[8,18,104,28],
            "style":{"font":{"latin":"Arial","ea":"黑体","size_pt":18},"color":"112233","bold":False}},
           {"id":"b","text":"第二行","bbox":[12,52,100,24],"frame_bbox":[10,50,104,28],
            "style":{"font":{"latin":"Arial","ea":"宋体","size_pt":17},"color":"223344","bold":True}}]
    groups=paragraph_groups(lines,[{"id":"body","lines":["a","b"]}])
    node=merged_nodes(groups)[0]
    assert node["text"]=="第一行\n第二行"
    assert node["evidence_ids"]==["text-evidence-a","text-evidence-b"]
    assert [p["font"]["ea"] for p in node["paragraphs"]]==["黑体","宋体"]
    assert node["paragraphs"][1]["left_indent_px"]==2
    assert node["paragraphs"][0]["line_spacing_pt"]==24


def test_inpaint_output_is_clipped_to_mask(tmp_path,monkeypatch):
    import text_repair
    source=np.full((60,120,3),240,np.uint8);source[20:40,40:45]=10
    mask=np.zeros(source.shape[:2],bool);mask[17:43,37:48]=True
    layer=np.dstack((source,np.full(source.shape[:2],255,np.uint8)))
    monkeypatch.setattr(text_repair,"remote_inpaint",lambda *a,**k:np.full_like(source,220))
    result=clean_text(source,[layer],mask,inpaint_url="test-local",output=tmp_path)
    assert np.array_equal(result[~mask],source[~mask])
    assert np.all(result[mask]==220)


def test_glyph_free_lower_layer_with_wrong_card_color_requires_model(tmp_path,monkeypatch):
    import text_repair
    source=np.full((100,180,3),(40,130,210),np.uint8)
    source[40:60,70:75]=255
    mask=np.zeros(source.shape[:2],bool);mask[37:63,67:78]=True
    lower=np.full((100,180,4),255,np.uint8)
    calls=[]
    def inpaint(image,target,*args,**kwargs):
        calls.append(target.copy())
        return np.full_like(image,(40,130,210))
    monkeypatch.setattr(text_repair,"remote_inpaint",inpaint)
    result=clean_text(source,[lower],mask,inpaint_url="local",output=tmp_path,
                      regions=[{"id":"label","mask":mask,"bright":True}])
    assert len(calls)==1
    assert np.all(result[mask]==(40,130,210))
    assert np.array_equal(result[~mask],source[~mask])


def test_residue_inspection_distinguishes_glyph_fringe_from_disconnected_card_edge():
    source=np.full((60,100,3),(50,110,180),np.uint8)
    source[:5]=255
    source[20:35,40:45]=255
    source[18:20,40:45]=230
    mask=np.ones(source.shape[:2],bool)
    line={"bbox":[35,20,20,20],"style":{"color":"FFFFFF"}}
    inspection=residue_inspection_mask(source,line,mask)
    assert inspection[18:20,40:45].all()
    assert not inspection[:5].any()
