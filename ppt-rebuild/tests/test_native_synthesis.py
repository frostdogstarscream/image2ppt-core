from pathlib import Path
import sys
import pytest
from PIL import Image, ImageDraw

repo = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(repo / "ppt-rebuild/src"), str(repo / "services/planner/src"), str(repo / "src")]
from native_plan import ledger
from native_canvas import build_canvas, measured_icon
from slideir_adapter import sha256


def test_background_is_independent_and_photo_owns_only_embedded_lines(tmp_path):
    source = tmp_path / "source.png"
    im = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(im).rectangle((10, 10, 59, 39), fill="green")
    im.save(source)
    evidence = [
        {"id": "l1", "bbox": [15, 18, 15, 8], "score": 0.8, "text": "sign"},
        {"id": "l2", "bbox": [15, 45, 20, 9], "score": 1, "text": "title"},
    ]
    shape = {
        "kind": "shape",
        "label": "card",
        "module": 0,
        "geometry": "rect",
        "box": [30, 30, 690, 850],
        "fill": "F4FAFF",
    }
    photo = {"kind": "photo", "label": "photo", "module": 0, "box": [100, 125, 600, 500]}
    modules, objects, lines = ledger(
        {"modules": [{"box": [0, 0, 700, 875], "label": "card"}], "objects": [shape, photo]},
        im.size,
        evidence,
    )
    modules = [
        dict(
            modules[1],
            objects=objects,
            inside=[{"line": "l1", "asset": objects[1]["id"]}],
            paragraphs=[{"lines": ["l2"], "role": "title"}],
        )
    ]
    ir, report = build_canvas(
        tmp_path,
        source,
        {"background": {"color": "FFFFFF"}, "modules": modules, "readings": lines},
        pack=None,
        layerd_url=None,
        token_file=None,
        inpaint_url=None,
        max_iterations=1,
        timeout=1,
    )
    assert ir["background"] == {"color": "FFFFFF"}
    assert ir["groups"][0]["member_ids"] == [n["id"] for n in ir["nodes"]]
    assert [n["text"] for n in ir["nodes"] if n["type"] == "text"] == ["title"]
    assert ir["assets"][0]["contains_evidence_ids"] == ["ev-" + objects[1]["id"], "l1"]
    assert report["embedded_text"][0]["text"] == "sign"
    assert all(a["sha256"] != sha256(source) for a in ir["assets"])


def test_rejects_page_screenshot_and_preserves_shape_eligibility():
    obs = {
        "modules": [],
        "objects": [{"kind": "photo", "module": -1, "label": "slide", "box": [0, 0, 1000, 1000]}],
    }
    with pytest.raises(ValueError, match="page-sized"):
        ledger(obs, (300, 200), [])
    obs["objects"][0]["kind"] = "shape"
    assert len(ledger(obs, (300, 200), [])[1]) == 1


def test_shifted_icon_proposal_keeps_lower_outline():
    im = Image.new("RGB", (160, 120), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((50, 35, 88, 73), fill="blue")
    d.rectangle((53, 88, 90, 97), fill="black")
    x, y, w, h = measured_icon(im, [48, 30, 44, 29], [{"bbox": [52, 86, 40, 13]}])
    assert x <= 50 and y <= 35 and x + w >= 89 and y + h >= 74 and y + h < 86


def test_distinct_icon_anchors_split_one_planned_paragraph():
    from native_canvas import separate_icon_rows
    rows={"a":{"bbox":[90,17,80,14]},"b":{"bbox":[90,52,76,14]}}
    icons=[dict(kind="icon",id=str(i),bbox=[37,y,28,25]) for i,y in enumerate([12,47])]
    groups=separate_icon_rows([dict(lines=["a","b"],role="body")],rows,icons)
    assert [g["lines"] for g in groups]==[["a"],["b"]]


def test_title_uses_free_width_within_carrier():
    from native_layout import fit_field
    row=dict(id="t",bbox=[27,8,28,18])
    field, size, wrap=fit_field("政策服务标题",[row],dict(role="title",bold=True),
                         dict(bbox=[0,0,200,90],objects=[dict(kind="shape",geometry="rect",bbox=[0,0,140,32])]),
                         {"t":row},(200,90))
    assert not wrap and field[2]>row["bbox"][2] and field[0]+field[2]<=140 and size>1


def test_return_stroke_uses_only_painted_legs():
    from native_canvas import routed_line
    im=Image.new("RGB",(160,100),"white")
    draw=ImageDraw.Draw(im)
    draw.line([(12,10),(148,10),(148,65)],fill="#6898e0",width=2)
    base=dict(id="return",type="shape",bbox=[12,10,136,55],evidence_ids=["e"])
    parts=routed_line(dict(label="return",stroke="6898E0",path=[[75,100],[925,100],[925,650]]),
                      im,[dict(bbox=[50,32,55,17])],base)
    assert len(parts)==2 and all(min(p["bbox"][2:])<3 for p in parts)
    assert [p["evidence_ids"] for p in parts]==[["e"],[]]


def test_traced_outer_arrow_remains_a_native_path():
    from native_canvas import routed_line
    im=Image.new("RGB",(320,85),"white")
    draw=ImageDraw.Draw(im)
    draw.line([(12,10),(283,10),(296,18),(296,44),(277,48)],fill="#7CB3FA",width=2)
    draw.polygon([(277,48),(285,44),(285,52)],fill="#7CB3FA")
    base=dict(id="return",type="shape",bbox=[9,8,290,48],evidence_ids=["e"])
    parts=routed_line(dict(label="return",arrow_end="left",stroke="7CB3FA"),
                      im,[dict(bbox=[80,22,60,15])],base)
    assert parts[0]["outline"] and len(parts[0]["outline"])>4
    assert parts[0]["geometry"]=="line" and parts[0]["evidence_ids"]==["e"]


def test_unpainted_space_inside_connector_allows_step_grouping():
    from native_layout import painted_overlap
    route = {"bbox": [10, 10, 100, 40], "outline": [
        [0,0], [100000,0], [100000,100000], [98000,100000], [98000,5000], [0,5000]]}
    assert not painted_overlap({"bbox": [30,20,25,10]}, route)
    assert painted_overlap({"bbox": [107,30,8,10]}, route)
