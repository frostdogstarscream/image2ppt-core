"""Paragraph and mixed-run styles survive the deterministic OOXML adapter."""
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pptx_xml import adapt_slide, NS


def test_multiline_styles_and_mixed_run_fonts_are_preserved():
    root = ET.fromstring(f'''<p:sld xmlns:p="{NS['p']}" xmlns:a="{NS['a']}"><p:cSld><p:spTree>
    <p:sp><p:nvSpPr><p:cNvPr id="1" name="paragraph"/></p:nvSpPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>placeholder</a:t></a:r></a:p></p:txBody></p:sp>
    </p:spTree></p:cSld></p:sld>''')
    first = {"text": "项目27项", "font": {"latin": "Arial", "ea": "微软雅黑", "size_pt": 18},
             "color": "112233", "bold": False, "line_spacing_pt": 24, "space_after_pt": 2,
             "runs": [{"text": "项目"}, {"text": "27项", "bold": True, "color": "0060B1",
                       "font": {"latin": "Arial", "ea": "黑体", "size_pt": 24}}]}
    second = {"text": "第二行", "font": {"latin": "Arial", "ea": "宋体", "size_pt": 15},
              "color": "445566", "bold": True, "left_indent_px": 4, "space_before_pt": 1}
    ir = {"canvas": {"source_px": [800, 450], "target_px": [800, 450]}, "groups": [],
          "nodes": [{"id": "paragraph", "type": "text", "font": first["font"], "paragraphs": [first, second]}]}
    result = adapt_slide(root, ir)
    paragraphs = result.findall(".//p:txBody/a:p", NS)
    assert len(paragraphs) == 2
    runs = paragraphs[0].findall("a:r/a:rPr", NS)
    assert [r.get("sz") for r in runs] == ["1800", "2400"]
    assert runs[1].find("a:ea", NS).get("typeface") == "黑体"
    assert runs[1].find("a:solidFill/a:srgbClr", NS).get("val") == "0060B1"
    assert paragraphs[1].find("a:r/a:rPr/a:ea", NS).get("typeface") == "宋体"
    assert paragraphs[1].find("a:pPr", NS).get("marL") == str(4 * 9525)
    assert paragraphs[0].find("a:pPr/a:lnSpc/a:spcPts", NS).get("val") == "2400"
