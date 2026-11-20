from PIL import Image, ImageDraw
from product.runtime.plate_audit import reconcile
from product.experiment import diagram_marks


def test_position_and_content_distinguish_retained_residual_and_new():
    def line(i, x, s):
        return {'id':i,'bbox':[x,2,30,10],'text':s}
    a, b, c, d = (line('a',0,'same'), line('b',60,'same'), line('c',120,'old'), line('d',180,'extra'))
    native = [b, {**c,'text':'fixed'}]
    result = reconcile([a,b,c], [a,b,d], native, [{'id':'mark','label':'text','ocr_ids':['a']}], {'c':'fixed'})
    assert [r['status'] for r in result] == ['retained','residual','native','plate_only']
    assert result[0]['mark_ids'] == ['mark']


def test_connector_requires_visible_link_and_excludes_neighbor_text(tmp_path):
    src = tmp_path/'source.png'
    img = Image.new('RGB',(360,180),'white')
    draw = ImageDraw.Draw(img)
    anchors = []
    for i,x in enumerate((50,146)):
        draw.ellipse((x,72,x+22,94),fill='navy')
        anchors.append({'id':str(i),'label':'icon','bbox':[x,72,22,22]})
    img.save(src)
    assert not diagram_marks(src, anchors, [])
    draw.line((72,83,146,83),fill='navy',width=3)
    img.save(src)
    assert len(diagram_marks(src, anchors, [])) == 1
    assert not diagram_marks(src, anchors,[{'bbox':[90,75,25,12],'ocr_line_ids':['L1']}])
