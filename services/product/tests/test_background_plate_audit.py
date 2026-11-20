from PIL import Image, ImageDraw
from product.runtime.plate_audit import reconcile
from product.experiment import diagram_marks


def test_diagram_recovers_gradient_and_keeps_separate_faint_marks():
    import numpy as np
    from product.recovery.diagram_stage import _diagram_backing

    y, x = np.indices((90, 120))
    background = np.repeat((150 + x // 4 + y // 5)[..., None], 3, axis=2).astype('uint8')
    distance = np.hypot(x - 60, y - 45)
    core = distance <= 13
    shadow = (distance > 13) & (distance <= 16)
    tile = background.copy()
    tile[core] = 45
    tile[shadow] -= 7
    tile[45, 79] -= 7
    blocked = np.zeros_like(core)
    blocked[45, 75] = True
    fitted, owned = _diagram_backing(tile, tile, np.zeros_like(core), core, blocked, 2,
                                     np.full_like(tile, 180))
    assert owned[45, 76] and not owned[45, 75] and not owned[45, 79]
    assert np.abs(fitted[owned].astype(float) - background[owned]).max() <= 2


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


def test_connected_diagram_transfers_only_silhouette(tmp_path):
    import numpy as np
    from product.parallel_fusion import fuse
    from product.parallel_rebuild import build
    from product.recovery.diagram_stage import stage_diagram
    from test_parallel import branches

    image = Image.new('RGB',(240,120),'white')
    draw = ImageDraw.Draw(image)
    # The pale rim extends beyond the detector's tight rectangle on the left.
    draw.ellipse((20,20,53,53),fill=(248,248,248))
    draw.ellipse((24,24,49,49),fill='navy')
    draw.ellipse((139,54,164,79),fill='navy')
    draw.line((36,36,151,66),fill='navy',width=4)
    draw.rectangle((48,61,64,69),fill='red')
    path=tmp_path/'source.png'
    image.save(path)
    snap,evidence=branches(path)
    graph=fuse(snap,evidence,{'lines':[]})
    baseline=tmp_path/'baseline'
    build(graph,snap,{'fonts':[]},baseline,strategy='off',text_candidates=False)
    route={'id':'group','bbox':[22,22,145,61],'anchor_boxes':[[24,24,25,25],[139,54,25,25]],
           'diagram_members':['one','two']}
    ir,oid=stage_diagram(baseline,tmp_path/'new',route)
    plate=np.asarray(Image.open(ir.parent/'source-plate.png').convert('RGB'))
    assert (plate[36,36] == 255).all()
    assert plate[64,53].tolist() == [255,0,0]
    asset=np.asarray(Image.open(ir.parent/(oid+'.png')))
    import json
    box = json.loads((ir.parent/'scene.json').read_text('utf8'))['selected'][oid]['bbox']
    assert box[0] < route['bbox'][0]
    assert asset[64-box[1],53-box[0],3]==0
