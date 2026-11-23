"""Ordinary wording stays editable from source line to the PowerPoint audit."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from product.materialize import digest
from product.recovery.region_artwork import verify_text_cutout
from product.runtime.batch_backbone import split_decorative_ocr_line, _text_stage_failure, _classify_failed_region
from product.runtime.plate_audit import bind_native_paragraphs, bind_separate_symbols, native_line_fields, reconcile
from product.runtime.host_baseline import delivery_defects


@pytest.mark.parametrize('ending', list('、，。；：！？）％…'))
def test_body_punctuation_keeps_every_source_character(ending):
    source = np.full((24, 120, 3), 246, 'uint8')
    for x in (12,24,36,48,63,84,106):
        source[5:17,x:x+4] = 20
    row = {'text':'属地支持与实施'+ending, 'bbox':[10,3,102,17]}
    assert split_decorative_ocr_line(source,row) == (row['text'],row['bbox'],'')


def test_only_isolated_edge_symbol_is_released():
    image = np.full((24, 112, 3), 248, 'uint8')
    for x in (6,31,42,54,66,78):
        image[7:15,x:x+5] = 30
    row = {'text':'● ABCDE', 'bbox':[5,4,81,15]}
    text, box, mark = split_decorative_ocr_line(image,row)
    assert (text,mark) == ('ABCDE','●') and box[0] == 31 and box[0]+box[2]==86
    # Move the real badge right up against the first glyph.
    image[7:15,6:11] = 248
    image[7:15,24:29] = 30
    assert split_decorative_ocr_line(image,row) == (row['text'],row['bbox'],'')


def test_numbered_fields_and_retry_share_the_original_failed_geometry():
    t = SimpleNamespace(id='field', text='合同交付、', bbox=[11,20,100,20], style='body')
    graph = SimpleNamespace(text=SimpleNamespace(texts=[t]), selected={},
                            asset=SimpleNamespace(objects=[SimpleNamespace(id='possible-peer',kind='decoration',bbox=[92,20,12,17])]))
    for members in ([], ['field','sibling']):
        row = _text_stage_failure(t, ValueError('native font frame allowance conflicts with canvas or protected peers'), members)
        result = _classify_failed_region(row,graph)
        assert result['max_repair_attempts'] and result['conflicting_peer_ids']==['possible-peer']
        assert result['failure']=='FONT_LAYOUT_FAILED' and result['group_members']==members


def test_measured_ui_chevron_survives_beside_a_label_but_math_operator_stays_native():
    image = np.full((30, 100, 3), 248, 'uint8')
    for x in (7,20,33):
        image[7:21,x:x+10] = 30
    for dy in range(13):
        x = 58 + min(dy,12-dy)
        image[7+dy,x:x+2] = 30
    row = {'text':'更多信息>', 'bbox':[5,4,64,22]}
    assert split_decorative_ocr_line(image,row)[2] == '>'
    assert split_decorative_ocr_line(image,{**row,'text':'<tag>'})[2] == ''
    assert split_decorative_ocr_line(image,{**row,'text':'预算 26>'})[2] == ''


def test_label_needs_the_delivered_symbol_and_native_wording_to_count():
    row = {'id':'line', 'text':'查看更多>', 'bbox':[10,10,88,22]}
    label = {'name':'label','text':'查看更多','bbox':[10,10,69,22]}
    icon = {'name':'arrow','bbox':[85,14,10,13]}
    record = {**row, 'render_text':label['text'], 'render_bbox':label['bbox'], 'separate_symbol':'>'}
    def audit(graphics):
        fields = bind_separate_symbols([label], {'label':['line']}, {'line':record}, graphics)
        return reconcile([row],[{'text':'>','bbox':icon['bbox']}],fields,[],{})
    assert audit([])[0]['status'] == 'unresolved'
    assert audit([dict(icon,bbox=[12,12,9,9])])[0]['status'] == 'unresolved'
    results = audit([icon])
    assert len(results) == 1 and results[0]['native_form'] == 'label_with_separate_symbol'
    assert results[0]['status'] == 'native' and results[0]['symbol_object'] == 'arrow'


def test_paragraph_match_needs_source_membership_exact_wording_and_order():
    lines = [{'id':'one','bbox':[10,12,90,12],'text':'核对需求、'},
             {'id':'two','bbox':[10,30,90,12],'text':'交付 26 项。'}]
    field = {'name':'body','bbox':[10,10,104,34],'text':'核对需求、\n交付 26 项。'}
    fields = bind_native_paragraphs(lines,[field],{'body':['one','two']},{})
    assert [r['native_form'] for r in reconcile(lines,[],fields,[],{})] == ['paragraph_member']*2
    for bad in (field | {'text':field['text'].replace('26','28')}, field | {'text':field['text'].replace('、','')}):
        assert not native_line_fields(lines[0],bind_native_paragraphs(lines,[bad],{'body':['one','two']},{}))
    assert not native_line_fields(lines[0],bind_native_paragraphs(lines,[field],{'body':['two','one']},{}))


def test_partial_image_fragment_and_wrong_native_punctuation_cannot_pass():
    line = {'id':'row','bbox':[10,10,200,22],'text':'由团队负责沟通、实施交付、'}
    image_word = {'text':'实施交付、','bbox':[134,11,71,19]}
    field = {'name':'native','text':line['text'][:-1],'bbox':line['bbox']}
    for fields,detected,status in (([field],[image_word],'retained'),([field],[],'unresolved'),
                                  ([dict(field,text=line['text'])],[image_word],'residual')):
        row = reconcile([line],detected,fields,[],{})[0]
        assert row['status'] == status
        assert delivery_defects({'verified':True,'pages':[{'page':1,'selected_editables':['native'],
                                'plate_audit':{'status':'MEASURED','rows':[row]}}]})


def test_cutout_can_retire_only_when_all_visible_ink_belongs_to_the_native_field(tmp_path):
    box = [2,3,12,7]
    paint = np.full((7,12,3),240,'uint8')
    rgba = np.zeros((7,12,4),'uint8')
    rgba[2:6,3:9] = (24,24,24,255)
    src = np.full((16,22,3),240,'uint8')
    src[5:9,5:11] = 24
    Image.fromarray(src).save(tmp_path/'source.png')
    asset = tmp_path/'text.png'
    Image.fromarray(rgba).save(asset)
    mask = np.zeros((16,22),bool)
    mask[5:9,5:11] = True
    proof = tmp_path/'owner.npz'
    np.savez(proof,mask=mask,restore_bbox=box,backing_patch=paint)
    ref = {'bbox':box,'path':asset.name,'sha256':digest(asset)}
    record = {'asset_reference':ref,'owner_evidence':{'file':proof.name,'sha256':digest(proof)},
              'payload':{'upgraded_from_region_crop':True},'move_safety':'MOVABLE','seed_ids':['seed']}
    receipt = {'asset_sha256':digest(asset),'seed':'seed','reading':'交付','ocr_identity':'fixture'}
    verify_text_cutout(tmp_path,record,receipt,'实施交付、',mask,box,paint)
    # An independent dot beside the word must survive, even with an exact OCR reading.
    rgba[1,1] = (30,30,30,255)
    Image.fromarray(rgba).save(asset)
    receipt['asset_sha256'] = ref['sha256'] = digest(asset)
    with pytest.raises(ValueError,match='independent artwork'):
        verify_text_cutout(tmp_path,record,receipt,'实施交付、',mask,box,paint)
