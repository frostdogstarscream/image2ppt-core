from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.materialize import digest, write
from product.office_sites import checked_sites, clear_position, edit_sites, record_site, site_geometry, visible_box
from product.parallel_review import panels


def test_original_site_plan_uses_resolved_relationships_and_preserves_scope(tmp_path):
    ir = tmp_path/'slideir.json'
    write(ir, {'source_sha256':'source', 'nodes':[{'id':name, 'type':kind} for name,kind in (
        ('unrelated','image'),('tile','shape'),('logo','image'),('unsafe','image'),('source-plate','image'))]})
    scene = {'source_sha256':'source', 'selected':{'tile':{'kind':'surface'},
             'unsafe':{'move_safety':'MOVE_UNSAFE'}}, 'relations':[]}
    write(tmp_path/'scene.json',scene)
    assert edit_sites(ir)==(['logo','unrelated'],[])
    edge={'source':'tile','target':'logo','kind':'contains','status':'AMBIGUOUS','evidence':['source']}
    scene['relations']=[edge]
    write(tmp_path/'scene.json',scene)
    assert edit_sites(ir,{'tile'})==([],[])
    assert edit_sites(ir,{'tile'},include_unverified=True)[2][0]['reason']=='UNPROVEN_RELATION'
    edge['status']='RESOLVED'
    write(tmp_path/'scene.json',scene)
    assert edit_sites(ir,{'tile'})==([],[('tile','logo')])
    assert edit_sites(ir,{'unrelated'})==(['unrelated'],[])
    scene['relations'].append({**edge,'target':'unrelated'})
    write(tmp_path/'scene.json',scene)
    assert edit_sites(ir,{'tile'})==([],[])
    assert edit_sites(ir,{'tile'},include_unverified=True)[2][0]['reason']=='MULTIPLE_CHILDREN'


@pytest.mark.parametrize('geometry, window', [
    ((100,100,90,80,0),(80,80,210,200)), ((270,180,120,100,33),(220,130,420,310)),
    ((0,0,400,300,0),(0,0,400,300))])
def test_picture_clears_full_old_site_even_when_pasteboard_is_needed(geometry, window):
    x,y = clear_position(geometry,window,400,300,6)
    bx,by,bw,bh = visible_box((x,y,*geometry[2:4]),geometry[4])
    assert bx>=window[2]+5 or bx+bw<=window[0]-5 or by>=window[3]+5 or by+bh<=window[1]-5


def site_fixture(folder, office_dir):
    ir=folder/'slideir.json'
    write(ir,{'source_sha256':'s','canvas':{'source_px':[100,60],'target_px':[100,60]},
              'nodes':[{'id':'logo','type':'image','bbox':[18,20,16,14]}]})
    write(folder/'scene.json',{'source_sha256':'s','selected':{'logo':{'restore_bbox':[16,18,20,18]}},'relations':[]})
    office_dir.mkdir(parents=True,exist_ok=True)
    after=np.full((60,100,3),(0,70,90),dtype=np.uint8)
    before=after.copy();before[20:34,18:34]=255
    after[20:33,72:86]=255
    preview=office_dir/'page-01-before.png';full=office_dir/'test-move.png'
    Image.fromarray(before).save(preview);Image.fromarray(after).save(full)
    (frame,window,_)=site_geometry(ir)[0]['logo']
    row=record_site(office_dir,1,['logo'],'move-clear',window,after,full,
                    {'logo':frame},{'logo':[54,15,10.8,9.45,0]})
    return ir,{'original_sites':[row],'original_site_previews':{'1':digest(preview)}}


def test_absent_incomplete_or_changed_original_site_evidence_stays_unverified(tmp_path):
    ir,result=site_fixture(tmp_path,tmp_path)
    with pytest.raises(ValueError,match='did not supply'):
        checked_sites(tmp_path,{},ir,1)
    row=result['original_sites'][0]
    assert checked_sites(tmp_path,result,ir,1)==[row]
    (tmp_path/row['crop']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatched'):
        checked_sites(tmp_path,result,ir,1)


@pytest.mark.parametrize('defect',['wrong_window','wrong_crop_pixels','unchanged_frame','wrong_source_frame',
                                   'outside_report','outside_render','absent_destination','missing_inventory'])
def test_reported_pass_with_valid_file_hashes_rejects_invalid_site(tmp_path,defect):
    ir,result=site_fixture(tmp_path,tmp_path)
    row=result['original_sites'][0]
    if defect=='wrong_window':
        row['window'][0]+=1
    elif defect=='wrong_crop_pixels':
        crop=tmp_path/row['crop']
        with Image.open(crop) as source:size=source.size
        Image.new('RGB',size,'red').save(crop)
        row['sha256']=digest(crop)
    elif defect=='unchanged_frame':
        row['reopened_frames_pt']=row['original_frames_pt']
    elif defect=='wrong_source_frame':
        row['original_frames_pt']['logo'][0]+=3
    elif defect=='outside_report':
        row['outside_pixels']=1
    elif defect in ('outside_render','absent_destination'):
        with Image.open(tmp_path/(row['render_sha256']+'-reopen.png')) as full:edited=full.copy()
        if defect=='outside_render':
            edited.putpixel((99,59),(200,100,120))
        else:
            edited.paste((0,70,90),(72,20,86,33))
        file=tmp_path/'changed.png';edited.save(file)
        row['render_sha256']=digest(file)
        file.rename(tmp_path/(row['render_sha256']+'-reopen.png'))
    else:
        del row['reopened_frames_pt']
    with pytest.raises(ValueError):
        checked_sites(tmp_path,result,ir,1)


def test_final_review_panel_displays_actual_disjoint_move_crop(tmp_path):
    ir,result=site_fixture(tmp_path,tmp_path)
    source=tmp_path/'page-01-before.png'
    Image.new('RGB',(100,60),'white').save(tmp_path/'01-logo-delete.png')
    files,refs=panels(source,source,ir,tmp_path,1,tmp_path/'panels',site_review=result)
    assert result['original_sites'][0] in refs
    with Image.open(files[-1]) as panel:
        assert panel.getpixel((812,panel.height-8))==(0,70,90)


def test_legacy_worker_pass_with_missing_site_stays_unknown_on_direct_route(tmp_path, monkeypatch):
    from product.runtime import office_bridge
    from product.runtime.service import RuntimeService
    job=tmp_path/('a'*32)
    job.mkdir()
    write(job/'request.json',{'orchestrator':'editable_reconstruction','visual_review_backend':'moma'})
    deck=job/'input.pptx'
    deck.write_bytes(b'synthetic')
    ir=job/'slideir.json'
    write(ir,{'source_sha256':'s','nodes':[{'type':'image','id':'logo'}]})
    write(job/'scene.json',{'source_sha256':'s','relations':[],'selected':{}})
    def old_worker(job, pptx, irs, directory, **options):
        from product.materialize import digest
        assert options=={'original_sites':True}
        return {'status':'PASS','pptx_sha256':digest(pptx),'office':{'pptx_sha256':digest(pptx)},
                'office_directory':str(directory),'files':{}}
    monkeypatch.setenv('IMAGE2PPT_OFFICE_MODE','remote')
    monkeypatch.setattr(office_bridge,'submit',old_worker)
    service=RuntimeService(tmp_path)
    result=service.office_check(service.state(job.name),deck,[ir],job/'check')
    assert result['status']=='UNKNOWN'
    assert result['failure']=='OFFICE_SITE_EVIDENCE_UNAVAILABLE'
    assert 'did not supply' in result['reason']


def test_worker_roundtrip_delivers_hashed_original_site_crops(tmp_path,monkeypatch):
    from product.runtime import office
    from product.runtime.office_bridge import execute_ticket, pack, unpack
    from product.materialize import digest
    packet=tmp_path/'request'
    packet.mkdir()
    write(packet/'request.json',{'irs':['page/slideir.json'],'original_sites':True})
    site_fixture(packet/'page',tmp_path/'fixture-output')
    (packet/'input.pptx').write_bytes(b'synthetic')
    archive=tmp_path/'request.zip'
    pack(packet,archive)
    def check(pptx,irs,output,**kwargs):
        assert kwargs=={'original_sites':True}
        edited=output/'office'
        edited.mkdir(parents=True)
        _,observations=site_fixture(irs[0].parent,edited)
        (edited/'scratch.pptx').write_bytes(b'probe copy')
        return {'status':'PASS','pptx_sha256':digest(pptx),'office':observations}
    monkeypatch.setattr(office,'check_local',check)
    class Client:
        def post(self,path,**kwargs):
            self.content=archive.read_bytes() if path.endswith('/input') else kwargs['content']
            return self
        def raise_for_status(self):
            pass
    client=Client()
    execute_ticket(client,{},tmp_path/'worker','test-binding')
    host=tmp_path/'host'
    unpack(client.content,host)
    import json
    result=json.loads((host/'result.json').read_text('utf8'))
    assert result['worker_binding']=='test-binding'
    assert checked_sites(host/'check/office',result['office'],packet/'page/slideir.json',1)
    assert not (host/'check/office/scratch.pptx').exists()
