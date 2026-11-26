"""Blind inference uses synthetic providers here. No evaluation text reaches their prompts."""
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from product.materialize import digest
from product.runtime import page_reading as reading


class Budget:
    def __init__(self):
        self.seen = []

    def reserve(self, kind, request):
        assert kind == 'model_calls'
        assert request not in self.seen
        self.seen.append(request)
        return True


def test_blind_correction_is_source_bound_reusable_and_includes_unselected_lines(tmp_path,monkeypatch):
    src = tmp_path/'slide.png'
    Image.new('RGB',(320,240),'white').save(src)
    model_responses = {
        'page':['会议材料整理','专业研发 50 名','西安甲公司','文字恢复','政策咨询','原有正确'],
        'crop':['会议材料整理','专业研发 50 名','西安甲公司','文字恢复','政策咨讯','原有正确'],
    }
    sessions = []
    class Actor:
        def __init__(self,model):
            self.model = model
            self.session = str(len(sessions)+1)
            sessions.append(self.session)
            self.last_usage = {'total_tokens':11}
        def ask(self,prompt,context,images,contract):
            view = 'page' if prompt==reading.PAGE_PROMPT else 'crop'
            assert set(context) == {'field_numbers'}
            assert not any(x in prompt for x in model_responses['page'])
            assert all(p.is_file() for p in images)
            assert (len(images)==2 and images[0]==src) if view=='page' else len(images)==1
            return {'fields':[{'number':n,'text':model_responses[view][n-1],'legible':n!=1,
                               'kind':('formal_name' if n==3 else 'ordinary')} for n in context['field_numbers']]}
    monkeypatch.setattr(reading,'MomaReviewer',Actor)
    lines = {}
    objects = []
    members = {}
    for n,word in enumerate(['会议材科理','专业研发 15 名','西安乙公司','文宇恢复','政策咨词','原有正确'],1):
        lid,oid = f'l-{n}',f't-{n}'
        box = [12,n*30,100,22]
        lines[lid] = dict(id=lid,bbox=box,text=word,score=.99)
        objects.append(SimpleNamespace(id=oid,text=word,spans=[],bbox=box))
        members[oid] = [lid]
    scene = SimpleNamespace(text=SimpleNamespace(texts=objects,members=members),
                            source_sha256=digest(src),relations=[],asset=SimpleNamespace(objects=[]),
                            selected={f't-{n}':{'kind':'text'} for n in (1,2,3,5,6)})
    state = SimpleNamespace(job=tmp_path)
    missing = {'1':[{'object_id':'t-4','reason':'no safe source backing'}]}
    budget = Budget()
    for run in ('first','retry'):
        obs,changes,calls = reading.observe_page('synthetic',state,{'regional_failures':missing},tmp_path/run,
                                                 1,scene,lines,None,src,Image.open(src),budget)
        rows = {r['object_id']:r for r in obs}
        assert [c['after'] for c in changes]==['会议材料整理','政策咨询']
        assert rows['t-1']['status']=='model_confirmed' and rows['t-5']['status']=='pending_confirmation'
        assert rows['t-2']['after']=='专业研发 15 名' and rows['t-3']['after']=='西安乙公司'
        assert rows['t-4']['delivery']=='raster_retained' and rows['t-4']['reconstruction_reason']=='no safe source backing'
        assert rows['t-4']['after']=='文字恢复' and not rows['t-4']['written_text']
        assert rows['t-6']['status']=='ocr_retained'
        assert all(reading.authorized(change,digest(src)) for change in changes)
        assert all(Path(c['receipt_file']).is_file() for c in calls)
        tampered = dict(changes[0],after='伪造改文')
        assert not reading.authorized(tampered,digest(src))
    assert len(budget.seen)==len(sessions)==2
    scene.relations = [SimpleNamespace(source='t-5',kind='intrinsic_text')]
    obs,changes,_ = reading.observe_page('synthetic',state,{},tmp_path/'intrinsic',1,
                                          scene,lines,None,src,Image.open(src),budget)
    intrinsic = next(row for row in obs if row['object_id']=='t-5')
    assert intrinsic['delivery']=='intrinsic_image_text' and not intrinsic['written_text']
    assert all(row['object_id']!='t-5' for row in changes)


def test_protected_value_requires_a_labeled_disjoint_witness():
    view = lambda n,s:{'answer':{'number':n,'text':s,'legible':True,'kind':'quantity'},'status':'ok'}
    old,new='合同金额 31 万元','合同金额 37 万元'
    text = {'id':'l1','text':old,'bbox':[0,0,10,5],'score':.99}
    other = {'id':'l2','text':new,'bbox':[0,10,10,5],'score':.99}
    candidates = [(1,text),(2,other)]
    page,crop = view(1,new), view(1,new)
    witness = reading.witness_for(new,text,candidates,{2:view(2,new)})
    assert witness and witness['ocr_id']=='l2'
    assert reading.decide(old,page,crop,None)[:2]==(old,'pending_confirmation')
    assert reading.decide(old,page,crop,witness)[:2]==(new,'model_confirmed')
    assert not reading.witness_for('37万元',text,[(2,dict(other,text='37万元'))],{2:view(2,'37万元')})
    assert not reading.witness_for(new,text,[(2,dict(other,bbox=text['bbox']))],{2:view(2,new)})
    assert reading.decide('未更名',view(1,'新项目'),view(1,'新项目'),None,project_name=True)[0] == '未更名'


def test_reader_unavailable_keeps_original():
    page={'status':'ok','answer':{'text':'文字修复','legible':False,'kind':'ordinary'}}
    failed={'status':'unavailable','answer':None}
    assert reading.decide('文字修复原值',page,failed,None)[:2] == ('文字修复原值','pending_confirmation')


def test_429_records_cause_and_preserves_source_without_more_billing(tmp_path,monkeypatch):
    src = tmp_path/'source.png'
    Image.new('RGB',(50,30),'white').save(src)
    class Actor:
        def __init__(self,model):
            self.session = 'once'
        def ask(self,*args):
            raise reading.PipelineError('moma_http_429')
    monkeypatch.setattr(reading,'MomaReviewer',Actor)
    line = dict(id='l',bbox=[2,2,25,10],text='资料',score=.8)
    scene = SimpleNamespace(source_sha256=digest(src),text=SimpleNamespace(texts=[],members={}),
                            relations=[],asset=SimpleNamespace(objects=[]),selected={})
    rows,changes,calls = reading.observe_page('synthetic',SimpleNamespace(job=tmp_path),{},tmp_path/'run',1,
        scene,{'l':line},None,src,Image.open(src),Budget())
    assert not changes and len(calls)==1 and calls[0]['reason']=='moma_http_429'
    assert rows[0]['reason']=='page_reader_unavailable' and rows[0]['delivery']=='raster_retained'


def test_delivery_reports_only_the_native_text_actually_selected():
    report = {'observations':[{'object_id':'t','delivery':'staged_for_office','before':'错','proposed':'对'},
                              {'object_id':'r','delivery':'raster_retained','before':'另','proposed':'另'}]}
    graph = {'text':{'texts':[{'id':'t','text':'错'},{'id':'r','text':'另'}]},'selected':{'t':{'kind':'text'}}}
    reading.bind_delivered_text(report,graph,True)
    assert [(r['delivery'],r['written_text']) for r in report['observations']]==[
        ('baseline_retained','错'),('raster_retained',None)]
    graph['text']['texts'][0]['text'] = '对'
    reading.bind_delivered_text(report,graph,False)
    assert report['observations'][0]['delivery']=='applied_after_office'


def test_unconfigured_full_page_reader_delivers_control(tmp_path,monkeypatch):
    from product.runtime import host_baseline,runner
    from product.materialize import write
    write(tmp_path/'request.json',{'host_stage0':True,'page_reading':True})
    calls = []
    monkeypatch.delenv('MOMA_API_KEY',raising=False)
    monkeypatch.setattr(host_baseline,'ensure_baseline',lambda job:calls.append('baseline'))
    monkeypatch.setattr(host_baseline,'selected_result',lambda job:None)
    monkeypatch.setattr(host_baseline,'publish',lambda job,agent,reasons:calls.extend(reasons))
    monkeypatch.setattr(runner,'_agent_run',lambda *args: (_ for _ in ()).throw(AssertionError('unexpected Agent call')))
    runner.run(tmp_path)
    assert calls==['baseline','READING_MODEL_UNAVAILABLE']


def test_multi_page_draft_contains_all_office_checked_pages(tmp_path,monkeypatch):
    import json
    from product.runtime import host_baseline as host
    from product.materialize import write
    write(tmp_path/'request.json',{'page_reading':True})
    full = tmp_path/'full.pptx'
    full.write_bytes(b'synthetic-combined')
    pages = [{'ir':'first.json'},{'ir':'second.json'}]
    current = {'pages':pages,'candidates':{},'reading_reports':{}}
    state = SimpleNamespace(read=lambda:current,
        mutate=lambda name,args,expected,request,fn:fn(current,tmp_path/request))
    class Service:
        def graph(self,row):
            return SimpleNamespace(selected={})
        def review_numbered_text(self,state,current,directory,*,page,object_ids):
            assert object_ids is None
            cid = 'c-'+str(page)
            current['candidates'][cid] = {'reading_receipt':cid,'object_ids':[str(page)]}
            return {'candidate_id':cid}
        def validate_candidate(self,*args,**kwargs):
            return {'status':'PASS'}
        def commit_region(self,*args,**kwargs):
            return {'status':'COMMITTED'}
        def compose_delivery(self,*args):
            pass
        def composition(self,current):
            return {'artifacts':{'pptx':str(full)}}
        def office_check(self,state,pptx,irs,directory):
            assert pptx==str(full) and irs==[Path(row['ir']) for row in pages]
            return {'status':'PASS'}
    monkeypatch.setattr(host,'RuntimeState',lambda job:state)
    monkeypatch.setattr(host,'RuntimeService',lambda root:Service())
    host.prepare_reading_draft(tmp_path)
    receipt = json.loads((tmp_path/'reading-draft.json').read_text('utf8'))
    assert receipt['pptx']==str(full) and receipt['pages']==2 and receipt['sha256']==digest(full)



def test_only_text_in_the_measured_photo_caption_gets_priority():
    objects = {'picture':{'kind':'asset','payload':{'source_frame':[10,20,120,80],'caption_boundary':80}}}
    lines = {s:{'id':s,'score':.999,'bbox':b} for s,b in [
        ('caption',[20,81,70,15]),('in-photo',[20,30,70,15]),('beside',[135,81,45,12])]}
    assert reading.photo_caption_lines(objects,lines)=={'caption'}
