import copy
import io
import json
from pathlib import Path
import sys
import uuid
import zipfile
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from product.contracts import Page,Review
from product.materialize import digest,write,verify_package
from product import pipeline


def page(source):
    return {'schema':'VisualPage/1','source_sha256':digest(source),'image_size':[800,450],
            'background':'FFFFFF','styles':[{'id':'title','role':'并列标题','ea':'Arial','latin':'Arial',
                                            'size_px':24,'bold':True,'color':'123456'}],
            'texts':[{'id':'t1','text':'Hello','bbox':[20,20,200,40],'z':1,'style':'title'}],
            'shapes':[],'crops':[],'repeated':[],'limitations':[]}


@pytest.fixture
def source(tmp_path):
    path=tmp_path/'source.png';Image.new('RGB',(800,450),'white').save(path);return path


def test_number_structure_regression(source):
    p=page(source);p['texts']=[{'id':f't{i}','text':s,'bbox':[20,20+i*80,500,50],'z':1,'style':'title'}
                            for i,s in enumerate(['01','02算网支撑与模型服务','03系统协同与数据利用'])]
    with pytest.raises(ValueError,match='separate'):Page.model_validate(p)
    p['texts'][1]['text']='02';p['texts'][2]['text']='03'
    with pytest.raises(ValueError,match='explicit repeated'):Page.model_validate(p)
    p['repeated']=[{'role':'编号','members':['t0','t1','t2'],'alignment':'left'}]
    Page.model_validate(p)
    p['texts'].append({'id':'footer','text':'02','bbox':[730,400,60,40],'z':5,'style':'title'})
    Page.model_validate(p)
    p['texts'][2]['bbox'][0]=30
    with pytest.raises(ValueError,match='alignment'):Page.model_validate(p)


def test_group_weight_cannot_diverge_and_emphasis_preserved(source):
    p=page(source);p['styles'].append({**p['styles'][0],'id':'em','bold':False})
    p['texts'][0]['spans']=[{'text':'He','style':'em'},{'text':'llo','style':'title'}]
    q=Page.model_validate(p);assert q.texts[0].spans[0].style=='em'
    p['texts'].append({**p['texts'][0],'id':'t2','bbox':[20,90,200,40],'style':'em'})
    p['repeated']=[{'role':'并列标题','members':['t1','t2']}]
    with pytest.raises(ValueError,match='share a style'):Page.model_validate(p)


def test_crop_ghosts_rejected(source):
    p=page(source);p['crops']=[{'id':'c','bbox':[20,20,220,60],'z':0,'description':'photo'}]
    with pytest.raises(ValueError,match='overlaps'):Page.model_validate(p)


def test_review_cannot_skip_checks():
    with pytest.raises(ValueError):Review.model_validate({'schema':'VisualReview/1','source_sha256':'a',
        'plan_sha256':'b','preview_sha256':'c','passed':True,'findings':[],'checked':['font']})


def test_italic_exemption_does_not_hide_size_or_graphics():
    r=Review.model_validate({'schema':'VisualReview/1','source_sha256':'a','plan_sha256':'b','preview_sha256':'c',
        'passed':False,'checked':['wording','structure','font','emphasis','layout','graphics'],
        'findings':[{'category':'font','aspect':'italic_only','description':'only slant'},
                    {'category':'font','aspect':'other','description':'too small'},
                    {'category':'graphics','aspect':'italic_only','description':'wrong skewed shape'}]})
    assert [f['description'] for f in pipeline.effective_findings(r)]==['too small','wrong skewed shape']


def test_gradient_contract_rejects_disordered_stops(source):
    p=page(source);p['shapes']=[{'id':'bg','bbox':[1,1,700,60],'z':0,'geometry':'roundRect','fill':'FFFFFF',
        'fill_gradient':{'angle_deg':0,'stops':[{'offset':0,'color':'FFFFFF'},{'offset':100000,'color':'AAAAAA'}]}}]
    Page.model_validate(p)
    p['shapes'][0]['fill_gradient']['stops'].reverse()
    with pytest.raises(ValueError,match='gradient'):Page.model_validate(p)


class FakeProvider:
    model='fixture'
    def __init__(self,plan,failed_reviews=0,bad_binding=False):
        self.plan=plan;self.failed_reviews=failed_reviews;self.bad_binding=bad_binding;self.calls=[]
    def ask(self,system,context,images,contract):
        self.calls.append(contract)
        if contract is Page:return self.plan
        failed=self.failed_reviews>0;self.failed_reviews-=1
        return {'schema':'VisualReview/1',**{k:context[k] for k in ('source_sha256','plan_sha256','preview_sha256')},
                **({'preview_sha256':'wrong'} if self.bad_binding else {}),'passed':not failed,
                'checked':['wording','structure','font','emphasis','layout','graphics'],
                'findings':[{'category':'font','description':'font differs','node_ids':['t1']}] if failed else []}


def test_pipeline_retries_as_unit_and_never_publishes_failed(source,tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'materialize',lambda plan,source,directory,*args:source)
    def compile(irs,out,pages):
        out.mkdir(parents=True);preview=out/'preview';preview.mkdir()
        Image.new('RGB',(800,450),'white').save(preview/'page-001.png')
        (out/'deck.pptx').write_bytes(b'fake-compiler')
        return {'pptx':str(out/'deck.pptx'),'preview':str(preview)}
    def run(name,provider,human_review=False):
        job=tmp_path/name;(job/'inputs').mkdir(parents=True)
        (job/'inputs/01.png').write_bytes(source.read_bytes());write(job/'request.json',{'sources':['01.png'],'delivery_for_human_review':human_review})
        return job,pipeline.run(job,provider=provider,review_provider=provider,compiler=compile,manifest={'fonts':[]})
    first=FakeProvider(page(source),failed_reviews=1)
    job,result=run('repair',first)
    assert result['status']=='SUCCEEDED' and len(first.calls)==4
    assert (job/'round-01').exists() and (job/'round-02').exists()
    failed=FakeProvider(page(source),failed_reviews=99)
    job,result=run('failed',failed)
    assert result['status']=='QUALITY_FAILED' and len(failed.calls)==6
    assert not (job/'delivery').exists()
    draft=FakeProvider(page(source),failed_reviews=99)
    job,result=run('human-review',draft,human_review=True)
    assert result['status']=='REVIEW_REQUIRED' and result['human_acceptance'] is False
    assert result['unresolved_pages']==[1] and len(draft.calls)==2
    assert (job/'delivery/result.pptx').exists()
    stale_draft=FakeProvider(page(source),bad_binding=True)
    job,result=run('invalid-review-draft',stale_draft,human_review=True)
    assert result['status']=='QUALITY_FAILED' and not (job/'delivery').exists()
    class RepairingValidation(FakeProvider):
        def __init__(self,p):super().__init__(p);self.step=0
        def ask(self,system,context,images,contract):
            value=copy.deepcopy(super().ask(system,context,images,contract))
            if contract is Page:
                self.step+=1
                if self.step==1:value['texts'][0]['bbox'][0]=-1
                elif self.step==2:value['texts'][0]['style']='missing'
                elif self.step==3:value['texts'][0]['text']='Hello\n'
            return value
    repaired=RepairingValidation(page(source))
    _,result=run('distinct-validation-progress',repaired,human_review=True)
    assert result['status']=='REVIEW_REQUIRED' and len(repaired.calls)==5
    stale=FakeProvider(page(source),bad_binding=True)
    _,result=run('stale',stale);assert result['status']=='QUALITY_FAILED'
    class Improving(FakeProvider):
        def __init__(self,p):super().__init__(p,failed_reviews=3);self.remaining=3
        def ask(self,system,context,images,contract):
            value=super().ask(system,context,images,contract)
            if contract is Review:
                value['findings']=[{'category':'font','description':f'issue {i}','node_ids':['t1']}
                                   for i in range(self.remaining)]
                value['passed']=not value['findings'];self.remaining-=1
            return value
    growing=Improving(page(source))
    _,result=run('more-than-six',growing)
    assert result['status']=='SUCCEEDED' and len(growing.calls)==8


def test_api_auth_upload_traversal_and_key_privacy(monkeypatch):
    from fastapi.testclient import TestClient
    from product.app import create_app,ROOT
    monkeypatch.setattr(pipeline,'settings',lambda:object())
    monkeypatch.setenv('OPENCODE_GO_API_KEY','test-review-key')
    root=ROOT/'outputs'/('product-test-'+uuid.uuid4().hex)
    app=create_app(root,'x'*32,run_worker=False)
    with TestClient(app) as client:
        assert client.get('/api/health').status_code==401
        headers={'Authorization':'Bearer '+'x'*32}
        blob=io.BytesIO();Image.new('RGB',(800,450),'white').save(blob,format='PNG')
        r=client.post('/api/jobs',headers=headers,files=[('files',('../../evil.png',blob.getvalue(),'image/png'))])
        assert r.status_code==202
        id=r.json()['id'];assert (root/id/'inputs/01.png').exists()
        assert client.get(f'/api/jobs/{id}/files/result.pptx',headers=headers).status_code==409
        assert client.get('/api/jobs/invalid',headers=headers).status_code==404
        assert client.post('/api/jobs',headers=headers,files=[('files',('x.png',b'not-image'))]).status_code==422
        monkeypatch.setenv('MOMA_API_KEY','before');monkeypatch.setenv('PLANNER_API_KEY','before')
        r=client.post('/api/provider-key',headers=headers,json={'key':'test-secret-123456'})
        assert r.status_code==200 and 'test-secret' not in r.text
        assert not any('test-secret' in p.read_text(errors='ignore') for p in root.rglob('*.json'))


def test_restart_does_not_repeat_paid_requests(tmp_path):
    from product.app import Queue
    q=Queue(tmp_path/'queue')
    with q.connect() as db:db.execute('INSERT INTO jobs VALUES(?,?,?)',('a'*32,'RUNNING',0))
    q.start()
    try:assert q.status('a'*32)['status']=='INTERRUPTED'
    finally:q.stop()


def test_package_rejects_missing_text(source,tmp_path):
    pptx=tmp_path/'bad.pptx'
    with zipfile.ZipFile(pptx,'w') as archive:
        archive.writestr('ppt/slides/slide1.xml','<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>')
    with pytest.raises(ValueError,match='text differs'):verify_package(pptx,[Page.model_validate(page(source))])


def test_crop_expansion_preserves_connected_icon_and_protects_text(source):
    from PIL import ImageDraw
    from product.crop_fit import complete_crops
    image=Image.open(source);ImageDraw.Draw(image).rectangle((500,100,550,150),fill='black');image.save(source)
    p=page(source);p['crops']=[{'id':'icon','bbox':[500,100,30,50],'z':0,'description':'icon'}]
    fixed,changes=complete_crops(Page.model_validate(p),source)
    assert changes and fixed.crops[0].bbox[0]+fixed.crops[0].bbox[2]>=552
    p['texts'].append({'id':'protected','text':'X','bbox':[540,100,120,50],'style':'title','z':2})
    fixed,changes=complete_crops(Page.model_validate(p),source)
    assert not changes and fixed.crops[0].bbox==[500,100,30,50]


def test_ink_measurement_excludes_thin_rules(source,tmp_path):
    from PIL import ImageDraw,ImageFont
    from product.visual_checks import compare_text
    image=Image.open(source);draw=ImageDraw.Draw(image)
    draw.text((20,20),'Hello',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',24),fill='#123456')
    image.save(source);preview=tmp_path/'preview.png'
    ImageDraw.Draw(image).line((120,38,215,38),fill='#123456',width=2);image.save(preview)
    metrics,_=compare_text(source,preview,Page.model_validate(page(source)),tmp_path/'detail.png')
    assert metrics[0]['width_ratio']==1 and metrics[0]['height_ratio']==1


def test_moma_plans_and_go_reviews_with_images(source,tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'materialize',lambda plan,source,directory,*args:source)
    class Planner:
        model='moma-fixture'
        def ask(self,system,context,images,contract):
            assert contract is Page and images
            return page(source)
    class Reviewer(FakeProvider):
        model='go-fixture'
        def ask(self,system,context,images,contract):
            assert contract is Review and len(images)>=2
            return super().ask(system,context,images,contract)
    def compile(irs,out,pages):
        out.mkdir(parents=True);previews=out/'preview';previews.mkdir()
        (previews/'page-001.png').write_bytes(source.read_bytes());(out/'deck.pptx').write_bytes(b'fixture')
        return {'pptx':str(out/'deck.pptx'),'preview':str(previews)}
    job=tmp_path/'roles';(job/'inputs').mkdir(parents=True)
    (job/'inputs/01.png').write_bytes(source.read_bytes());write(job/'request.json',{'sources':['01.png']})
    result=pipeline.run(job,provider=Planner(),review_provider=Reviewer(page(source)),compiler=compile,manifest={'fonts':[]})
    assert result['status']=='SUCCEEDED' and result['review_model']=='go-fixture'
    calls=json.loads((job/'calls.json').read_text())
    assert [(c['kind'],c['model']) for c in calls]==[('planning','moma-fixture'),('reviewing','go-fixture')]
    class FailedReview(Reviewer):
        def ask(self,*args):raise pipeline.PipelineError('opencode_http_503')
    failed=tmp_path/'failed-review';(failed/'inputs').mkdir(parents=True)
    (failed/'inputs/01.png').write_bytes(source.read_bytes());write(failed/'request.json',{'sources':['01.png']})
    with pytest.raises(pipeline.PipelineError,match='opencode_http_503'):
        pipeline.run(failed,provider=Planner(),review_provider=FailedReview(page(source)),compiler=compile,manifest={'fonts':[]})
    calls=json.loads((failed/'calls.json').read_text())
    assert calls[-1]['model']=='go-fixture' and calls[-1]['state']=='FAILED' and len(calls)==2
