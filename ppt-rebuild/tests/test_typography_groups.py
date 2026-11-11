import copy
import pytest
from _helpers import SRC
from typography_groups import apply_group,choose_groups,validate_groups,verify_groups,immutable


def fixture():
    font={'ea':'A','latin':'Latin','size_pt':20}
    p=lambda text,bold: {'text':text,'font':font.copy(),'bold':bold,'color':'112233'}
    ir={'nodes':[{'id':'t','type':'text','text':'业务\n应用','bbox':[1,2,100,100],
        'font':font.copy(),'bold':False,'paragraphs':[p('业务',False),p('应用',True)]},
        {'id':'u','type':'text','text':'收入27项','bbox':[200,2,100,100],
         'font':font.copy(),'bold':True,'color':'112233','paragraphs':[
             {**p('收入27项',True),'runs':[{'text':'收入','font':font.copy(),'bold':True},
                    {'text':'27项','color':'0062AA','bold':True}]}]}]}
    style={'ea':'B','latin':'Latin','size_pt':20,'bold':False}
    groups=[{'id':'body','members':['t/p0','t/p1','u/p0'],'candidates':[style],
             'exceptions':{'u/p0/r1':{'bold':True,'size_pt':26}}}]
    manifest={'fonts':[{'family':f,'bold':b,'missing_characters':''}
                       for f in ['A','B','Latin'] for b in [False,True]]}
    return ir,groups,manifest


def test_multiline_and_stale_run_styles_share_one_style_preserving_emphasis():
    ir,g,m=fixture(); result,accepted,rejected=choose_groups(ir,g,m,lambda *a:True)
    assert not rejected
    assert [p['bold'] for p in result['nodes'][0]['paragraphs']]==[False,False]
    runs=result['nodes'][1]['paragraphs'][0]['runs']
    assert runs[0]['bold'] is False and runs[0]['font']['ea']=='B'
    assert runs[1]['bold'] is True and runs[1]['font']['size_pt']==26
    assert runs[1]['color']=='0062AA'
    assert immutable(ir)==immutable(result)
    verify_groups(result,g,accepted)


def test_group_rejection_is_atomic_and_next_candidate_can_succeed():
    ir,g,m=fixture(); g[0]['candidates'].append({**g[0]['candidates'][0],'size_pt':19})
    calls=[]
    def check(before,after,group):
        calls.append(copy.deepcopy(before))
        return after['nodes'][0]['font']['size_pt']==19
    result,accepted,rejected=choose_groups(ir,g,m,check)
    assert calls==[ir,ir] and not rejected
    assert result['nodes'][1]['font']['size_pt']==19


@pytest.mark.parametrize('answer',[False,None,{},1])
def test_no_true_gate_means_entire_group_retained(answer):
    ir,g,m=fixture(); result,accepted,rejected=choose_groups(ir,g,m,lambda *a:answer)
    assert result==ir and not accepted and rejected==['body']


def test_repeated_application_is_noop_and_does_not_render():
    ir,g,m=fixture(); result,_,_=choose_groups(ir,g,m,lambda *a:True)
    again,_,_=choose_groups(result,g,m,lambda *a:pytest.fail('no-op rendered'))
    assert result==again


@pytest.mark.parametrize('mutation',['missing','duplicate','unknown','nan','color','exception'])
def test_invalid_map_rejected(mutation):
    ir,g,m=fixture()
    if mutation=='missing':g[0]['members'].pop()
    elif mutation=='duplicate':g[0]['members'].append('t/p0')
    elif mutation=='unknown':g[0]['members'][0]='bad/p0'
    elif mutation=='nan':g[0]['candidates'][0]['size_pt']=float('nan')
    elif mutation=='color':g[0]['candidates'][0]['color']='FFFFFF'
    else:g[0]['exceptions']['t/p0/r8']={'bold':True}
    with pytest.raises(ValueError):validate_groups(ir,g)


def test_missing_real_bold_face_or_glyph_rejects_whole_group():
    ir,g,m=fixture();m['fonts']=[f for f in m['fonts'] if not (f['family']=='B' and f['bold'])]
    result,accepted,rejected=choose_groups(ir,g,m,lambda *a:True)
    assert result==ir and rejected==['body']


def test_final_group_verifier_catches_partial_override():
    ir,g,m=fixture();result,accepted,_=choose_groups(ir,g,m,lambda *a:True)
    result['nodes'][0]['paragraphs'][1]['bold']=True
    with pytest.raises(ValueError):verify_groups(result,g,accepted)


def test_office_family_name_uses_windows_face_not_short_mac_alias():
    from typography_plan import windows_family_name
    from fontTools.ttLib import newTable
    names=newTable('name');names.names=[]
    names.setName('Qihei Lenovo',1,1,0,0)
    names.setName('Qihei Lenovo 60S',1,3,1,0x409)
    assert windows_family_name(names)=='Qihei Lenovo 60S'


def test_shared_style_can_correct_inconsistent_old_bold_flags():
    from typography_pipeline import shared_styles
    a={'id':'a','bold':False,'font_size_px':20,'color':'112233'}
    b={**a,'id':'b','bold':True}
    plans={'01':{'styles':[a,b]}}
    result=shared_styles(plans,{'groups':[{'canonical':'01:b','members':['01:a','01:b']}]},'01')
    assert result['a']['bold'] is True


def test_actual_render_gate_rejects_background_and_outside_changes():
    from page_typography import verify_rendered_change
    import numpy as np
    ir={'canvas':{'source_px':[30,20]},'nodes':[{'type':'text','bbox':[5,5,10,5]}]}
    bg=np.zeros((20,30,3),np.uint8);old=bg.copy();old[7,7]=255
    candidate=bg.copy();candidate[7,8]=255
    assert verify_rendered_change(old,candidate,bg,bg,ir)
    candidate[18,28]=255
    with pytest.raises(ValueError,match='protected'):verify_rendered_change(old,candidate,bg,bg,ir)
    changed=bg.copy();changed[0,0]=255
    with pytest.raises(ValueError,match='Non-text'):verify_rendered_change(old,old,bg,changed,ir)
