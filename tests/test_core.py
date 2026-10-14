"""Unit/fault tests. FakeBackend isolates orchestration; real rendering is tested separately."""
import copy
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from image2ppt.compiler import ArtifactBackend, execute
from image2ppt.contracts import ProbeCase, ProbeCases
from image2ppt.providers import ManualDraftProvider, ReplayProvider
from image2ppt.quality import edit_copy, evaluate, inspect_pptx, probe
from image2ppt.runner import Runner, regress
from image2ppt.storage import digest, inventory, local_file, read_json, verify_files, write_json


def miniature(file, text='A', grouped=False):
    shape=f'''<p:sp><p:nvSpPr><p:cNvPr id="2" name="title"/></p:nvSpPr><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="285750" cy="95250"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr b="0"><a:solidFill><a:srgbClr val="111111"/></a:solidFill></a:rPr><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>'''
    if grouped:
        shape='<p:grpSp><p:nvGrpSpPr><p:cNvPr id="3" name="g"/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="285750" cy="95250"/><a:chOff x="0" y="0"/><a:chExt cx="285750" cy="95250"/></a:xfrm></p:grpSpPr>'+shape+'</p:grpSp>'
    parts={
      'ppt/presentation.xml':'<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="609600" cy="304800"/></p:presentation>',
      'ppt/_rels/presentation.xml.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="slides/slide1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"/></Relationships>',
      'ppt/slides/slide1.xml':'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree>'+shape+'</p:spTree></p:cSld></p:sld>'}
    with ZipFile(file,'x') as z:
        for name,data in parts.items(): z.writestr(name,data)


class FakeBackend:
    """Not a production renderer; deterministic stage fault injection."""
    def __init__(self,root):
        self.root=root; self.timeout=300; self.compiles=0; self.renders=0; self.fail_render=False; self.version='v1'; self.residue=False
        write_json(root/'receipt.json',{'fake':True})
    def doctor(self): return {'implementation_sha256':self.version,'fake':True}
    def validate(self,file):
        data=read_json(file)
        if data.get('schema_version')!='0.1': raise ValueError('unsupported IR')
        return {'valid':True}
    def compile(self,file):
        self.compiles+=1
        directory=file.parent/'output'; directory.mkdir()
        shutil.copytree(file.parent/'source',directory/'source')
        shutil.copytree(file.parent/'assets',directory/'assets')
        shutil.copyfile(file,directory/'slideir.snapshot.json')
        miniature(directory/'result.pptx')
        write_json(directory/'quality.json',{'validation_receipt':'receipt.json'})
        return {'output_dir':str(directory),'pptx':str(directory/'result.pptx')}
    def render(self,pptx,output):
        self.renders+=1
        if self.fail_render: raise TimeoutError('injected renderer timeout')
        output.mkdir()
        image=Image.new('RGB',(64,32),'white')
        if self.residue:
            from PIL import ImageDraw
            ImageDraw.Draw(image).rectangle((2,2,15,8),fill='black')
        file=output/'page-001.png'; image.save(file)
        return {'pptx_sha256':digest(pptx),'pages':[{'path':str(file)}]}


class CoreTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.build').mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='core-test-',dir=ROOT/'.build')
        self.root=Path(self.temporary.name).resolve(); self.root.relative_to((ROOT/'.build').resolve())
        source=self.root/'bundle'; (source/'source').mkdir(parents=True); (source/'assets').mkdir()
        Image.new('RGB',(64,32),'white').save(source/'source/page.png')
        self.ir={'schema_version':'0.1','source_image':'source/page.png','source_sha256':digest(source/'source/page.png'),
                 'canvas':{'source_px':[64,32],'target_px':[64,32]},'background':{'color':'FFFFFF'},'review_status':'needs_review','review_notes':[],
                 'assets':[],'groups':[],'evidence':[{'id':'e','kind':'text','bbox':[0,0,30,10],'text':'A'}],
                 'nodes':[{'id':'title','type':'text','bbox':[0,0,30,10],'z':1,'evidence_ids':['e'],'text':'A','font':{'ea':'Arial','latin':'Arial','size_pt':6},'color':'111111'}]}
        self.input=source/'slideir.json'; write_json(self.input,self.ir)
        self.backend=FakeBackend(self.root)
    def tearDown(self):
        self.root.relative_to((ROOT/'.build').resolve())
        self.temporary.cleanup()
    def run_build(self): return Runner(self.backend).replay(self.root/'runs',ir=self.input)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for text in ['{"a":1,"a":2}','{"a":NaN}']:
            file=self.root/'bad.json'; file.write_text(text)
            with self.assertRaises(ValueError): read_json(file)
    def test_paths_are_confined(self):
        for path in ['../receipt.json','https://example.org/a','C:/a']:
            with self.assertRaises((ValueError,FileNotFoundError)): local_file(self.input.parent,path)
    def test_missing_and_wrong_hash(self):
        files=inventory(self.input.parent,['slideir.json'])
        files['slideir.json']['sha256']='0'*64
        with self.assertRaises(ValueError): verify_files(self.input.parent,files)
        with self.assertRaises(FileNotFoundError): verify_files(self.input.parent,{'missing':files['slideir.json']})
    def test_raw_provider_draft_rejected(self):
        file=self.root/'draft.json'; write_json(file,{'regions':[]})
        with self.assertRaises(ValueError): ManualDraftProvider().load(file)
    def test_replay_manifest_works_after_original_input_disappears(self):
        result=self.run_build()
        self.input.rename(self.input.with_suffix('.original'))
        loaded=ReplayProvider().load(result['build'])
        other=Runner(self.backend).replay(self.root/'other',build=result['build'])
        self.assertTrue(Path(loaded.ir_file).exists()); self.assertEqual(other['execution_status'],'COMPLETED')
    def test_completed_resume_reuses_verified_revision(self):
        first=self.run_build(); second=Runner(self.backend).resume(first['run'])
        self.assertTrue(second['reused']); self.assertEqual(self.backend.compiles,1)
    def test_renderer_failure_resumes_without_recompiling(self):
        self.backend.fail_render=True
        with self.assertRaises(RuntimeError): self.run_build()
        run=next((self.root/'runs').glob('run-*'))
        self.assertEqual(read_json(run/'run.json')['status'],'FAILED')
        self.backend.fail_render=False
        result=Runner(self.backend).resume(run)
        self.assertEqual(result['execution_status'],'COMPLETED'); self.assertEqual(self.backend.compiles,1)
    def test_corrupt_cache_is_not_reused(self):
        self.backend.fail_render=True
        with self.assertRaises(RuntimeError): self.run_build()
        run=next((self.root/'runs').glob('run-*')); state=read_json(run/'run.json')
        cached=run/state['stages']['compile']['data']['bundle']/'result.pptx'; cached.write_bytes(b'corruption')
        self.backend.fail_render=False; Runner(self.backend).resume(run)
        self.assertEqual(self.backend.compiles,2); self.assertTrue(read_json(run/'run.json')['invalidated'])
    def test_implementation_change_invalidates_downstream(self):
        first=self.run_build(); self.backend.version='v2'
        second=Runner(self.backend).resume(first['run'])
        self.assertEqual(self.backend.compiles,2); self.assertNotEqual(first['build'],second['build'])
    def test_modified_input_replay_has_new_key(self):
        first=self.run_build(); self.ir['nodes'][0]['font']['size_pt']=7; write_json(self.input,self.ir)
        second=self.run_build()
        self.assertNotEqual(read_json(first['build'])['input_key'],read_json(second['build'])['input_key'])
    def test_damaged_committed_artifact_creates_new_revision(self):
        first=self.run_build(); Path(first['preview']).write_bytes(b'bad')
        second=Runner(self.backend).resume(first['run'])
        self.assertNotEqual(first['build'],second['build']); self.assertEqual(Path(first['preview']).read_bytes(),b'bad')
    def test_sqlite_index_can_be_reconstructed(self):
        first=self.run_build(); database=Path(first['run']).parent/'index.sqlite3'; database.unlink()
        Runner(self.backend).resume(first['run'])
        from contextlib import closing
        with closing(sqlite3.connect(database)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM revisions').fetchone()[0],1)
    def test_f04_compares_actual_styles(self):
        file=self.root/'styles.pptx'; miniature(file,'AB')
        ir=copy.deepcopy(self.ir); ir['nodes'][0].update(text='AB',runs=[{'text':'A','color':'111111','bold':False},{'text':'B','color':'0068D6','bold':True}])
        result=evaluate(ir,inspect_pptx(file))
        found=next(c for c in result['checks'] if c['id']=='run_style:title')
        self.assertEqual(found['status'],'FAIL'); self.assertEqual(found['evidence']['mismatch_character_indices'],[1])
    def test_direct_delete_and_original_preserved(self):
        file=self.root/'original.pptx'; miniature(file); before=digest(file)
        changed=edit_copy(file,self.root/'changed.pptx',ProbeCase(id='delete',action='delete_text',targets=['title']))
        self.assertEqual(changed['slides'][0]['objects'],[]); self.assertEqual(before,digest(file))
    def test_direct_group_move_preserves_child_transform(self):
        file=self.root/'group.pptx'; miniature(file,grouped=True)
        after=edit_copy(file,self.root/'moved.pptx',ProbeCase(id='move',action='move_group',targets=['g'],delta_px=[24.0,16.0]))
        self.assertEqual(after['slides'][0]['objects'][0]['transform']['off'],{'x':24*9525,'y':16*9525})
    def test_probe_contract_rejects_ambiguous_operations(self):
        for data in [dict(id='x',action='delete_text',targets=['title'],expected_solid='FFFFFF'),dict(id='x',action='move_group',targets=['title'],delta_px=[float('inf'),0.0])]:
            with self.assertRaises(ValueError): ProbeCase(**data)
    def test_labelled_synthetic_residue_detects_failure(self):
        build=self.run_build(); self.backend.residue=True
        cases=self.root/'cases.json'; write_json(cases,{'cases':[{'id':'residue','action':'delete_text','targets':['title'],'roi':[0,0,30,10],'synthetic':True,'expected_solid':'FFFFFF'}]})
        result=probe(build['build'],cases,self.backend)
        found=next(c for c in result['cases'][0]['checks'] if c['id']=='labelled_residue')
        self.assertEqual(found['status'],'FAIL'); self.assertTrue(result['original_preserved'])
    def test_regression_continues_after_bad_page(self):
        files=inventory(self.input.parent,['slideir.json','source/page.png'])
        suite=self.root/'suite.json'; write_json(suite,{'pages':[{'id':'bad','ir':'absent.json','baseline_files':files},{'id':'good','ir':'bundle/slideir.json','baseline_files':files}]})
        result=regress(suite,self.root/'batch',self.backend)
        self.assertEqual([p['execution_status'] for p in result['pages']],['FAILED','COMPLETED'])
    def test_empty_truncated_and_failed_subprocess_responses(self):
        for code in ['pass',"print('{')","print('[]')","raise SystemExit(1)"]:
            with self.assertRaises((ValueError,RuntimeError)): execute([sys.executable,'-c',code],self.root,2)
    def test_subprocess_timeout(self):
        with self.assertRaises(TimeoutError): execute([sys.executable,'-c','import time; time.sleep(30)'],self.root,0.2)

    def test_timeout_kills_owned_descendant(self):
        import time
        marker=self.root/'descendant-survived'
        ready=self.root/'descendant-started'
        child='import time; from pathlib import Path; Path('+repr(str(ready))+').write_text("started"); time.sleep(2); Path('+repr(str(marker))+').write_text("alive")'
        parent='import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'+repr(child)+']); time.sleep(30)'
        with self.assertRaises(TimeoutError): execute([sys.executable,'-c',parent],self.root,0.6)
        self.assertTrue(ready.exists())
        time.sleep(2)
        self.assertFalse(marker.exists())

    def test_canonical_validator_rejects_duplicate_ids(self):
        duplicate=copy.deepcopy(self.ir['nodes'][0]); duplicate['z']=2
        self.ir['nodes'].append(duplicate); write_json(self.input,self.ir)
        with self.assertRaises(RuntimeError): ArtifactBackend(ROOT).validate(self.input)

    def test_damaged_pptx_is_rejected(self):
        pptx=self.root/'broken.pptx'; pptx.write_bytes(b'not a zip')
        from zipfile import BadZipFile
        with self.assertRaises(BadZipFile): inspect_pptx(pptx)

    def test_output_outside_compiler_root(self):
        tools=self.root/'toolchain'; tools.mkdir()
        backend=FakeBackend(tools)
        result=Runner(backend).replay(self.root/'elsewhere',ir=self.input)
        self.assertEqual(result['execution_status'],'COMPLETED')
        self.assertTrue(list((tools/'.build').glob('core-input-*/output/result.pptx')))
        manifest=read_json(result['build'])
        verify_files(Path(result['build']).parent,manifest['files'])


if __name__=='__main__': unittest.main()
