"""M1.1 boundary regressions; real compilation is verified separately."""
import copy
import json
import shutil
import sqlite3
import subprocess
import sys
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
import test_core as baseline
ROOT, miniature = baseline.ROOT, baseline.miniature
from image2ppt.contracts import ProbeCases, RegressionSuite
from image2ppt.quality import evaluate, inspect_pptx, style_value
from image2ppt.runner import Runner, regress
from image2ppt.storage import digest, read_json, rebuild_index, write_json
from image2ppt.compiler import ArtifactBackend


class M11Tests(unittest.TestCase):
    setUp = baseline.CoreTests.setUp
    tearDown = baseline.CoreTests.tearDown
    run_build = baseline.CoreTests.run_build

    def test_installed_code_change_invalidates_real_fingerprint_and_cache(self):
        installed=self.root/'installed'; installed.mkdir()
        shutil.copytree(ROOT/'src/image2ppt',installed/'image2ppt',ignore=shutil.ignore_patterns('__pycache__'))
        module=installed/'image2ppt/quality.py'
        original=module.read_text(encoding='utf-8')
        source_hash=digest(ROOT/'src/image2ppt/quality.py')
        code='''import json,sys
sys.path.insert(0,sys.argv[1])
from image2ppt.compiler import ArtifactBackend
from image2ppt import quality
ArtifactBackend.invoke=lambda *a: {"engine":"fixture","artifact_version":"1","helpers":{}}
result=ArtifactBackend(sys.argv[2]).doctor()
result['behavior']=quality.m11_marker()
print(json.dumps(result))'''
        def fingerprint(value):
            module.write_text(original+'\ndef m11_marker(): return '+repr(value)+'\n',encoding='utf-8')
            result=subprocess.run([sys.executable,'-B','-c',code,str(installed),str(ROOT)],capture_output=True,text=True,encoding='utf-8',timeout=20,check=True)
            return json.loads(result.stdout)
        first=fingerprint('old'); self.backend.version=first['implementation_sha256']
        build=self.run_build()
        second=fingerprint('new-behavior'); self.backend.version=second['implementation_sha256']
        resumed=Runner(self.backend).resume(build['run'])
        self.assertNotEqual(first['behavior'],second['behavior'])
        self.assertNotEqual(first['python_core_fingerprint'],second['python_core_fingerprint'])
        self.assertEqual(first['js_compiler_fingerprint'],second['js_compiler_fingerprint'])
        self.assertEqual(source_hash,digest(ROOT/'src/image2ppt/quality.py'))
        self.assertEqual(self.backend.compiles,2)
        self.assertNotEqual(build['build'],resumed['build'])
        self.assertEqual(Path(second['python_core']['package_directory']),installed/'image2ppt')

    def test_corrupt_index_does_not_block_resume_or_change_original(self):
        build=self.run_build(); run=Path(build['run']); db=run.parent/'index.sqlite3'
        db.write_bytes(b'broken sqlite original'); before=digest(db)
        with self.assertLogs('image2ppt.storage',level='WARNING'):
            result=Runner(self.backend).resume(run)
        self.assertTrue(result['reused']); self.assertEqual(before,digest(db))
        self.assertEqual(read_json(run/'index-warning.json')['status'],'DEGRADED')
        rebuilt=rebuild_index(run.parent,run.parent/'rebuilt.sqlite3')
        self.assertEqual(rebuilt['runs_indexed'],1); self.assertEqual(before,digest(db))
        with closing(sqlite3.connect(rebuilt['database'])) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM revisions').fetchone()[0],1)
        with self.assertRaises(FileExistsError): rebuild_index(run.parent,db)

    def test_toolchain_change_has_separate_fingerprint(self):
        tools=self.root/'tools'; (tools/'scripts').mkdir(parents=True)
        script=tools/'scripts/main.mjs'; script.write_text('// before')
        with patch.object(ArtifactBackend,'invoke',side_effect=lambda *args: {'engine':'fixture'}):
            first=ArtifactBackend(tools).doctor()
            script.write_text('// after')
            second=ArtifactBackend(tools).doctor()
        self.assertEqual(first['python_core_fingerprint'],second['python_core_fingerprint'])
        self.assertNotEqual(first['js_compiler_fingerprint'],second['js_compiler_fingerprint'])

    def test_index_diagnostic_write_error_does_not_mask_task_error(self):
        self.backend.fail_render=True
        with patch('image2ppt.storage.index_run',side_effect=sqlite3.DatabaseError('bad index')), \
             patch('image2ppt.storage.write_json',side_effect=PermissionError('diagnostic blocked')), \
             self.assertLogs('image2ppt.storage',level='WARNING'), \
             self.assertRaisesRegex(RuntimeError,'injected renderer timeout'):
            self.run_build()

    def test_locked_index_is_bounded_and_resume_succeeds(self):
        build=self.run_build(); run=Path(build['run'])
        with closing(sqlite3.connect(run.parent/'index.sqlite3')) as lock:
            lock.execute('BEGIN EXCLUSIVE')
            with self.assertLogs('image2ppt.storage',level='WARNING'):
                result=Runner(self.backend).resume(run)
            self.assertTrue(result['reused'])
            self.assertEqual(read_json(run/'index-warning.json')['attempts'],2)
            lock.rollback()
        self.assertTrue(Runner(self.backend).resume(run)['reused'])

    def test_original_render_error_survives_persistent_index_error(self):
        output=self.root/'runs'; output.mkdir(); (output/'index.sqlite3').write_bytes(b'bad')
        self.backend.fail_render=True
        with self.assertLogs('image2ppt.storage',level='WARNING'), self.assertRaisesRegex(RuntimeError,'injected renderer timeout'):
            self.run_build()
        run=next(output.glob('run-*')); state=read_json(run/'run.json')
        self.assertEqual(state['error']['type'],'TimeoutError')
        self.assertEqual(state['status'],'FAILED')

    def style_case(self,color,bold):
        file=self.root/'style.pptx'; miniature(file)
        actual=inspect_pptx(file); actual['slides'][0]['objects'][0]['runs'][0].update(color=color,bold=bold)
        ir=copy.deepcopy(self.ir); ir['nodes'][0]['runs']=[{'text':'A','color':'0068D6','bold':False}]
        return next(c for c in evaluate(ir,actual)['checks'] if c['id']=='run_style:title')

    def test_known_color_error_with_unknown_bold_fails(self):
        result=self.style_case('000000',None)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['evidence']['property_mismatches']['color'],[0])
        self.assertEqual(result['evidence']['unknown_properties']['bold'],[0])

    def test_known_bold_error_with_unknown_color_fails(self):
        self.assertEqual(self.style_case(None,True)['status'],'FAIL')

    def test_matching_styles_pass(self):
        self.assertEqual(self.style_case('0068D6',False)['status'],'PASS')

    def test_only_unknown_styles_require_review(self):
        self.assertEqual(self.style_case('0068D6',None)['status'],'REVIEW_REQUIRED')

    def test_explicit_theme_color_does_not_fall_back_to_default_rgb(self):
        ns='http://schemas.openxmlformats.org/drawingml/2006/main'
        explicit=ET.fromstring(f'<a:rPr xmlns:a="{ns}"><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:rPr>')
        default=ET.fromstring(f'<a:defRPr xmlns:a="{ns}"><a:solidFill><a:srgbClr val="FF0000"/></a:solidFill></a:defRPr>')
        self.assertIsNone(style_value([explicit,default],'color'))
        self.assertEqual(style_value([None,default],'color'),'FF0000')

    def test_image_replaced_by_same_named_shape_is_failure(self):
        file=self.root/'object.pptx'; miniature(file)
        ir=copy.deepcopy(self.ir); ir['nodes'][0]['type']='image'
        review=evaluate(ir,inspect_pptx(file))
        self.assertIn('object_type:title',review['known_failures'])

    def test_empty_suites_are_rejected_before_creating_output(self):
        for model,data in [(RegressionSuite,{'pages':[]}),(ProbeCases,{'cases':[]})]:
            with self.assertRaisesRegex(ValueError,'at least 1'): model.model_validate(data)
        file=self.root/'empty.json'; write_json(file,{'pages':[]})
        with self.assertRaisesRegex(ValueError,'at least 1'): regress(file,self.root/'batch',self.backend)
        self.assertFalse((self.root/'batch').exists())


if __name__=='__main__': unittest.main()
