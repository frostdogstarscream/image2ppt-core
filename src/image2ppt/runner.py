import os
import shutil
import time
from pathlib import Path
from uuid import uuid4
from PIL import Image
from . import __version__
from .compiler import ArtifactBackend
from .contracts import BuildManifest, ProviderResult, RegressionSuite, RunState
from .prepare import import_bundle
from .providers import ManualDraftProvider, ReplayProvider
from .quality import evaluate, inspect_pptx, probe
from .storage import digest, index_run, inventory, key, local_file, now, read_json, run_lock, verify_files, write_json


class Runner:
    def __init__(self, backend=None):
        self.backend=backend or ArtifactBackend()

    def replay(self, output, *, ir=None, build=None):
        if bool(ir)==bool(build):
            raise ValueError('provide exactly one of ir/build')
        root=Path(output).resolve()
        root.mkdir(parents=True,exist_ok=True)
        run=root/('run-'+uuid4().hex[:16])
        run.mkdir()
        state={'format_version':'1','status':'PENDING','origin':{'kind':'replay' if build else 'manual','file':str(Path(build or ir).resolve())},
               'config':{'timeout_seconds':self.backend.timeout,'network_calls':0,'project_root':str(self.backend.root)},'stages':{},'created_at':now()}
        write_json(run/'run.json',state,exclusive=True)
        return self.resume(run)

    def resume(self, run):
        run=Path(run).resolve(strict=True)
        with run_lock(run):
            state=RunState.model_validate(read_json(run/'run.json')).model_dump(exclude_none=True)
            if state.get('format_version')!='1':
                raise ValueError('unsupported run format')
            if state.get('config',{}).get('network_calls')!=0:
                raise ValueError('M1 does not support network calls')
            self.backend.timeout=state['config']['timeout_seconds']
            def save():
                RunState.model_validate(state)
                write_json(run/'run.json',state)
                index_run(run,state)
            try:
                state['status']='RUNNING'; state.pop('error',None); save()
                if 'input' not in state:
                    provider=(ReplayProvider() if state['origin']['kind']=='replay' else ManualDraftProvider()).load(state['origin']['file'])
                    self.backend.validate(provider.ir_file)
                    destination=run/'input'/uuid4().hex[:12]
                    files=import_bundle(provider.ir_file,destination)
                    self.backend.validate(destination/'slideir.json')
                    state['provider']=provider.model_dump()
                    state['input']={'directory':destination.relative_to(run).as_posix(),'files':files,'key':key(files)}
                    save()
                provider=ProviderResult.model_validate(state['provider'])
                input_dir=(run/state['input']['directory']).resolve(strict=True)
                input_dir.relative_to(run)
                verify_files(input_dir,state['input']['files'])
                if key(state['input']['files'])!=state['input']['key']:
                    raise ValueError('input manifest key mismatch')
                environment=self.backend.doctor()
                pipeline_key=key({'input':state['input']['key'],'environment':environment,'config':state['config']})
                differences=[]
                if provider.previous_environment:
                    differences=[k for k in sorted(set(environment)|set(provider.previous_environment)) if environment.get(k)!=provider.previous_environment.get(k)]
                # Recover a commit interrupted before index/state update.
                for file in sorted((run/'revisions').glob('r*/build.json'),reverse=True):
                    try:
                        manifest=BuildManifest.model_validate(read_json(file))
                        if manifest.pipeline_key!=pipeline_key or manifest.run_id!=run.name:
                            continue
                        verify_files(file.parent,manifest.files)
                        if state.get('build')==file.relative_to(run).as_posix() and state.get('build_sha256')!=digest(file):
                            continue
                        state.update(status='COMPLETED',build=file.relative_to(run).as_posix(),build_sha256=digest(file)); save()
                        return self.result(run,file,True)
                    except (ValueError,OSError):
                        continue

                def stage(name, stage_key, operation):
                    cached=state['stages'].get(name)
                    if cached and cached['key']==stage_key:
                        try:
                            verify_files(run,cached['files'])
                            cached['cache_hit']=True; save()
                            return cached['data']
                        except (ValueError,OSError):
                            state.setdefault('invalidated',[]).append({'stage':name,'reason':'artifact verification failed','at':now()})
                    workspace=run/'work'/(name+'-'+uuid4().hex[:12])
                    workspace.mkdir(parents=True)
                    start=time.monotonic()
                    data,files=operation(workspace)
                    record={'key':stage_key,'elapsed_seconds':round(time.monotonic()-start,4),'data':data,
                            'files':inventory(run,[Path(p).relative_to(run).as_posix() for p in files]),'cache_hit':False}
                    state['stages'][name]=record; save()
                    return data

                def validate(workspace):
                    result=self.backend.validate(input_dir/'slideir.json')
                    file=workspace/'validation.json'; write_json(file,result,exclusive=True)
                    return {'report':file.relative_to(run).as_posix()},[file]
                stage('validate',key([state['input']['key'],environment['implementation_sha256']]),validate)

                def compile_stage(workspace):
                    compiler_input=workspace/'input'
                    try:
                        workspace.relative_to(self.backend.root)
                    except ValueError:
                        # The retained Artifact finalizer confines final files to its
                        # toolchain root. Compile there, then copy the verified bundle.
                        compiler_input=self.backend.root/'.build'/('core-input-'+uuid4().hex)
                    import_bundle(input_dir/'slideir.json',compiler_input)
                    result=self.backend.compile(compiler_input/'slideir.json')
                    bundle=Path(result['output_dir']).resolve(strict=True)
                    bundle.relative_to(compiler_input)
                    if compiler_input!=workspace/'input':
                        shutil.copytree(bundle,workspace/'compiled-bundle')
                        bundle=workspace/'compiled-bundle'
                    # Preserve validator receipt even if the legacy .build directory is later unavailable.
                    legacy=read_json(bundle/'quality.json')
                    receipt=(self.backend.root/legacy['validation_receipt']).resolve(strict=True)
                    receipt.relative_to(self.backend.root)
                    shutil.copyfile(receipt,bundle/'validation.legacy.json')
                    return {'bundle':bundle.relative_to(run).as_posix()},[p for p in bundle.rglob('*') if p.is_file()]
                compiled=stage('compile',key([pipeline_key,'compile']),compile_stage)
                bundle=(run/compiled['bundle']).resolve(strict=True); bundle.relative_to(run)
                pptx=local_file(bundle,'result.pptx')

                def render_stage(workspace):
                    result=self.backend.render(pptx,workspace/'pages')
                    if len(result['pages'])!=1:
                        raise ValueError('SlideIR 0.1 requires one rendered page')
                    ir_data=read_json(input_dir/'slideir.json')
                    png=Path(result['pages'][0]['path']).resolve(strict=True); png.relative_to(workspace)
                    with Image.open(png) as image:
                        if list(image.size)!=ir_data['canvas']['target_px']:
                            raise ValueError('renderer canvas differs from SlideIR')
                    receipt=workspace/'render.json'; write_json(receipt,result,exclusive=True)
                    return {'preview':png.relative_to(run).as_posix(),'receipt':receipt.relative_to(run).as_posix()},[png,receipt]
                rendered=stage('render',key([pipeline_key,digest(pptx),'render']),render_stage)

                def check_stage(workspace):
                    objects=inspect_pptx(pptx)
                    ir_data=read_json(bundle/'slideir.snapshot.json')
                    if objects['slide_count']!=1 or objects['size_emu']!=[round(v*9525) for v in ir_data['canvas']['target_px']]:
                        raise ValueError('PPTX page geometry mismatch')
                    review=evaluate(ir_data,objects)
                    review['environment_differences']=differences
                    review['render_binding']={'pptx_sha256':digest(pptx),'preview_sha256':digest(local_file(run,rendered['preview']))}
                    for name,data in [('objects.json',objects),('review.json',review)]:
                        write_json(workspace/name,data,exclusive=True)
                    return {'objects':(workspace/'objects.json').relative_to(run).as_posix(),'review':(workspace/'review.json').relative_to(run).as_posix()},[workspace/'objects.json',workspace/'review.json']
                checked=stage('check',key([pipeline_key,digest(pptx),digest(local_file(run,rendered['preview'])),'check']),check_stage)

                destination=run/'staging'/uuid4().hex[:12]
                destination.mkdir(parents=True)
                for item in bundle.rglob('*'):
                    if item.is_file():
                        relative=item.relative_to(bundle)
                        if relative.as_posix()=='slideir.snapshot.json': relative=Path('slideir.json')
                        elif relative.as_posix()=='quality.json': relative=Path('legacy-quality.json')
                        elif relative.as_posix()=='preview.png': continue
                        target=destination/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(item,target)
                for source,name in [(rendered['preview'],'preview.png'),(checked['objects'],'objects.json'),(checked['review'],'review.json')]:
                    shutil.copyfile(local_file(run,source),destination/name)
                write_json(destination/'config.json',state['config'],exclusive=True)
                self.backend.validate(destination/'slideir.json')
                revisions=run/'revisions'; revisions.mkdir(exist_ok=True)
                index=max([int(p.name[1:]) for p in revisions.glob('r[0-9]*') if p.is_dir()]+[0])+1
                revision=f'r{index:03d}'
                files=inventory(destination,[p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()])
                manifest=BuildManifest(core_version=__version__,run_id=run.name,revision=revision,quality_status='REVIEW_REQUIRED',
                    input_key=state['input']['key'],pipeline_key=pipeline_key,provider=provider,environment=environment,
                    environment_differences=differences,config=state['config'],stages=[{'name':n,**s} for n,s in state['stages'].items()],
                    files=files,created_at=now())
                write_json(destination/'build.json',manifest.model_dump(),exclusive=True)
                final=revisions/revision
                os.rename(destination,final)
                state.update(status='COMPLETED',build=(final/'build.json').relative_to(run).as_posix(),build_sha256=digest(final/'build.json'))
                save()
                return self.result(run,final/'build.json',False)
            except Exception as error:
                state.update(status='FAILED',error={'type':type(error).__name__,'message':str(error),'at':now()})
                save()
                raise RuntimeError(f"run failed; evidence preserved at {run}: {error}") from error

    @staticmethod
    def result(run, build, reused):
        manifest=BuildManifest.model_validate(read_json(build))
        return {'run':str(run),'build':str(build),'pptx':str(build.parent/'result.pptx'),
                'preview':str(build.parent/'preview.png'),'review':str(build.parent/'review.json'),
                'execution_status':'COMPLETED','quality_status':manifest.quality_status,'reused':reused,'network_calls':0}


def regress(suite_file, output, backend):
    suite_file=Path(suite_file).resolve(strict=True)
    suite=RegressionSuite.model_validate(read_json(suite_file))
    directory=Path(output).resolve()/('regression-'+uuid4().hex[:12]); directory.mkdir(parents=True)
    shutil.copyfile(suite_file,directory/'suite.json')
    results=[]
    for page in suite.pages:
        try:
            file=(suite_file.parent/page.ir).resolve(strict=True)
            verify_files(file.parent,page.baseline_files)
            result=Runner(backend).replay(directory/page.id,ir=file)
            if page.cases:
                cases=(suite_file.parent/page.cases).resolve(strict=True)
                result['probes']=probe(result['build'],cases,backend)
                if result['probes']['execution_status']=='FAILED':
                    result['execution_status']='FAILED'
            results.append({'id':page.id,**result})
        except Exception as error:
            results.append({'id':page.id,'execution_status':'FAILED','error':str(error)})
        write_json(directory/'report.json',{'suite_sha256':digest(suite_file),'pages':results,'network_calls':0,'human_acceptance':False})
    return {'report':str(directory/'report.json'),'pages':results,'execution_status':'FAILED' if any(p['execution_status']=='FAILED' for p in results) else 'COMPLETED'}
