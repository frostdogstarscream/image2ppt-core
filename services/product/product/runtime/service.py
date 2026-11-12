"""Business tools reuse production perception, recovery and private adapters."""
import json
import os
from pathlib import Path
import shutil

from pydantic import BaseModel, ConfigDict, Field

from ..materialize import digest, write
from ..parallel_contracts import SceneGraph
from ..parallel_store import Store, key
from ..resource_lease import lease
from .state import RuntimeState, RuntimeConflict
from .primitives import PrimitiveTools
from .exploration import ExplorationTools


TOOLS = ('get_job_context', 'run_perception', 'inspect_scene', 'rebuild_region',
         'repair_region', 'validate_candidate', 'commit_region', 'compose_delivery',
         'validate_office', 'get_evidence', 'record_visual_review', 'finalize_job')
TOOLS += ('inspect_image', 'crop_image', 'compare_images', 'run_ocr', 'analyze_text_regions',
          'analyze_image', 'decompose_layers', 'segment_object', 'extract_background',
          'fit_background', 'inpaint_background', 'complete_text_mask', 'validate_mask',
          'unmix_foreground', 'validate_rgba', 'list_fonts', 'fit_text_layout',
          'create_scene', 'update_scene', 'create_candidate', 'commit_candidate',
          'rollback_region', 'compile_pptx', 'run_image_script', 'run_parallel_pipeline')
TOOLS += ('record_observation',)


class VisualReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    passed: bool
    findings: list[dict] = Field(max_length=100)
    minor_deviations: list[dict] = Field(max_length=100)
    evidence_ids: list[str] = Field(min_length=2)
    checked: list[str]
    reason: str = Field(min_length=1, max_length=4000)


class RuntimeService(ExplorationTools):
    def __init__(self, root):
        self.root = Path(root).resolve()

    def state(self, job_id):
        import re
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise ValueError('invalid job identifier')
        job = self.root/job_id
        if not job.is_relative_to(self.root) or not (job/'request.json').is_file():
            raise ValueError('unknown job')
        request = json.loads((job/'request.json').read_text('utf8'))
        if request.get('orchestrator') != 'opencode' or request.get('visual_review_backend') != 'opencode':
            raise ValueError('job is not registered for OpenCode orchestration')
        return RuntimeState(job)

    def call(self, tool, job_id, *, expected_version=None, idempotency_key='', **arguments):
        if tool not in TOOLS:
            raise ValueError('unknown runtime tool')
        state = self.state(job_id)
        if tool == 'get_job_context':
            result = state.read()
            result['sources'] = [state.register(p, 'source') for p in sorted((state.job/'inputs').glob('*.png'))]
            request=json.loads((state.job/'request.json').read_text('utf8'))
            shortcuts={'run_perception','run_parallel_pipeline','rebuild_region','repair_region'} if request.get('tool_profile')=='basic' else set()
            result['tool_catalog'] = [name for name in TOOLS if name not in shortcuts]
            result['goal'] = json.loads((state.job/'request.json').read_text('utf8')).get('goal', 'Faithful editable PPTX')
            result['budget']={'remaining_seconds':state.remaining_seconds(),
                'wall_time_seconds':request.get('runtime_seconds',7200),
                'agent_turns':request.get('agent_steps',64),'tool_limits':request.get('adaptive_budget',{})}
            return result
        if tool == 'inspect_scene':
            page = self.page(state.read(), arguments['page'])
            graph = SceneGraph.model_validate_json(Path(page['scene']).read_text('utf8'))
            from ..parallel_regions import regions
            diagnostics=[{'object_id':text.id,'owner':graph.ownership[text.id],
                          'ocr_members':graph.text.members.get(text.id,[]),
                          'missing_ocr_members':not graph.text.members.get(text.id)} for text in graph.text.texts
                         if graph.ownership[text.id]!='RESOLVED']
            return {'version': state.read()['version'], 'scene': graph.model_dump(),'ownership_diagnostics':diagnostics,
                    'regions': regions(graph)[0], 'perception_failures': page['failures']}
        if tool == 'get_evidence':
            return [state.evidence(identifier) for identifier in arguments['evidence_ids']]
        if tool in ('commit_candidate', 'commit_region') and expected_version is None:
            raise ValueError('committing requires an explicit scene version')
        method = getattr(self, tool)
        return state.mutate(tool, arguments, expected_version, idempotency_key,
                            lambda current, directory: method(state, current, directory, **arguments))

    @staticmethod
    def page(current, page):
        if type(page) is not int or not 1 <= page <= len(current['pages']) or not current['pages'][page-1]:
            raise ValueError('unknown page')
        return current['pages'][page-1]

    @staticmethod
    def graph(page):
        return SceneGraph.model_validate_json(Path(page['scene']).read_text('utf8'))

    def run_perception(self, state, current, directory):
        if current['pages']:
            raise RuntimeConflict('perception already completed; use existing source-bound evidence')
        from ..parallel_branches import snapshot, perceive, local_ocr, layer_evidence
        from ..parallel_fusion import fuse
        from .provider import planning_actor
        from ..fonts import font_manifest
        from ..parallel_rebuild import build
        from ..parallel_typography import calibrate
        from ..parallel_text_anchor import anchor
        from ..asset_geometry import normalize
        from ..recovery.text_groups import coalesce
        from ..recovery.adaptive_policy import AdaptiveBudget, BudgetedActor
        import numpy as np
        from PIL import Image
        request = json.loads((state.job/'request.json').read_text('utf8'))
        store = Store(state.job/'state')
        budget = AdaptiveBudget(store, request.get('adaptive_budget'))
        provider = BudgetedActor(planning_actor(state.job), budget)
        cache = state.job/'fresh-cache' if request.get('fresh_perception') else Path(
            os.environ.get('IMAGE2PPT_PARALLEL_CACHE', str(state.job/'cache')))
        cache.mkdir(exist_ok=True, parents=True)
        calls = Store(cache/'model-calls')
        store.call = calls.call
        fonts = font_manifest([])
        write(state.job/'fonts.json', fonts)
        current['fonts'] = str(state.job/'fonts.json')
        for name in request['sources']:
            source = (state.job/'inputs'/name).resolve()
            if not source.is_relative_to(state.job/'inputs'):
                raise ValueError('source path escaped inputs')
            folder = directory/f'page-{len(current["pages"])+1:02d}'
            snap = snapshot(source, folder/'snapshot')
            model_manifest = os.environ.get('IMAGE2PPT_OCR_MANIFEST')
            if not model_manifest:
                raise ValueError('offline OCR manifest is required')
            branches, failures = perceive(snap, store, provider, fonts, cache, model_manifest)
            ocr = local_ocr(snap, cache, model_manifest)
            branches = coalesce(branches, ocr, store)
            branches['asset'], geometry = normalize(branches['asset'], store)
            graph = fuse(snap, branches, ocr)
            pixels = np.array(Image.open(source).convert('RGB'))
            with lease('cpu'):
                graph = anchor(graph, ocr, pixels, fonts, store)
                graph = calibrate(graph, pixels, fonts, store, ocr=ocr, adaptive=True)
                graph = anchor(graph, ocr, pixels, fonts, store, fitted=True)
            write(folder/'perceived-scene.json', graph.model_dump())
            write(folder/'ocr.json', ocr)
            write(folder/'asset-geometry.json', geometry)
            layers = next((p.parent for p in (cache/'layers').glob(snap['source_sha256']+'*/manifest.json')),None)
            if layers is not None:
                from ..object_extract import load_page
                load_page(layers)
            # Initial backing has no accepted objects. Every edit goes through a regional action.
            ir, _ = build(graph, snap, fonts, folder/'backing', strategy='off', text_candidates=False)
            current['pages'].append({'snapshot': snap, 'scene': str(ir.parent/'scene.json'),
                'perceived_scene': str(folder/'perceived-scene.json'), 'ir': str(ir),
                'layers': str(layers) if layers else None, 'failures': failures})
        current['perception_complete'] = all(not p['failures'] for p in current['pages'])
        return {'status': 'PERCEIVED', 'pages': len(current['pages']),
                'complete': current['perception_complete']}

    def rebuild_region(self, state, current, directory, *, page, object_id, strategy='auto'):
        row = self.page(current, page)
        graph = self.graph(row)
        if object_id in graph.selected:
            raise RuntimeConflict('object already committed')
        fonts = json.loads(Path(current['fonts']).read_text('utf8'))
        source = row['snapshot']['path']
        baseline = Path(row['ir']).parent
        from ..recovery.adaptive_policy import AdaptiveBudget
        request = json.loads((state.job/'request.json').read_text('utf8'))
        budget = AdaptiveBudget(Store(state.job/'state'), request.get('adaptive_budget'))
        from ..recovery.text_probe import probe_one
        from ..recovery.asset_retry import retry_asset_extraction
        from ..recovery.agent_asset_stage import stage_asset_candidate
        texts = {t.id for t in graph.text.texts}
        assets = {a.id for a in graph.asset.objects}
        layer_manifest=Path(row['layers'])/'manifest.json' if row.get('layers') else None
        fingerprint = key({'object': object_id, 'parent': digest(row['ir']), 'scene':digest(row['scene']),
                           'fonts':digest(current['fonts']),'strategy': strategy,
                           'layers':digest(layer_manifest) if layer_manifest and layer_manifest.exists() else None,
                           'sam':current.get('sam',{}).get(str(page)+':'+object_id),
                           'implementation': self.code_binding()})
        previous = [c for c in current['candidates'].values() if c['fingerprint'] == fingerprint]
        if previous:
            return {'status': 'ALREADY_ATTEMPTED', 'candidate_id': previous[0]['id'],
                    'reason': 'same evidence and strategy already tried'}
        count = sum(c['object_id'] == object_id and c['page'] == page for c in current['candidates'].values())
        if count >= budget.limits['complex_region_candidates']:
            return {'status': 'BLOCKED', 'failure': 'CANDIDATE_BUDGET_EXHAUSTED'}
        candidate = {'id': 'c-'+key(str(directory.relative_to(state.job)))[:24], 'object_id': object_id, 'page': page,
                     'parent_ir_sha256': digest(row['ir']), 'fingerprint': fingerprint,
                     'parent_scene_sha256':digest(row['scene']),
                     'status': 'BLOCKED', 'strategy': strategy, 'code': self.code_binding()}
        try:
            if object_id in texts:
                if strategy not in ('auto', 'next_background'):
                    raise ValueError('unsupported text strategy')
                excluded = [name for c in current['candidates'].values()
                            if c['page'] == page and c['object_id'] == object_id
                            for name in c.get('backgrounds', [])]
                from .background import inpaint_service
                from ..parallel_config import ROOT
                config_file = ROOT/'.runtime/parallel-services.json'
                config = json.loads(config_file.read_text('utf8')).get('lama') if config_file.exists() else None
                inpaint = inpaint_service(config, state.job/'cache/lama', budget) if config else None
                result = probe_one(graph, object_id, source, row['layers'], baseline, fonts,
                                   directory/'text', excluded_candidates=excluded, inpaint=inpaint)
                candidate['backgrounds'] = [a['candidate'] for a in result['attempts']]
                if result['state'] == 'STAGED_LOCAL':
                    candidate.update(status='STAGED_LOCAL', ir=result['staged_ir'],
                                     trial=result['candidate_directory'], kind='text')
                else:
                    candidate.update(failure=result.get('failure', 'BACKGROUND_REPAIR_FAILED'))
            elif object_id in assets:
                if not row['layers']:
                    raise ValueError('LayerD evidence unavailable')
                chosen = 'verified-backing' if strategy == 'auto' else strategy
                if chosen not in ('verified-backing', 'source-halo', 'sam'):
                    raise ValueError('unsupported asset strategy')
                options = {}
                if chosen == 'sam':
                    from ..object_stage import infer_sam
                    plan = directory/'sam-plan.json'
                    write(plan, {'schema': 'LogicalObjects/2', 'source_sha256': graph.source_sha256,
                        'image_size': graph.image_size, 'objects': [a.model_dump() for a in graph.asset.objects],
                        'limitations': []})
                    options = {'sam_plan': plan, 'sam_directory': infer_sam(
                        source, plan, directory/'sam', 'sam3', [object_id])}
                sam = current.get('sam', {}).get(str(page)+':'+object_id)
                if chosen == 'sam' and sam:
                    options = {'sam_plan': sam['plan'], 'sam_directory': sam['directory']}
                result = retry_asset_extraction(graph, object_id, source, row['layers'], baseline,
                                                directory/'asset', strategy=chosen, **options)
                if result['status'] == 'CANDIDATE_READY':
                    ir = stage_asset_candidate(result['candidate_directory'], graph, baseline,
                                               directory/'staged', layer_directory=row['layers'])
                    candidate.update(status='STAGED_LOCAL', ir=str(ir), kind='asset')
                else:
                    candidate.update(failure='ASSET_EXTRACTION_UNCERTAIN', reason=result.get('reason'))
            else:
                raise ValueError('unknown object; inspect scene for supported complete regions')
        except Exception as exc:
            from ..recovery.product_pipeline import failure
            candidate.update(failure=failure(exc, 'generation'), error_type=type(exc).__name__)
        if candidate.get('ir'):
            candidate['files'] = {str(p): digest(p) for p in Path(candidate['ir']).parent.rglob('*') if p.is_file()}
            candidate['evidence'] = [state.register(p, 'candidate_observation') for p in directory.rglob('*.png')
                                     if p.name in ('source_crop.png', 'written_crop.png', 'asset.png', 'recomposed.png')]
        current['candidates'][candidate['id']] = candidate
        write(directory/'candidate-receipt.json', candidate)
        return candidate

    @staticmethod
    def code_binding():
        package = Path(__file__).resolve().parents[1]
        return key({str(p.relative_to(package)): digest(p) for p in package.rglob('*.py')})

    def repair_region(self, state, current, directory, *, candidate_id, strategy):
        previous = current['candidates'][candidate_id]
        if previous['status'] == 'COMMITTED':
            raise ValueError('accepted candidates require no repair')
        if strategy == previous['strategy']:
            raise ValueError('repair requires a changed strategy')
        return self.rebuild_region(state, current, directory, page=previous['page'],
                                   object_id=previous['object_id'], strategy=strategy)

    def validate_candidate(self, state, current, directory, *, candidate_id):
        candidate = current['candidates'][candidate_id]
        if candidate.get('exploratory'):
            preview = self.compile_pptx(state, current, directory/'preview', candidate_id=candidate_id)
            return {'status': 'OBSERVATION_READY', 'preview': preview,
                    'ownership_authority': False, 'uncertainties': ['Formal source-owner and editing checks have not run.']}
        if candidate['status']=='VALIDATED':
            self.check_candidate(current,candidate)
            for path,sha in candidate['validation'].get('files',{}).items():
                if digest(path)!=sha:raise RuntimeConflict('candidate validation evidence changed')
            return {'candidate_id':candidate_id,**candidate['validation'],'cache_hit':True}
        if candidate['status'] != 'STAGED_LOCAL':
            if candidate.get('validation'):
                return {'candidate_id':candidate_id,**candidate['validation'],'cache_hit':True}
            return {'status':'REVIEW_REQUIRED','candidate_id':candidate_id,'office_status':'NOT_RUN',
                    'failure':candidate.get('failure','CANDIDATE_NOT_STAGED'),
                    'failed_checks':candidate.get('failed_checks',[]),'evidence':candidate.get('evidence',[])}
        self.check_candidate(current, candidate)
        ir = Path(candidate['ir'])
        graph = SceneGraph.model_validate_json((ir.parent/'scene.json').read_text('utf8'))
        row = self.page(current, candidate['page'])
        original = self.graph(row)
        if {t.id: t.text for t in graph.text.texts} != {t.id: t.text for t in original.text.texts}:
            raise ValueError('confirmed text changed')
        if graph.source_sha256 != original.source_sha256:
            raise ValueError('source changed')
        from .compiler import compile_local as compile_pages
        from ..parallel_pipeline import effective_texts
        with lease('compiler'):
            artifacts = compile_pages([ir], directory/'compiled', [effective_texts(graph, ir)])
        result = self.office_check(state, artifacts['pptx'], [ir], directory/'office-check')
        if candidate.get('kind')=='text':
            from .office import frame_feedback
            feedback=frame_feedback(result,graph,candidate['object_id'])
            if feedback:
                result['repair_feedback']=feedback
        candidate['validation'] = result
        candidate['status'] = 'VALIDATED' if result['status'] == 'PASS' else 'REVIEW_REQUIRED'
        return {'candidate_id': candidate_id, **result}

    def check_candidate(self, current, candidate):
        row = self.page(current, candidate['page'])
        if candidate['parent_ir_sha256'] != digest(row['ir']):
            raise RuntimeConflict('candidate parent changed; regenerate against current scene')
        if candidate.get('parent_scene_sha256')!=digest(row['scene']):
            raise RuntimeConflict('candidate scene evidence changed; regenerate against current scene')
        if candidate['code'] != self.code_binding():
            raise RuntimeConflict('candidate implementation changed')
        for path, sha in candidate.get('files', {}).items():
            if digest(path) != sha:
                raise RuntimeConflict('candidate bytes changed')

    def commit_region(self, state, current, directory, *, candidate_id):
        candidate = current['candidates'][candidate_id]
        if candidate.get('exploratory'):
            self.check_candidate(current, candidate)
            approval = directory/'submission'
            approval.mkdir(parents=True)
            staged = PrimitiveTools.create_candidate(self, state, current, approval, **candidate['proposal'])
            if staged.get('status') != 'STAGED_LOCAL':
                return {'status': 'SUBMISSION_BLOCKED', 'checks': staged, 'preview_candidate_id': candidate_id}
            checked = self.validate_candidate(state, current, directory/'submission-check', candidate_id=staged['id'])
            if checked.get('status') != 'PASS':
                return {'status': 'SUBMISSION_BLOCKED', 'checks': checked, 'preview_candidate_id': candidate_id}
            return self.commit_region(state, current, directory/'commit', candidate_id=staged['id'])
        if candidate['status'] != 'VALIDATED' or candidate['validation']['status'] != 'PASS':
            raise ValueError('candidate requires successful hard and actual edit checks')
        self.check_candidate(current, candidate)
        for path, sha in candidate['validation'].get('files', {}).items():
            if digest(path) != sha:
                raise RuntimeConflict('candidate Office evidence changed')
        row = self.page(current, candidate['page'])
        row.update(ir=candidate['ir'], scene=str(Path(candidate['ir']).parent/'scene.json'))
        candidate['status'] = 'COMMITTED'
        current.pop('composition', None)
        current.pop('visual_review', None)
        return {'status': 'COMMITTED', 'object_id': candidate['object_id']}

    commit_candidate = commit_region

    def compose_delivery(self, state, current, directory):
        request = json.loads((state.job/'request.json').read_text('utf8'))
        for index in range(1, len(request['sources'])+1):
            if len(current['pages']) < index or not current['pages'][index-1]:
                self.create_scene(state, current, directory/f'raster-page-{index}', page=index, evidence={})
        if not current['pages']:
            raise ValueError('perception is required')
        if current.get('composition'):
            composition=self.composition(current)
            return {'status':'COMPOSED','pptx_sha256':digest(composition['artifacts']['pptx']),'cache_hit':True}
        from .compiler import compile_local as compile_pages
        from ..parallel_pipeline import effective_texts
        irs = [Path(p['ir']) for p in current['pages']]
        with lease('compiler'):
            artifacts = compile_pages(irs, directory/'compiled',
                                      [effective_texts(self.graph(p), ir) for p, ir in zip(current['pages'], irs)])
        binding = {str(p): digest(p) for p in [Path(artifacts['pptx']), *irs]}
        current['composition'] = {'artifacts': artifacts, 'files': binding, 'office': {'status': 'NOT_RUN'}}
        current.pop('visual_review', None)
        return {'status': 'COMPOSED', 'pptx_sha256': digest(artifacts['pptx'])}

    def compile_pptx(self, state, current, directory, *, candidate_id=None):
        if candidate_id is None:
            return self.compose_delivery(state, current, directory)
        candidate = current['candidates'][candidate_id]
        self.check_candidate(current, candidate)
        if not candidate.get('ir'):
            return {'status': 'MISSING_INPUT', 'required': ['candidate IR']}
        from .compiler import compile_local
        from ..parallel_pipeline import effective_texts
        ir = Path(candidate['ir'])
        graph = self.graph({'scene': str(ir.parent/'scene.json')})
        with lease('compiler'):
            artifacts = compile_local([ir], directory/'compiled', [effective_texts(graph, ir)])
        references = [state.register(p, 'exploratory_preview') for p in (directory/'compiled').rglob('*')
                      if p.is_file() and p.suffix.lower() in ('.png', '.pptx')]
        candidate['preview'] = {'artifacts': artifacts, 'pptx_sha256': digest(artifacts['pptx'])}
        return {'status': 'COMPILED', 'candidate_id': candidate_id, 'artifacts': artifacts,
                'evidence': references, 'delivery_approval': False}

    @staticmethod
    def office_check(state, pptx, irs, directory):
        from ..recovery.adaptive_policy import AdaptiveBudget
        identity={'pptx':digest(pptx),'irs':[digest(ir) for ir in irs],
                  'implementation':RuntimeService.code_binding(),
                  'mode':os.environ.get('IMAGE2PPT_OFFICE_MODE','local')}
        if identity['mode']=='remote':
            binding=state.job.parent/'office-worker-binding.json'
            identity['worker']=json.loads(binding.read_text('utf8')) if binding.exists() else {'binding':'unknown'}
        else:
            from ..fonts import font_manifest
            identity['fonts']=font_manifest([])
        receipt=state.root/'office-cache'/(key(identity)+'.json')
        if receipt.exists():
            cached=json.loads(receipt.read_text('utf8'))
            result=cached.get('result',{'status':'UNKNOWN','failure':'OFFICE_RESPONSE_UNKNOWN',
                                      'original_operation':cached['operation']})
            for path,sha in result.get('files',{}).items():
                if digest(path)!=sha:raise RuntimeConflict('cached Office evidence changed')
            write(Path(directory)/'checks.json',result)
            return {**result,'cache_hit':True}
        request=json.loads((state.job/'request.json').read_text('utf8'))
        budget=AdaptiveBudget(Store(state.job/'state'),request.get('adaptive_budget'))
        budget.reserve('office_validations',identity)
        write(receipt,{'identity':identity,'status':'UNKNOWN','operation':str(directory)})
        if os.environ.get('IMAGE2PPT_OFFICE_MODE', 'local') == 'remote':
            from .office_bridge import submit
            result=submit(state.job, pptx, irs, directory)
        else:
            from .office import check_local
            result=check_local(pptx, irs, directory)
        write(receipt,{'identity':identity,'operation':str(directory),'result':result})
        return result

    def run_parallel_pipeline(self, state, current, directory):
        if current['pages']:
            raise ValueError('pipeline shortcut requires a separate unplanned scene')
        from ..parallel_pipeline import run
        from .provider import planning_actor
        prepared = run(state.job, prepare_only=True,provider=planning_actor(state.job))
        current['fonts'] = str(state.job/'fonts.json')
        for graph, snap, ir, failures, layers in prepared:
            current['pages'].append({'scene': str(ir.parent/'scene.json'), 'snapshot': snap,
                'ir': str(ir), 'failures': failures, 'layers': layers})
        # The same production region tools operate on the resulting evidence.
        for page, row in enumerate(current['pages'], 1):
            graph = self.graph(row)
            for identifier in [t.id for t in graph.text.texts]+[a.id for a in graph.asset.objects]:
                if identifier in self.graph(row).selected:
                    continue
                target = directory/f'page-{page}'/key(identifier)[:16]
                target.mkdir(parents=True)
                candidate = self.rebuild_region(state, current, target, page=page, object_id=identifier)
                if candidate.get('status') != 'STAGED_LOCAL':
                    continue
                check = target/'check'
                check.mkdir()
                self.validate_candidate(state, current, check, candidate_id=candidate['id'])
                if candidate['status'] == 'VALIDATED':
                    self.commit_region(state, current, target, candidate_id=candidate['id'])
        result = self.compose_delivery(state, current, directory/'composition')
        return {**result, 'visual_review': 'REQUIRES_AGENT_OBSERVATION', 'backend': 'opencode'}

    def validate_office(self, state, current, directory, *, candidate_id=None):
        if candidate_id:
            candidate = current['candidates'][candidate_id]
            self.check_candidate(current, candidate)
            preview = candidate.get('preview')
            if not preview:
                return {'status': 'MISSING_INPUT', 'required': ['compiled candidate preview']}
            pptx = preview['artifacts']['pptx']
            if digest(pptx) != preview['pptx_sha256']:
                raise RuntimeConflict('preview bytes changed')
            result = self.office_check(state, pptx, [Path(candidate['ir'])], directory)
            return {'status': 'MEASURED' if result['status'] != 'UNKNOWN' else 'UNKNOWN',
                    'checks': result, 'delivery_approval': False}
        composition = self.composition(current)
        result = self.office_check(state, composition['artifacts']['pptx'], [Path(p['ir']) for p in current['pages']], directory)
        composition['office'] = result
        if result['status'] != 'PASS':
            return result
        from ..parallel_review import panels
        required = []
        for i, page in enumerate(current['pages'], 1):
            preview = Path(result['office_directory'])/f'page-{i:02d}-before.png'
            required.append(state.register(page['snapshot']['path'], 'source'))
            required.append(state.register(preview, 'office_preview'))
            images, _ = panels(page['snapshot']['path'], preview, page['ir'],
                               result['office_directory'], i, directory/f'panels-{i:02d}')
            required.extend(state.register(path, 'edit_panel') for path in images)
        required.append(state.register(directory/'checks.json', 'hard_checks'))
        composition['required_evidence'] = required
        current.pop('visual_review', None)
        return {**result, 'required_evidence': required}

    @staticmethod
    def composition(current):
        composition = current.get('composition')
        if not composition:
            raise ValueError('compose delivery first')
        for path, sha in composition['files'].items():
            if digest(path) != sha:
                raise RuntimeConflict('composed artifacts changed')
        return composition

    def record_visual_review(self, state, current, directory, *, review):
        review = VisualReview.model_validate(review)
        required_checks = {'content', 'completeness', 'layout', 'residual', 'background', 'editing'}
        if set(review.checked) != required_checks or (review.passed and review.findings):
            raise ValueError('incomplete or contradictory visual review')
        composition = self.composition(current)
        if composition['office']['status'] != 'PASS':
            raise ValueError('actual editing and residual evidence required')
        required = {r['id'] for r in composition['required_evidence']}
        if not required <= set(review.evidence_ids):
            raise ValueError('review omitted required source, preview or edit evidence')
        for identifier in review.evidence_ids:
            _, reference = state.evidence(identifier)
            if not reference['provided_to_agent']:
                raise ValueError('evidence has not been provided to Agent')
        current['visual_review'] = {**review.model_dump(), 'backend': 'opencode',
                                    'composition_sha256': key(composition),
                                    'status': 'PASS' if review.passed else 'REVIEW_REQUIRED'}
        write(directory/'visual-review.json', current['visual_review'])
        return current['visual_review']

    def finalize_job(self, state, current, directory):
        composition = self.composition(current)
        visual = current.get('visual_review', {})
        if (composition['office']['status'] != 'PASS' or visual.get('status') != 'PASS'
                or visual.get('composition_sha256') != key(composition)):
            return {'status': 'REVIEW_REQUIRED', 'reason': 'current Office and OpenCode evidence required'}
        for reference in composition['required_evidence']:
            state.evidence(reference['id'])
        selected = [self.graph(page).selected for page in current['pages']]
        # Completeness has no independent denominator in this first integration.
        status = 'PARTIAL_EDITABLE' if any(selected) else 'RASTER_ONLY'
        delivery = state.job/'delivery'
        delivery.mkdir(exist_ok=True)
        shutil.copyfile(composition['artifacts']['pptx'], delivery/'result.pptx')
        from ..parallel_rebuild import build
        from ..parallel_pipeline import safe_graph, effective_texts
        from .compiler import compile_local as compile_pages
        fonts = json.loads(Path(current['fonts']).read_text('utf8'))
        stable_irs = []
        for index, page in enumerate(current['pages']):
            ir, _ = build(safe_graph(self.graph(page)), page['snapshot'], fonts,
                          directory/f'stable-{index}', strategy='off', text_candidates=False)
            stable_irs.append(ir)
        with lease('compiler'):
            stable = compile_pages(stable_irs, directory/'stable-compiled',
                                   [effective_texts(self.graph(p), ir) for p, ir in zip(current['pages'], stable_irs)])
        shutil.copyfile(stable['pptx'], delivery/'stable-fallback.pptx')
        office_dir = Path(composition['office']['office_directory'])
        for index in range(1, len(current['pages'])+1):
            shutil.copyfile(office_dir/f'page-{index:02d}-before.png', delivery/f'page-{index:02d}.png')
        delivery_state='SUCCEEDED' if any(selected) else 'REVIEW_REQUIRED'
        result = {'status': delivery_state, 'completion': status, 'visual_review_backend': 'opencode',
                  'office_pass': True, 'visual_review': visual, 'pages': len(selected),
                  'editable_objects': [sorted(page) for page in selected],
                  'unresolved_objects': [[identifier for identifier in self.graph(page).ownership
                                          if identifier not in chosen]
                                         for page, chosen in zip(current['pages'], selected)],
                  'recall': None, 'human_acceptance': False, 'automatic_publication': False,
                  'workflow_states': {'candidate_generation': 'COMPLETED', 'regional_commit': 'COMPLETED',
                                      'final_qa': 'PASS', 'automatic_publication': 'GATED'}}
        write(delivery/'summary.json', result)
        result['artifacts'] = {p.name: digest(p) for p in delivery.iterdir() if p.is_file()}
        write(state.job/'result.json', result)
        current['status'] = delivery_state
        return result
