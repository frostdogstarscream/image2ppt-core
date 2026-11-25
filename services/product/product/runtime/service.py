"""Business tools reuse production perception, recovery and private adapters."""
import json
from collections import Counter
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
TOOLS += ('record_observation', 'list_evidence', 'get_progress',
          'prepare_page_evidence', 'batch_reconstruct_page', 'crop_region',
          'parse_page_structure', 'audit_delivery')


TOOLS += ('plan_marked_page','inspect_marked_region','review_numbered_text')


class VisualReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    passed: bool
    findings: list[dict] = Field(max_length=100)
    observations: list[dict] = Field(default_factory=list, max_length=100)
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
        request = json.loads((state.job/'request.json').read_text('utf8'))
        host = request.get('host_stage0')
        if host:
            from .host_evidence import AGENT_TOOLS, context, normalize_arguments
            if tool not in AGENT_TOOLS:
                raise ValueError('tool unavailable in the host-precomputed workflow')
            if tool in ('get_job_context','get_progress'):
                return context(state,request)
        if tool == 'list_evidence':
            limit=max(1,min(int(arguments.get('limit',30)),100))
            with state.db() as db:
                rows=db.execute('SELECT id,path,sha,role,delivered FROM evidence WHERE role LIKE ? ORDER BY rowid DESC LIMIT ?',('%'+arguments.get('role','')+'%',limit)).fetchall()
            return {'evidence':[{'id':r[0],'name':Path(r[1]).name,'sha256':r[2],'role':r[3],'provided_to_agent':bool(r[4])} for r in rows],
                    'note':'References are task-bound; use get_evidence to actually view contents.'}
        if tool == 'get_progress':
            current=state.read()
            pages=[]
            for index,page in enumerate(current['pages'],1):
                if not page:continue
                graph=self.graph(page)
                ir=json.loads(Path(page['ir']).read_text('utf8'))
                pages.append({'page':index,'draft_texts':len(graph.text.texts),'draft_assets':len(graph.asset.objects),
                    'committed_objects':len(graph.selected),'native_texts':sum(n['type']=='text' for n in ir['nodes']),
                    'native_shapes':sum(n['type']=='shape' for n in ir['nodes']),
                    'independent_images':sum(n['type']=='image' and n['id']!='source-plate' for n in ir['nodes']),
                    'region_crops':sum(1 for r in graph.selected.values() if r.get('region_crop')),
                    'movable_region_crops':sum(1 for r in graph.selected.values() if r.get('region_crop') and r.get('move_safety')=='MOVABLE')})
            with state.db() as db:
                recent=[json.loads(r[0]) for r in db.execute('SELECT payload FROM events ORDER BY id DESC LIMIT 12')]
            return {'version':current['version'],'pages':pages,'candidates':[{k:c.get(k) for k in ('id','page','object_id','status','failure')} for c in current['candidates'].values()],
                    'recent_events':recent,'remaining_seconds':state.remaining_seconds(),
                    'reconstruction':'NATIVE_OBJECTS_COMMITTED' if any(p['committed_objects'] for p in pages) else 'NO_EDITABLE_RECONSTRUCTION'}
        if tool == 'get_job_context':
            result = state.read()
            import re
            sandbox_configured=bool(shutil.which('docker') and re.fullmatch(r'[\w./:-]+@sha256:[a-f0-9]{64}',os.environ.get('IMAGE2PPT_SANDBOX_IMAGE','')))
            result['tool_availability']={'run_image_script':{'status':'CONFIGURED_NOT_PROBED' if sandbox_configured else 'UNAVAILABLE','reason':'requires Docker and a locally provisioned digest-pinned image'}}
            result['progress'] = self.call('get_progress',job_id)
            result['recent_evidence'] = self.call('list_evidence',job_id,limit=12)['evidence']
            result['sources'] = [state.register(p, 'source') for p in sorted((state.job/'inputs').glob('*.png'))]
            request=json.loads((state.job/'request.json').read_text('utf8'))
            shortcuts={'run_perception','run_parallel_pipeline','rebuild_region','repair_region',
                       'prepare_page_evidence','batch_reconstruct_page'} if request.get('tool_profile')=='basic' else set()
            if request.get('experiment_profile') in (None,'baseline'):
                shortcuts.update(('plan_marked_page','inspect_marked_region','review_numbered_text'))
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
            marked = str(arguments['page']) in state.read().get('experiment_marks',{})
            if marked and not arguments.get('full'):
                return {'version':state.read()['version'],'ownership_counts':dict(Counter(graph.ownership.values())),
                        'selected_ids':list(graph.selected),'perception_failures':page['failures']}
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
        if arguments.get('candidate_id') is not None and arguments['candidate_id'] not in state.read()['candidates']:
            raise ValueError('unknown candidate_id: region IDs and object IDs are not candidates. Use the candidate ID returned by create_candidate; get_progress lists actual candidates. No operation was reserved.')
        if tool=='fit_background' and not (all(arguments.get(k) is not None for k in ('page','object_id')) or all(arguments.get(k) is not None for k in ('evidence_id','bbox'))):
            raise ValueError('fit_background requires page+object_id OR evidence_id+bbox; no operation was reserved')
        method = getattr(self, tool)
        def execute(current,directory):
            if not host:
                return method(state,current,directory,**arguments)
            from .host_evidence import compact, questions, refresh, scoped_ids
            if tool=='prepare_page_evidence':
                return refresh(self,state,current,directory,arguments['page'])
            args = normalize_arguments(state,current,tool,arguments)
            if tool=='batch_reconstruct_page':
                from .review_targets import repair_inputs
                requested = scoped_ids(state,current,args['page'],arguments.get('block'),arguments.get('object_ids'))
                inputs = repair_inputs(current,args['page'],args['object_ids'])
            outcome = method(state,current,directory,**args)
            if tool=='batch_reconstruct_page':
                k = str(args['page'])
                old = current.setdefault('regional_failures',{}).get(k,[])
                attempt = {'inputs':inputs, 'candidate_id':outcome.get('candidate_id'), 'object_ids':args['object_ids'],
                           'skipped_unchanged_or_exhausted': sorted(set(requested)-set(args['object_ids'])),
                           'finding_ids':{oid:[f['finding_id'] for f in current.get('review_issues',{}).get(k,{}).get(oid,[])]
                                          for oid in args['object_ids']}, 'committed_ids':[],
                           'office_attempts':outcome.get('office_attempts'), 'status':outcome.get('status')}
                current.setdefault('repair_rounds', {}).setdefault(k, []).append(attempt)
                outcome['repair_attempt'] = attempt.copy()
                touched = set(args['object_ids'])
                current['regional_failures'][k] = [r for r in old if r['object_id'] not in touched]+outcome.get('failed_regions',[])
            if tool=='commit_candidate' and outcome.get('status')=='COMMITTED':
                page = current['candidates'][args['candidate_id']]['page']
                graph = self.graph(self.page(current,page))
                from .marked_routes import route_trace
                committed = set(outcome.get('object_ids', []))
                bindings = {r['mark_id']:set(r['selected_ids']) for r in
                            route_trace(current.get('marked_plans',{}).get(str(page),[]),graph)}
                for attempt in current.get('repair_rounds',{}).get(str(page),[]):
                    if attempt['candidate_id'] == args['candidate_id']:
                        attempt['committed_ids'] = [i for i in attempt['object_ids'] if i in committed
                                                    or bindings.get(i,set()) & committed]
                        attempt['status'] = 'COMMITTED'
                failures = current.get('regional_failures',{}).get(str(page),[])
                current['questions'][str(page)] = questions(state,current,page,failures,directory)
            if tool=='validate_office' and not args.get('candidate_id') and outcome.get('scorecard_audit'):
                from .review_targets import expose_review
                outcome['review_routing'] = expose_review(self,state,current,directory,outcome['scorecard_audit'])
            if tool in ('commit_candidate','batch_reconstruct_page') and current.get('repair_rounds'):
                write(directory/'repair-history.json',current['repair_rounds'])
                current['repair_history_evidence'] = state.register(directory/'repair-history.json','repair_history')['id']
            return compact(tool,outcome)
        return state.mutate(tool, arguments, expected_version, idempotency_key,execute)

    @staticmethod
    def page(current, page):
        if type(page) is not int or not 1 <= page <= len(current['pages']) or not current['pages'][page-1]:
            raise ValueError('unknown page')
        return current['pages'][page-1]

    @staticmethod
    def graph(page):
        return SceneGraph.model_validate_json(Path(page['scene']).read_text('utf8'))

    def prepare_page_evidence(self, state, current, directory, *, page, auto_create_scene=True,
                              run_layers=True, run_layout=None, run_structure=None):
        from .batch_backbone import prepare_page_evidence
        return prepare_page_evidence(self, state, current, directory, page=page,
                                     auto_create_scene=auto_create_scene, run_layers=run_layers,
                                     run_layout=run_layout, run_structure=run_structure)

    def parse_page_structure(self, state, current, directory, *, page, use_vlm=True):
        from .page_structure import parse_page_structure
        source_path = self.source(state, page)
        if str(page) not in current.get('ocr', {}):
            ocr_dir = directory / 'ocr'
            ocr_dir.mkdir(parents=True, exist_ok=True)
            self.run_ocr(state, current, ocr_dir, page=page)
        ocr = json.loads(Path(current['ocr'][str(page)]).read_text('utf8'))
        fonts = json.loads(Path(current['fonts']).read_text('utf8')) if 'fonts' in current else None
        layout_file = current.get('layout', {}).get(str(page))
        layout_data = json.loads(Path(layout_file).read_text('utf8')) if (layout_file and Path(layout_file).is_file()) else None
        info = parse_page_structure(state, source_path, ocr, layout=layout_data, fonts=fonts, use_vlm=bool(use_vlm))
        if info.get('applied_ocr_corrections'):
            proof_file = directory / 'ocr-proofread.json'
            write(proof_file, ocr)
            current.setdefault('ocr', {})[str(page)] = str(proof_file)
        out_file = directory / 'page-structure.json'
        write(out_file, info)
        current.setdefault('page_structure', {})[str(page)] = str(out_file)
        return info

    def review_numbered_text(self, state, current, directory, *, page, object_ids, explicit=False):
        from .text_reading import stage_readings
        if not (state.job/'host-baseline/manifest.json').is_file():
            raise ValueError('source reading requires a frozen baseline')
        return stage_readings(self, state, current, directory, page=page, object_ids=object_ids, explicit=explicit)

    def audit_delivery(self, state, current, directory, *, use_vlm=True):
        from .independent_review import run_delivery_audit
        return run_delivery_audit(self, state, current, directory, use_vlm=bool(use_vlm))

    def batch_reconstruct_page(self, state, current, directory, *, page, object_ids=None,
                               include_assets=True, include_containers=True, include_region_crops=False,
                               allow_move_unsafe=False, use_inpaint=True, verify_office=True, panel_regions=None):
        from .batch_backbone import batch_reconstruct_page
        return batch_reconstruct_page(self, state, current, directory, page=page, object_ids=object_ids,
                                      include_assets=include_assets, include_containers=include_containers,
                                      include_region_crops=include_region_crops,
                                      allow_move_unsafe=allow_move_unsafe,
                                      use_inpaint=use_inpaint, verify_office=verify_office, panel_regions=panel_regions)

    def plan_marked_page(self, state, current, directory, *, page, fields, append=False):
        from ..materialize import write
        path=current.get('experiment_marks',{}).get(str(page))
        if not path:
            raise ValueError('prepare numbered evidence for this page first')
        rows=json.loads(Path(path).read_text(encoding='utf8'))['rows']
        by_mark={r['mark']:r for r in rows}
        plans=[]
        for field in fields:
            row=by_mark[field['mark']]
            parents=[by_mark[n]['id'] for n in field['move_with']]
            plans.append({'id':row['id'],'bbox':row['bbox'],'role':field['role'],
                          **({'shape':field['shape']} if 'shape' in field else {}),
                          'move_with':parents,'route':field['route'],'source_text':row.get('text','')})
        previous=current.get('marked_plans',{}).get(str(page),[]) if append else []
        merged={r['id']:r for r in [*previous,*plans]}
        write(directory/'plan.json',list(merged.values()))
        current.setdefault('marked_plans',{})[str(page)]=list(merged.values())
        return {'page':page,'fields':plans,'planned_count':len(merged)}

    def inspect_marked_region(self, state, current, directory, *, page, bbox):
        from ..experiment import detect
        source=self.source(state,page)
        request=json.loads((state.job/'request.json').read_text(encoding='utf8'))
        profile=request.get('experiment_profile')
        if profile in (None,'baseline'):
            raise ValueError('local numbered inspection requires a marked profile')
        from PIL import Image
        with Image.open(source) as im:
            x,y,w,h=bbox
            if min(x,y)<0 or min(w,h)<=0 or x+w>im.width or y+h>im.height:
                raise ValueError('invalid local inspection window')
        data=detect(source,profile,directory/'models',bbox)
        inventory=Path(current['experiment_marks'][str(page)])
        old=json.loads(inventory.read_text(encoding='utf8'))
        previous={r['id']:r for r in old['rows']}
        next_mark=max((r['mark'] for r in previous.values()),default=0)+1
        for row in data['regions']:
            if row['id'] not in previous:
                row['mark']=next_mark
                next_mark+=1
                previous[row['id']]=row
        old['rows']=list(previous.values())
        write(directory/'marks.json',old)
        current['experiment_marks'][str(page)]=str(directory/'marks.json')
        return {'detected':[previous[r['id']] for r in data['regions']], 'models':data['models']}

    def crop_region(self, state, current, directory, *, page, object_ids=None, bbox=None,
                    allow_movable=True, allow_move_unsafe=False, upgrade_containers=True):
        from .. import layout_evidence
        from ..recovery.region_crop import collect_crop_seeds, stage_region_crops
        row = self.page(current, page)
        graph = self.graph(row)
        logo_ids = {
            x['id'] for x in current.get('scene_relations', {}).get(str(page), {}).get('logo_lockup_candidates', [])
            if x.get('id')
        }
        raw_unrec = current.get('batch_unrecoverable_texts', {}).get(str(page))
        unrec_set = set(raw_unrec) if raw_unrec is not None else None
        protected_text_ids: set[str] | None = None
        layout_report: list[dict] = []

        if object_ids is None and bbox is None:
            layout_data = None
            layout_file = current.get('layout', {}).get(str(page))
            if layout_file and Path(layout_file).is_file():
                layout_data = json.loads(Path(layout_file).read_text('utf8'))
            elif layout_evidence.available():
                try:
                    layout_data = layout_evidence.detect(
                        {'path': row['snapshot']['path'], 'source_sha256': graph.source_sha256},
                        directory / 'layout-cache',
                    )
                except Exception as exc:
                    layout_report.append({'id': 'layout_evidence', 'state': 'ERROR', 'reason': f'{type(exc).__name__}: {exc}'})
            seeds, protected_text_ids = collect_crop_seeds(
                graph,
                unrecoverable_text_ids=unrec_set,
                layout=layout_data,
                logo_ids=logo_ids,
            )
            if not seeds:
                return {'status': 'NO_CROPS_STAGED', 'page': page, 'report': layout_report}
        else:
            seeds = []
            if bbox is not None:
                seeds.append({'id': 'manual-crop', 'kind': 'manual', 'bbox': [float(v) for v in bbox], 'text': ''})
            wanted = set(object_ids or [])
            for t in graph.text.texts:
                if t.id in wanted and t.id not in graph.selected:
                    seeds.append({'id': t.id, 'kind': 'text', 'bbox': list(t.bbox), 'text': t.text})
            for a in graph.asset.objects:
                if a.id in wanted and a.id not in graph.selected:
                    seeds.append({'id': a.id, 'kind': 'asset', 'bbox': list(a.bbox), 'text': a.intrinsic_text})
            if not seeds:
                raise ValueError('no uncommitted matching objects or bbox for region crop')
            _, base_protected = collect_crop_seeds(graph, unrecoverable_text_ids=unrec_set, logo_ids=logo_ids)
            protected_text_ids = base_protected - wanted

        scoped = json.loads((state.job/'request.json').read_text('utf8')).get('object_scoped_delivery', False)
        if scoped and bbox is not None:
            raise ValueError('object fallback requires a source object ID')
        if scoped:
            ordinary = {t.id for t in graph.text.texts if t.id not in graph.selected and t.id not in logo_ids}
            seeds = [s for s in seeds if s['id'] not in ordinary]
            protected_text_ids = (protected_text_ids or set()) | ordinary
            if not seeds:
                return {'status':'NO_CROPS_STAGED', 'page':page, 'report':layout_report}
        crop_ir, staged_ids, report = stage_region_crops(
            Path(row['ir']).parent, directory / 'staged',
            seeds=seeds, allow_movable=allow_movable, allow_move_unsafe=allow_move_unsafe and not scoped, single_object_only=scoped,
            protected_text_ids=protected_text_ids,
        )
        full_report = [*layout_report, *report]
        if not staged_ids:
            return {'status': 'NO_CROPS_STAGED', 'page': page, 'report': full_report}
        if upgrade_containers and object_ids is None and bbox is None:
            from ..recovery.container_stage import stage_containers_and_surfaces
            try:
                cont_ir, cont_ids = stage_containers_and_surfaces(
                    crop_ir.parent, directory / 'post-crop-containers', logo_ids=logo_ids, stable_ids=scoped
                )
                if cont_ids:
                    crop_ir = cont_ir
                    staged_ids = [*staged_ids, *cont_ids]
            except Exception as exc:
                full_report.append({'id': 'post_crop_containers', 'state': 'SKIPPED', 'reason': str(exc)})
        return self._stage_asset_candidate(state, current, directory, page, crop_ir, staged_ids,
                                           'region_crop', full_report)

    def stage_photo_frame(self, state, current, directory, page, frame):
        from ..recovery.marked_surfaces import stage_marked_surfaces
        from .batch_backbone import committed_graphic_evidence
        row = self.page(current, page)
        request = json.loads((state.job/'request.json').read_text('utf8'))
        verified = committed_graphic_evidence(current, row, page, self.graph(row)) if request.get('object_scoped_delivery') else {}
        ir, ids = stage_marked_surfaces(Path(row['ir']).parent, directory/'staged', [frame], set(),
                                        verified_assets=verified)
        if not ids:
            response = {'status': 'NO_FRAME_STAGED', 'page': page}
            if frame.get('required_card_frames'):
                response['required_card_frames'] = frame['required_card_frames']
            return response
        result = self._stage_asset_candidate(state, current, directory, page, ir, ids, 'photo_frame', [])
        if request.get('object_scoped_delivery'):
            current['candidates'][result['candidate_id']]['card_repair_targets'] = [frame['id']]
        return result

    def _stage_asset_candidate(self, state, current, directory, page, ir, ids, strategy, report):
        row = self.page(current, page)
        cid = 'c-' + key([str(directory.relative_to(state.job)), strategy])[:24]
        current['candidates'][cid] = {
            'id': cid, 'page': page, 'object_id': ids[0], 'object_ids': ids,
            'kind': 'batch', 'status': 'STAGED_LOCAL', 'ir': str(ir),
            'parent_ir_sha256': digest(row['ir']), 'parent_scene_sha256': digest(row['scene']),
            'strategy': strategy, 'code': self.code_binding(),
            'isolation_allowed': strategy != 'photo_frame',
            'fingerprint': key({'objects': ids, 'ir': digest(ir)}),
            'files': {str(p): digest(p) for p in ir.parent.rglob('*') if p.is_file()},
        }
        return {'status': 'STAGED_LOCAL', 'candidate_id': cid, 'page': page,
                'staged_object_ids': ids, 'report': report,
                'next_step': f'Call commit_candidate(candidate_id="{cid}", expected_version={current["version"] + 1}) to validate and commit.'}

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
                if chosen not in ('verified-backing', 'source-halo', 'sam', 'rmbg2'):
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

    def _verify_staged_candidate_graph(self, current, candidate):
        self.check_candidate(current, candidate)
        ir = Path(candidate['ir'])
        graph = SceneGraph.model_validate_json((ir.parent / 'scene.json').read_text('utf8'))
        row = self.page(current, candidate['page'])
        original = self.graph(row)
        from .text_reading import authorize_readings
        reading = None
        if candidate.get('kind') == 'text_reading':
            ref = candidate['reading_receipt']
            if candidate['files'].get(ref) != digest(ref):
                raise RuntimeConflict('reading receipt changed')
            reading = json.loads(Path(ref).read_text('utf8'))
        authorize_readings(original, graph, reading)
        if graph.source_sha256 != original.source_sha256:
            raise ValueError('source changed')
        dropped = set(original.selected) - graph.selected.keys()
        if dropped:
            import numpy as np
            from ..recovery.region_artwork import verify_text_cutout
            for oid in set(graph.selected) - original.selected.keys():
                field = graph.selected[oid]
                for aid, evidence in field.get('text_cutouts', {}).items():
                    if aid not in dropped or field.get('kind') != 'text' or evidence.get('field') != oid:
                        raise ValueError('unexpected source cutout replacement')
                    proof = field['owner_evidence']
                    file = (ir.parent/proof['file']).resolve()
                    if not file.is_relative_to(ir.parent.resolve()) or digest(file) != proof['sha256']:
                        raise RuntimeConflict('text cutout replacement mask changed')
                    with np.load(file, allow_pickle=False) as ink:
                        verify_text_cutout(Path(row['ir']).parent, original.selected[aid], evidence,
                            field['payload']['text'], ink['mask'], list(ink['restore_bbox']), ink['backing_patch'])
                    dropped.remove(aid)
        if dropped:
            raise ValueError('candidate dropped previously committed objects')
        for oid in set(graph.selected) - set(original.selected):
            if graph.ownership.get(oid) != 'RESOLVED':
                raise ValueError(f'cannot commit unresolved object {oid}')
            rec = graph.selected[oid]
            ref = rec.get('owner_evidence')
            if not ref:
                if rec.get('region_crop') and rec.get('move_safety') == 'MOVE_UNSAFE':
                    aref = rec.get('asset_reference') or {}
                    afile = (ir.parent / aref.get('path', '')).resolve() if aref.get('path') else None
                    if not afile or not afile.is_relative_to(ir.parent.resolve()) or digest(afile) != aref.get('sha256'):
                        raise RuntimeConflict(f'region crop asset binding mismatch for {oid}')
                    continue
                raise ValueError(f'missing owner evidence for {oid}')
            pfile = (ir.parent / ref['file']).resolve()
            if not pfile.is_relative_to(ir.parent.resolve()) or digest(pfile) != ref['sha256']:
                raise RuntimeConflict('candidate owner proof changed')
        rb = graph.raster_backing or {}
        if rb.get('path') and rb.get('sha256'):
            plate_file = (ir.parent / rb['path']).resolve()
            if not plate_file.is_relative_to(ir.parent.resolve()) or digest(plate_file) != rb['sha256']:
                raise RuntimeConflict('candidate raster backing changed')
        return ir, graph, row, original

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
        ir, graph, _, _ = self._verify_staged_candidate_graph(current, candidate)
        from .compiler import compile_local as compile_pages
        from .card_edit_preservation import previous_edits_lost
        from ..parallel_pipeline import effective_texts
        def reject_regression(issues, attempts):
            result = {'status':'REVIEW_REQUIRED','failure':'PREVIOUS_NATIVE_EDIT_LOST',
                      'office_status':'NOT_RUN_FOR_REGRESSING_CANDIDATE','office_attempts':attempts,
                      'failed_regions':issues, 'card_repair_targets':candidate['card_repair_targets'],
                      'parent_ir_sha256':candidate['parent_ir_sha256']}
            candidate.update(status='REVIEW_REQUIRED', validation=result)
            return {'candidate_id':candidate_id, **result}
        issues = previous_edits_lost(state, current, candidate, ir)
        if issues:
            return reject_regression(issues, 0)
        with lease('compiler'):
            artifacts = compile_pages([ir], directory/'compiled', [effective_texts(graph, ir)])
        result = self.office_check(state, artifacts['pptx'], [ir], directory/'office-check')
        attempts = result.get('office_attempts', 0)
        if candidate.get('kind') == 'batch' and candidate.get('isolation_allowed', candidate.get('strategy') != 'photo_frame'):
            from .batch_backbone import restage_batch_candidate_subset
            for attempt in range(1, 3):
                if result.get('status') != 'REVIEW_REQUIRED':
                    break
                if not restage_batch_candidate_subset(self, state, current, directory/f'isolate-{attempt}', candidate, result):
                    break
                ir, graph, _, _ = self._verify_staged_candidate_graph(current, candidate)
                issues = previous_edits_lost(state, current, candidate, ir)
                if issues:
                    return reject_regression(issues, attempts)
                with lease('compiler'):
                    artifacts = compile_pages([ir], directory/f'compiled-repair-{attempt}', [effective_texts(graph, ir)])
                result = self.office_check(state, artifacts['pptx'], [ir], directory/f'office-repair-{attempt}')
                attempts += result.get('office_attempts', 0)
        result = {**result, 'office_attempts': attempts}
        if candidate.get('kind')=='text':
            from .office import frame_feedback
            feedback=frame_feedback(result,graph,candidate['object_id'])
            if feedback:
                result['repair_feedback']=feedback
        candidate['validation'] = result
        if result.get('status') == 'PASS':
            candidate['status'] = 'VALIDATED'
            candidate['validated_bundle'] = {
                'irs': [digest(ir)],
                'pptx': str(artifacts['pptx']),
                'pptx_sha256': digest(artifacts['pptx']),
            }
        elif result.get('status') == 'UNKNOWN':
            candidate['status'] = 'STAGED_LOCAL'
        else:
            candidate['status'] = 'REVIEW_REQUIRED'
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
            # A formal check of an existing proposal is not a second candidate route.
            if staged.get('id') in current['candidates']:
                current['candidates'][staged['id']]['submission_of'] = candidate_id
            if staged.get('status') != 'STAGED_LOCAL':
                return {'status': 'SUBMISSION_BLOCKED', 'checks': staged, 'preview_candidate_id': candidate_id}
            checked = self.validate_candidate(state, current, directory/'submission-check', candidate_id=staged['id'])
            if checked.get('status') != 'PASS':
                return {'status': 'SUBMISSION_BLOCKED', 'checks': checked, 'preview_candidate_id': candidate_id}
            return self.commit_region(state, current, directory/'commit', candidate_id=staged['id'])
        if candidate.get('status') == 'COMMITTED':
            obj_ids = candidate.get('object_ids') or [candidate.get('object_id')]
            return {'status': 'COMMITTED', 'already_committed': True, 'candidate_id': candidate_id,
                    'object_id': candidate.get('object_id'), 'object_ids': obj_ids, 'committed_count': len(obj_ids)}
        if candidate.get('status') == 'STAGED_LOCAL':
            self.validate_candidate(state, current, directory/'auto-validate', candidate_id=candidate_id)
        if candidate.get('validation', {}).get('failure') == 'PREVIOUS_NATIVE_EDIT_LOST':
            return {'candidate_id':candidate_id, 'committed_count':0, **candidate['validation']}
        if candidate.get('status') != 'VALIDATED' or candidate.get('validation', {}).get('status') != 'PASS':
            raise ValueError('candidate requires successful hard and actual edit checks')
        ir, _, row, _ = self._verify_staged_candidate_graph(current, candidate)
        for path, sha in candidate['validation'].get('files', {}).items():
            if digest(path) != sha:
                raise RuntimeConflict('candidate Office evidence changed')
        if candidate.get('kind') == 'text_reading':
            k = str(candidate['page'])
            ocr = json.loads(Path(current['ocr'][k]).read_text('utf8'))
            lines = {ln['id']:ln for ln in ocr['lines']}
            receipt = json.loads(Path(candidate['reading_receipt']).read_text('utf8'))
            from .reading_targets import commit_readings
            commit_readings(lines,receipt['changes'])
            write(directory/'verified-ocr.json', ocr)
            current['ocr'][k] = str(directory/'verified-ocr.json')
        row.update(ir=str(ir), scene=str(ir.parent/'scene.json'))
        candidate['status'] = 'COMMITTED'
        if candidate.get('validated_bundle'):
            current['last_verified_bundle'] = candidate['validated_bundle']
        else:
            current.pop('last_verified_bundle', None)
        current.pop('composition', None)
        current.pop('visual_review', None)
        obj_ids = candidate.get('object_ids') or [candidate['object_id']]
        return {'status': 'COMMITTED', 'candidate_id': candidate_id, 'object_id': candidate['object_id'],
                'object_ids': obj_ids, 'committed_count': len(obj_ids)}

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
        ir_digests = [digest(ir) for ir in irs]
        verified_bundle = current.get('last_verified_bundle')
        if (verified_bundle and verified_bundle.get('irs') == ir_digests
                and Path(verified_bundle.get('pptx', '')).is_file()
                and digest(verified_bundle['pptx']) == verified_bundle.get('pptx_sha256')):
            out_dir = directory/'compiled'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_pptx = out_dir/'result.pptx'
            shutil.copyfile(verified_bundle['pptx'], out_pptx)
            artifacts = {'pptx': str(out_pptx)}
        else:
            with lease('compiler'):
                artifacts = compile_pages(irs, directory/'compiled',
                                          [effective_texts(self.graph(p), ir) for p, ir in zip(current['pages'], irs)])
        binding = {str(p): digest(p) for p in [Path(artifacts['pptx']), *irs]}
        if current.get('marked_plans'):
            from xml.etree import ElementTree
            from zipfile import ZipFile
            from .marked_routes import bind_plan,route_trace
            traces=[]
            with ZipFile(artifacts['pptx']) as deck:
                for i,page in enumerate(current['pages'],1):
                    if str(i) not in current['marked_plans']:
                        continue
                    graph=self.graph(page)
                    plans=bind_plan(current['marked_plans'][str(i)],current['experiment_marks'][str(i)],graph,page['snapshot']['path'])
                    slide=ElementTree.fromstring(deck.read(f'ppt/slides/slide{i}.xml'))
                    names={n.get('name') for n in slide.iter('{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr')}
                    fields=route_trace(plans,graph)
                    for f in fields:
                        f['pptx_ids']=sorted(names.intersection(f['selected_ids']))
                    traces.append({'page':i,'fields':fields})
            write(directory/'delivery-route-trace.json',{'pptx_sha256':digest(artifacts['pptx']),'pages':traces})
            binding[str(directory/'delivery-route-trace.json')]=digest(directory/'delivery-route-trace.json')
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
        request=json.loads((state.job/'request.json').read_text('utf8'))
        options = {'check_owned_groups':True} if request.get('object_scoped_delivery') else {}
        identity={'pptx':digest(pptx),'irs':[digest(ir) for ir in irs],
                  'implementation':RuntimeService.code_binding(),
                  'mode':os.environ.get('IMAGE2PPT_OFFICE_MODE','local')}
        identity.update(options)
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
            return {**result,'cache_hit':True,'office_attempts':0}
        budget=AdaptiveBudget(Store(state.job/'state'),request.get('adaptive_budget'))
        budget.reserve('office_validations',identity)
        write(receipt,{'identity':identity,'status':'UNKNOWN','operation':str(directory)})
        provisional = receipt
        if os.environ.get('IMAGE2PPT_OFFICE_MODE', 'local') == 'remote':
            from .office_bridge import submit
            result=submit(state.job, pptx, irs, directory)
            result['office_attempts'] = 1
            if result.get('failure') == 'QA_INFRASTRUCTURE_UNAVAILABLE' and result.get('status') == 'UNKNOWN':
                from ..parallel_store import BudgetExceeded
                remaining = state.remaining_seconds()
                try:
                    retry = (remaining is None or remaining > 0) and budget.reserve(
                        'office_validations', {'retry': identity})
                except BudgetExceeded:
                    retry = False
                if retry:
                    first_reason = result.get('reason')
                    result = submit(state.job, pptx, irs, Path(directory)/'retry',
                                    timeout=min(1800, remaining) if remaining else 1800)
                    result.update(office_attempts=2, retry_reason=first_reason)
            actual = result.get('worker_binding')
            if actual and identity['worker'].get('binding') != actual:
                # A different worker may claim the ticket after the last poll.
                # Bind the completed receipt to the worker that did the check.
                identity = {**identity, 'worker':{'binding':actual}}
                receipt = state.root/'office-cache'/(key(identity)+'.json')
        else:
            from .office import check_local
            result=check_local(pptx, irs, directory, **options)
            result['office_attempts'] = 1
        write(receipt,{'identity':identity,'operation':str(directory),'result':result})
        if receipt != provisional:
            provisional.unlink(missing_ok=True)
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
            canonical_src = self.source(state, i)
            src_path = canonical_src if digest(canonical_src) == page['snapshot']['source_sha256'] else page['snapshot']['path']
            required.append(state.register(src_path, 'source'))
            required.append(state.register(preview, 'office_preview'))
            images, _ = panels(page['snapshot']['path'], preview, page['ir'],
                               result['office_directory'], i, directory/f'panels-{i:02d}')
            required.extend(state.register(path, 'edit_panel') for path in images)
        required.append(state.register(directory/'checks.json', 'hard_checks'))
        composition['required_evidence'] = required
        current.pop('visual_review', None)
        current.pop('independent_audit', None)
        from .independent_review import run_delivery_audit
        incremental = (state.job/'host-baseline/manifest.json').is_file()
        det_audit = run_delivery_audit(self, state, current, directory/'scorecard-audit', use_vlm=incremental)
        return {**result, 'required_evidence': required, 'scorecard_audit': det_audit}

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
            raise ValueError('checked must include all six categories. findings means unresolved defects and requires passed=false; positive notes belong in observations, accepted differences in minor_deviations.')
        composition = self.composition(current)
        if composition['office']['status'] != 'PASS':
            raise ValueError('actual editing and residual evidence required')
        supplied_refs = {identifier: state.evidence(identifier)[1] for identifier in review.evidence_ids}
        unseen = [identifier for identifier, ref in supplied_refs.items() if not ref['provided_to_agent']]
        if unseen:
            raise ValueError('evidence has not been provided to Agent: '+', '.join(unseen)+
                             '; use get_evidence to receive the actual content before review')
        supplied_role_shas = {(ref['role'], ref['sha256']) for ref in supplied_refs.values()}
        missing = [r['id'] for r in composition['required_evidence']
                   if r['id'] not in supplied_refs and (r['role'], r['sha256']) not in supplied_role_shas]
        if missing:
            raise ValueError('review omitted required source, preview or edit evidence: '+', '.join(missing))
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
        raw_selected = [self.graph(page).selected for page in current['pages']]
        editable = [
            {oid: rec for oid, rec in sel.items()
             if not (rec.get('region_crop') and rec.get('move_safety') == 'MOVE_UNSAFE')}
            for sel in raw_selected
        ]
        in_place_crops = [
            sorted(oid for oid, rec in sel.items()
                   if rec.get('region_crop') and rec.get('move_safety') == 'MOVE_UNSAFE')
            for sel in raw_selected
        ]
        # Completeness has no independent denominator in this first integration.
        status = 'PARTIAL_EDITABLE' if any(editable) else 'RASTER_ONLY'
        host = (state.job/'host-baseline/manifest.json').is_file()
        delivery = state.root/'agent-delivery' if host else state.job/'delivery'
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
        # Ensure independent scorecard & optional stateless VLM audit match exact composition
        from .independent_review import run_delivery_audit
        from ..tolerance_config import load_tolerance_config
        audit = current.get('independent_audit')
        has_vlm_creds = bool(
            os.environ.get('IMAGE2PPT_AGENT_API_KEY')
            or os.environ.get('OPENCODE_GO_API_KEY')
            or os.environ.get('MOMA_API_KEY')
        )
        if not audit or audit.get('composition_sha256') != key(composition):
            audit = run_delivery_audit(self, state, current, directory/'final-audit', use_vlm=has_vlm_creds)
        elif has_vlm_creds and all(p.get('vlm_audit', {}).get('status') == 'SKIPPED' for p in audit.get('pages', [])):
            audit = run_delivery_audit(self, state, current, directory/'final-audit', use_vlm=True)

        recalls = []
        for p_audit in audit.get('pages', []):
            pidx = p_audit['page']
            sc_src = Path(p_audit['scorecard_file'])
            ow_src = Path(p_audit['owner_preview_file'])
            if sc_src.is_file():
                shutil.copyfile(sc_src, delivery/('scorecard.json' if len(current['pages']) == 1 else f'scorecard-{pidx:02d}.json'))
            if ow_src.is_file():
                shutil.copyfile(ow_src, delivery/('owner-preview.png' if len(current['pages']) == 1 else f'owner-preview-{pidx:02d}.png'))
            if p_audit.get('inventory', {}).get('recall') is not None:
                recalls.append(p_audit['inventory']['recall'])

        tol_cfg = load_tolerance_config()
        sc_gate = tol_cfg.get('scorecard_gate', {})
        max_plate = float(sc_gate.get('max_succeeded_raster_plate_content_fraction', 0.85))
        plate_exceeded = any(
            float(p.get('summary', {}).get('raster_plate_content_fraction', 0.0)) > max_plate
            for p in audit.get('pages', [])
        )
        audit_ok = audit.get('passed') is True and not plate_exceeded
        delivery_state = 'SUCCEEDED' if (any(editable) and audit_ok) else 'REVIEW_REQUIRED'
        avg_recall = round(sum(recalls) / len(recalls), 4) if recalls else None
        result = {'status': delivery_state, 'completion': status, 'reconstruction_succeeded': any(editable), 'visual_review_backend': 'opencode',
                  'office_pass': True, 'visual_review': visual, 'independent_audit': audit,
                  'tolerance_profile': tol_cfg.get('profile'),
                  'pages': len(editable),
                  'editable_objects': [sorted(page) for page in editable],
                  'in_place_region_crops': in_place_crops,
                  'native_objects': [sorted(k for k,v in sel.items() if v.get('kind') in ('text','surface','shape','table','chart')) for sel in editable],
                  'movable_assets': [sorted(k for k,v in sel.items() if v.get('kind')=='asset') for sel in editable],
                  'unresolved_objects': [[identifier for identifier in self.graph(page).ownership
                                          if identifier not in chosen]
                                         for page, chosen in zip(current['pages'], raw_selected)],
                  'recall': avg_recall, 'human_acceptance': False, 'automatic_publication': False,
                  'workflow_states': {'candidate_generation': 'COMPLETED', 'regional_commit': 'COMPLETED',
                                      'final_qa': 'PASS' if audit_ok else 'REVIEW_REQUIRED',
                                      'automatic_publication': 'GATED'}}
        write(delivery/'summary.json', result)
        result['artifacts'] = {p.name: digest(p) for p in delivery.iterdir() if p.is_file()}
        write(state.root/'agent-result.json' if host else state.job/'result.json', result)
        current['status'] = delivery_state
        return result
