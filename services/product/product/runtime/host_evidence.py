"""Page-isolated perception and numbered decisions shared by the host and the Agent."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..materialize import digest, write
from .marked_routes import iou


def read(file):
    return json.loads(Path(file).read_text(encoding='utf8'))


def prepare_all(service, state, current, directory, *, sources):
    service.list_fonts(state, current, directory / 'fonts')

    def prepare(page):
        work = directory / f'p{page}'
        work.mkdir()
        local = {'pages': [], 'fonts': current['fonts'], 'version': current['version']}
        result = service.prepare_page_evidence(state, local, work, page=page)
        inventory = read(local['experiment_marks'][str(page)])
        if inventory['source_sha256'] != sources[page-1]:
            raise ValueError('preflight inventory source mismatch')
        return page, local, result

    with ThreadPoolExecutor(max_workers=min(2, len(sources))) as pool:
        for page, local, result in pool.map(prepare, range(1, len(sources)+1)):
            k = str(page)
            current['pages'].append(local['pages'][page-1])
            for name in ('ocr','layers','layout','scene_relations','deferred_assets','experiment_marks','page_structure'):
                if k in local.get(name, {}):
                    current.setdefault(name, {})[k] = local[name][k]
            current.setdefault('host_evidence', {})[k] = {
                'source_sha256': sources[page-1], 'blocks': result.get('blocks', []),
                'inventory_sha256': digest(current['experiment_marks'][k]),
                'ocr_status':read(local['ocr'][k]).get('status','READY'),
                'layerd_status': result['layerd_status'], 'layout_status': result['layout_status'],
            }
            rows = read(current['experiment_marks'][k])['rows']
            fields = deterministic_plan(rows, current['scene_relations'][k])
            current.setdefault('marked_plans', {})[k] = fields
    return {'sources': sources, 'pages': len(sources)}


def deterministic_plan(rows, relations):
    """Only assign backed OCR, detected table/image and logo routes. Other marks remain proposals."""
    logos = {r['id'] for r in relations.get('logo_lockup_candidates', [])}
    result = []
    for r in rows:
        label = r['label']
        if r['id'] in logos:
            route, role = 'preserve', 'brand'
        elif r.get('ocr_ids') and label == 'text':
            route, role = 'text', 'field'
        elif label == 'table':
            route, role = 'table', 'table'
        elif label == 'image':
            route, role = 'asset', 'image'
        else:
            continue
        result.append({'id':r['id'], 'bbox':r['bbox'], 'role':role, 'route':route,
                       'source_text':r.get('text',''), 'origin':'host', 'move_with':[]})
    return result


def questions(state, current, page, failures, directory):
    """Persist bounded block inventories; send references and counts in the initial context."""
    k = str(page)
    inventory = read(current['experiment_marks'][k])
    rows = inventory['rows']
    by_id = {r['id']: r for r in rows}
    uncertain = {r['id'] for r in read(current['ocr'][k])['lines'] if r.get('score', 1) < .94 and not r.get('proofread_verified')}
    choices = {r['id']:r['route'] for r in current['marked_plans'][k]}
    graph = read(current['pages'][page-1]['scene'])
    selected = graph['selected']
    owners = set(selected)
    for obj in selected.values():
        owners.update(obj.get('seed_ids', []))
        owners.add(obj.get('mark_id'))
    failed = {f['object_id']: f.get('diagnostic_category') or f.get('failure','unverified') for f in failures}
    detected = [r for r in rows if r.get('sources')]
    children = {r['parent'] for r in rows if r.get('parent')}
    blocks = []
    for index, block in enumerate(current['host_evidence'][k]['blocks'], 1):
        info, _ = state.evidence(block['inventory']['id'])
        items = []
        for entry in read(info):
            r = by_id[entry['id']]
            oid = r['id']
            reasons = []
            if oid in failed and oid not in owners:
                reasons.append(failed[oid])
            if r['label'] == 'uncertain_ocr' or (r['label']=='text' and uncertain.intersection(r.get('ocr_ids', []))):
                reasons.append('low_confidence_ocr')
            elif oid not in choices and oid not in owners:
                reasons.append('undecided_route')
            if oid in children and r['label'] == 'icon' and r.get('intersecting_ocr_ids'):
                reasons.append('possible_card_or_group')
            conflicts = [s['mark'] for s in detected if r.get('sources') and
                         s['label']!=r['label'] and iou(r['bbox'],s['bbox'])>=.70]
            if conflicts:
                reasons.append('detector_conflict')
            if reasons:
                items.append({'mark':r['mark'], 'id':oid, 'label':r['label'], 'issues':reasons,
                              'route':choices.get(oid), 'committed':oid in owners,
                              'parent_mark':by_id[r['parent']]['mark'] if r.get('parent') else None,
                              'conflicting_marks':conflicts[:12],'source_text':r.get('text','')[:120]})
        if items:
            file = directory / f'questions-{page}-{index}.json'
            write(file, {'source_sha256':inventory['source_sha256'], 'page':page, 'block':index, 'items':items})
            blocks.append({'block':index, 'count':len(items),
                           'issues': dict(Counter(tag for it in items for tag in it['issues'])),
                           'marks':[it['mark'] for it in items], 'image':block['image']['id'],
                           'questions':state.register(file,'numbered_questions')['id']})
    return blocks


AGENT_TOOLS = {'review_numbered_text','get_job_context','get_progress','get_evidence','prepare_page_evidence',
               'inspect_scene','plan_marked_page','batch_reconstruct_page','crop_region',
               'commit_candidate','compile_pptx','validate_office','record_visual_review','finalize_job'}


def scoped_ids(state, current, page, block, object_ids):
    k = str(page)
    allowed = set()
    matching = [b for b in current['questions'][k] if block is None or b['block']==block]
    for b in matching:
        f, _ = state.evidence(b['questions'])
        allowed.update(it['id'] for it in read(f)['items'])
    ids = set(object_ids) if object_ids is not None else allowed
    if block is None and object_ids is None:
        raise ValueError('choose a question block or issue object IDs from this page')
    if not 1<=len(ids)<=30:
        raise ValueError(f'choose 1 to 30 issue object IDs from this page (got {len(ids)})')
    if not ids <= allowed:
        invalid = sorted(ids - allowed)
        raise ValueError(
            f'choose issue object IDs from this page only; non-issue or already-committed IDs={invalid[:12]}'
        )
    return sorted(ids)


ROLES = ['field','title','label','caption','body','number','project','industry','table','card',
         'role_box','screenshot','dashboard','image','icon','decoration','brand','group','other']


def normalize_arguments(state,current,tool,arguments):
    args = dict(arguments)
    if tool=='review_numbered_text':
        ids = scoped_ids(state, current, args['page'], None, args['object_ids'])
        return {'page':args['page'], 'object_ids':ids}
    if tool=='batch_reconstruct_page':
        page = str(args['page'])
        ids = scoped_ids(state,current,args['page'],args.get('block'),args.get('object_ids'))
        rounds = current.setdefault('repair_rounds', {}).get(page, [])
        if len(rounds) >= 4 or any(r.get('object_ids') == ids and not r.get('progress') for r in rounds):
            raise ValueError('local repair limit reached; review the current delivery')
        return {'page':args['page'],'object_ids':ids,'verify_office':True,'include_region_crops':False}
    if tool=='crop_region':
        out = {'page': int(args['page']), 'allow_movable': True,
               'allow_move_unsafe': bool(args.get('allow_move_unsafe', False)),
               'upgrade_containers': bool(args.get('upgrade_containers', True))}
        if args.get('object_ids') is not None:
            oids = [str(x) for x in args['object_ids']]
            if not 1 <= len(oids) <= 30:
                raise ValueError('select 1 to 30 object_ids for crop_region')
            out['object_ids'] = oids
        if args.get('bbox') is not None:
            out['bbox'] = [float(v) for v in args['bbox']]
        return out
    if tool=='plan_marked_page':
        page = args['page']
        fields = args['fields']
        rows = read(current['experiment_marks'][str(page)])['rows']
        by_mark = {r['mark']:r['id'] for r in rows}
        id_to_mark = {r['id']:r['mark'] for r in rows}
        if not fields or len(fields)>30:
            raise ValueError('select 1 to 30 numbered questions')
        for f in fields:
            if (set(f)!={'mark','move_with','role','route'} or f['role'] not in ROLES or
                    f['route'] not in ('text','asset','surface','table','preserve') or
                    any(n not in by_mark for n in [f['mark'],*f['move_with']])):
                raise ValueError('use only numbered marks, existing partners, a role and a route')
        try:
            scoped_ids(state,current,page,None,[by_mark[f['mark']] for f in fields])
        except ValueError as exc:
            allowed_marks = []
            for b in current['questions'][str(page)]:
                qf, _ = state.evidence(b['questions'])
                for it in read(qf)['items']:
                    m_val = it['mark'] if 'mark' in it else id_to_mark.get(it.get('id'))
                    if m_val is not None:
                        allowed_marks.append(m_val)
            bad_marks = [f['mark'] for f in fields if f['mark'] not in set(allowed_marks)]
            if bad_marks:
                bad_desc = [f"mark {m} ({by_mark.get(m)})" for m in bad_marks[:12]]
                raise ValueError(f"{exc}; invalid marks={bad_desc}; valid_issue_marks={sorted(set(allowed_marks))[:30]}") from None
            raise
        return {'page':page,'fields':fields,'append':True}
    return args


def compact(tool,result):
    if tool=='plan_marked_page':
        return {'page':result['page'],'planned_count':result['planned_count'],
                'object_ids':[r['id'] for r in result['fields']]}
    if tool=='batch_reconstruct_page':
        return {**{k:result[k] for k in ('page','status','candidate_id','failed_count','staged_count') if k in result},
                'failures':dict(Counter(r.get('diagnostic_category','UNKNOWN') for r in result.get('failed_regions',[])))}
    if tool=='crop_region':
        return {**{k:result[k] for k in ('page','status','candidate_id','object_id','object_ids') if k in result},
                'report':result.get('report', [])[:20]}
    if tool=='validate_office':
        audit = result.get('scorecard_audit', {})
        return {**{k:result[k] for k in ('status','required_evidence','failure','pptx_sha256') if k in result},
                'audit_passed': audit.get('passed'), 'defects': audit.get('critical_defects',[])[:30],
                'missing_objects': [{'page':p['page'],'items':p['summary'].get('missing_inventory_items',[])[:30]}
                                    for p in audit.get('pages',[])]}
    if tool in ('finalize_job','commit_candidate'):
        return {k:result[k] for k in ('status','reason','candidate_id','committed_count','office_pass','completion') if k in result}
    return result


def context(state,request):
    from .host_baseline import summary
    c = state.read()
    base = read(state.job/'host-baseline/scorecard.json')
    pages = []
    for i,page in enumerate(c['pages'],1):
        blocks = c['questions'][str(i)]
        pages.append({'page':i,'committed_objects':len(read(page['scene'])['selected']),
                      'evidence':{k:v for k,v in c['host_evidence'][str(i)].items() if k.endswith('_status')},
                      'issue_blocks':[{k:v for k,v in b.items() if k!='marks'} for b in blocks],
                      'source':state.register(state.job/'inputs'/request['sources'][i-1],'source')['id']})
    return {'version':c['version'],'pages':pages,'tool_catalog':sorted(AGENT_TOOLS),'goal':request.get('goal'),
            'baseline':{'origin':'deterministic-baseline','pptx_sha256':base['pptx_sha256'],
                        'office_status':base['office_status'],'pages':summary(base)},
            'remaining_seconds':state.remaining_seconds()}


def refresh(service,state,current,directory,page):
    # Keep accepted scene/IR immutable. New measurements can only replace evidence
    # if the original IDs, text and geometry still agree with the committed scene.
    import copy
    local = copy.deepcopy(current)
    k = str(page)
    for field in ('ocr','layout','layers'):
        local.get(field,{}).pop(k,None)
    result = service.prepare_page_evidence(state,local,directory,page=page)
    old = read(current['experiment_marks'][k])
    new = read(local['experiment_marks'][k])
    if old!=new or read(current['ocr'][k]).get('lines')!=read(local['ocr'][k]).get('lines'):
        return {'page':page,'status':'EVIDENCE_CHANGED','adopted':False,'reason':'frozen object binding differs'}
    for field in ('ocr','layout','layers','experiment_marks'):
        if k in local.get(field,{}):
            current.setdefault(field,{})[k] = local[field][k]
    return {'page':page,'status':'PREPARED','adopted':True,'layerd_status':result['layerd_status'],
            'layout_status':result['layout_status']}
