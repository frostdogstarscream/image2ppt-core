"""Keep independently verified edits when replacing source-owned content."""
import json
from pathlib import Path

from ..materialize import digest
from ..parallel_store import key


def previous_edits_lost(state, current, candidate, candidate_ir, *, all_edits=False, graph=None):
    if not all_edits and not candidate.get('card_repair_targets'):
        return []
    page = candidate['page']
    parent_ir = Path(current['pages'][page - 1]['ir'])
    before = json.loads(parent_ir.read_bytes())
    after = json.loads(Path(candidate_ir).read_bytes())
    kinds = {node['id']:node['type'] for node in after['nodes']}
    protected = {'shape','text','table','chart'}
    if all_edits:
        protected.add('image')
    # The graph verifier separately proves ownership when a raster word becomes native text.
    converted = {aid for oid, record in (graph.selected.items() if graph is not None else ())
                 if kinds.get(oid) == 'text' for aid in record.get('text_cutouts', ())}
    lost = {
        node['id']:node['type'] for node in before['nodes']
        if node['type'] in protected and node['id'] != 'source-plate' and node['id'] not in converted
        and kinds.get(node['id']) != node['type']
    }
    # Identical membership preserves a group's selection even if its generated name changes.
    new_groups = {frozenset(group['member_ids']) for group in after.get('groups', [])}
    lost_groups = {group['id']:set(group['member_ids']) for group in before.get('groups', [])
                   if frozenset(group['member_ids']) not in new_groups}
    if not lost and not lost_groups:
        return []
    parent_sha = digest(parent_ir)
    page_hashes = [digest(row['ir']) for row in current['pages'] if row and row.get('ir')]
    confirmed = {}
    for receipt in sorted((state.root/'office-cache').glob('*.json')):
        proof = json.loads(receipt.read_bytes())
        identity = proof.get('identity', {})
        if receipt.stem != key(identity):
            continue
        irs = identity.get('irs')
        if irs == [parent_sha]:
            checked_page = 1
        elif len(page_hashes) == len(current['pages']) and irs == page_hashes:
            checked_page = page
        else:
            continue
        result = proof.get('result', {})
        office = result.get('office', {})
        if (not identity.get('pptx') or result.get('pptx_sha256') != identity['pptx']
                or office.get('pptx_sha256') != identity['pptx']
                or office.get('status') not in {'PASS_APP_AUTOMATED','FAIL_APP_AUTOMATED'}):
            continue
        actions = {}
        for record in office.get('records', []):
            oid = record['id']
            if (record.get('page') != checked_page or record.get('saved_reopened') is not True
                    or record.get('outside_pixels') != 0 or (oid not in lost and oid not in lost_groups)):
                continue
            if oid in lost_groups and set(record.get('group_members', [])) != lost_groups[oid]:
                continue
            actions.setdefault(oid, set()).add(record.get('action'))
        for oid, completed in actions.items():
            if {'move-scale','delete'} <= completed:
                confirmed[oid] = {'object_id':oid, 'kind':lost.get(oid, 'group'),
                                  'failure':'PREVIOUS_NATIVE_EDIT_LOST', 'parent_pptx_sha256':identity['pptx'],
                                  'card_repair_targets':candidate.get('card_repair_targets', []),
                                  'conflict_ids':sorted(lost_groups.get(oid, ())),
                                  'reason':'Reconstruction would remove an Office-verified independent edit on the current page.'}
    return [confirmed[oid] for oid in sorted(confirmed)]
