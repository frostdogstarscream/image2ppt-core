"""Per-object durable budget, content cache and reason-specific repair routing."""
import json
from pathlib import Path
from .materialize import write
from .object_registry import fingerprint


def repair_route(reasons):
    values=set(reasons)
    if values & {'wrong_layer_selected','unknown_layer'} or ('no_visible_pixels' in values and 'ambiguous_components' not in values):return 'replan_layers'
    if values & {'protected_text_ink','uncertain_text_ownership'}:return 'local_text_evidence'
    if values & {'incomplete_owner_set','replacement_contains_other_content','shape_not_wholly_owned','group_transaction_requires_group_preservation'}:return 'replan_owners'
    if values & {'object_ownership_overlap','locked_alpha_overlap'}:return 'joint_arbitration'
    if 'confirmed_boundary_needs_replan' in values:return 'replan_layers'
    if 'independent_visual_review' in values:return 'replan_layers'
    if values & {'ambiguous_components','missing_required_parts'}:return 'sam_prompt'
    return 'retain_parent'


class ObjectSchedule:
    def __init__(self,path):
        self.path=Path(path)
        self.data=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'schema':'ObjectSchedule/1','objects':{}}

    def reserve(self,key,kind,definition):
        if kind not in ('planning_revision','sam_prompt'):raise ValueError('unknown object budget')
        obj=self.data['objects'].setdefault(key,{'calls':[]})
        identity=fingerprint(definition)
        previous=next((c for c in obj['calls'] if c['kind']==kind and c['definition']==identity),None)
        if previous:return previous,False
        used=sum(c['kind']==kind and (kind=='sam_prompt' or c['state']=='RESERVED' or
            (c['state']=='SUCCEEDED' and (c.get('result') or {}).get('substantive_change',True))) for c in obj['calls'])
        if used>=2 or sum(c['kind']==kind for c in obj['calls'])>=6:return None,False
        call={'kind':kind,'definition':identity,'state':'RESERVED'}
        obj['calls'].append(call);write(self.path,self.data)
        return call,True

    def finish(self,call,state,result=None):
        if state not in ('SUCCEEDED','FAILED','INTERRUPTED'):raise ValueError('invalid call state')
        call.update(state=state,result=result);write(self.path,self.data)
