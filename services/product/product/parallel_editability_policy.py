"""Task policy sidecar. Preference never relaxes content or ownership gates."""
from typing import Literal
from pydantic import Field
from .contracts import Strict


class EditabilityProfile(Strict):
    schema_version: Literal['EditabilityProfile/1']=Field(default='EditabilityProfile/1',alias='schema')
    object_id: str
    semantic_role: str
    edit_priority: Literal['HIGH','MEDIUM','LOW']='HIGH'
    visual_sensitivity: Literal['HIGH','MEDIUM','LOW']='MEDIUM'
    operations: list[Literal['replace_text','change_font','move','resize']]=Field(default_factory=lambda:['replace_text','change_font','move','resize'])
    content_integrity: Literal['STRICT']='STRICT'
    policy: Literal['FIDELITY_FIRST','BALANCED','EDITABILITY_FIRST']='BALANCED'
    basis: list[str]=Field(default_factory=lambda:['text_branch','task_default'])


def profiles(graph,policy='BALANCED'):
    styles={s.id:s for s in graph.text.styles}
    intrinsic={r.source for r in graph.relations if r.kind=='intrinsic_text'}
    return [EditabilityProfile(object_id=t.id,semantic_role=styles[t.style].role,
        edit_priority='LOW' if t.id in intrinsic else 'HIGH',
        visual_sensitivity='HIGH' if t.id in intrinsic else 'MEDIUM',policy=policy).model_dump(by_alias=True)
        for t in graph.text.texts]
