"""Host integration for one source-backed geometry repair and text retry."""
from pathlib import Path

from ..agent_service.model_gateway import ModelGateway
from ..agent_service.skill_loader import load_skill
from ..asset_geometry import normalize
from ..materialize import digest, write
from ..parallel_fusion import fuse
from ..parallel_contracts import SceneGraph
from ..parallel_store import key
from .agent_decision import RegisteredTool, RegionalRepairAgent, ToolCandidate
from .text_probe import probe_one


def run(raw_graph, graph, snap, ocr, baseline, fonts, layers, report, provider, cache, store, output, skill_directory):
    """The model chooses a tool. Only this host callback changes private data."""
    raw_graph=SceneGraph.model_validate(raw_graph.model_dump())
    graph=SceneGraph.model_validate(graph.model_dump())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if report['status']!='ADAPTED_XYXY_TO_XYWH':
        result={'status':'NOT_NEEDED','reason':'no coherent asset coordinate failure','published':False}
        write(output/'status.json',result)
        return result
    if not skill_directory:
        result={'status':'UNKNOWN','reason':'regional agent skill unavailable','published':False}
        write(output/'status.json',result)
        return result
    styles={s.id:s for s in raw_graph.text.styles}
    # Use the source's semantic role, not a page-specific ID or string.
    candidates=[t for t in raw_graph.text.texts if raw_graph.ownership[t.id]=='RESOLVED' and not t.spans
                and any(role in styles[t.style].role.lower() for role in ('label','标签','说明'))]
    if not candidates:
        candidates=[t for t in raw_graph.text.texts if raw_graph.ownership[t.id]=='RESOLVED' and not t.spans]
    for text in sorted(candidates,key=lambda t:(t.bbox[0],t.bbox[1])):
        prior=probe_one(raw_graph,text.id,snap['path'],layers,baseline,fonts,output/'f')
        if prior['state']=='STAGED_LOCAL':continue
        failure=output/'failure.json'
        write(failure,{'schema':'RegionalSourceFailure/1','source_sha256':graph.source_sha256,
            'object_id':text.id,'actual_probe':prior,'coordinate_evidence':report})
        def execute(directory):
            asset,adaptation=normalize(raw_graph.asset,store)
            branches={'text':raw_graph.text,'asset':asset,'surface':raw_graph.surface}
            repaired=fuse(snap,branches,ocr)
            # The already source-anchored text definitions are preserved exactly.
            repaired.text=graph.text.model_copy(deep=True)
            repaired=SceneGraph.model_validate(repaired.model_dump())
            write(directory/'coordinate-adaptation.json',adaptation)
            receipt=probe_one(repaired,text.id,snap['path'],layers,baseline,fonts,directory/'p')
            write(directory/'execution-receipt.json',receipt)
            if receipt['state']!='STAGED_LOCAL':
                raise ValueError('registered repair did not produce a stageable text candidate')
            return ToolCandidate(repaired,Path(receipt['candidate_directory']))
        tool=RegisteredTool('normalize_asset_geometry_and_retry_text',
            'Repair the proven whole-branch xyxy/xywh interpretation, re-fuse source ownership, '
            'and run the existing safe text-mask/background/staging tool once. No content or safety gates change.',
            key({str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('text_probe.py'),
                Path(__file__).parents[1]/'asset_geometry.py']}),execute)
        gateway=ModelGateway(provider,Path(cache)/'model-calls')
        policy=load_skill(skill_directory)
        write(output/'policy-binding.json',{'version':policy.version,'files':policy.files})
        session=RegionalRepairAgent(raw_graph,text.id,snap['path'],[failure],tools=[tool],gateway=gateway,
            policy=policy,store=store,output=output,max_decisions=3,max_tools=1)
        result=session.run()
        write(output/'status.json',result)
        return result
    result={'status':'NOT_NEEDED','reason':'no failed semantic text region','published':False}
    write(output/'status.json',result)
    return result
