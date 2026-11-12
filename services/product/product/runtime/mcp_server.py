"""Image2PPT MCP stdio server. stdout is reserved for MCP frames."""
import argparse
import asyncio
import base64
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool

from ..parallel_config import configure
from .service import RuntimeService, TOOLS, VisualReview


READ_ONLY = {'get_job_context', 'inspect_scene', 'get_evidence', 'list_evidence', 'get_progress'}
DESCRIPTION = {
    'get_job_context': 'Read job status, version and immutable original-image evidence references.',
    'run_perception': 'Run OCR and text/asset/surface perception once and create a source-safe initial scene.',
    'inspect_scene': 'Read scene, complete object IDs, ownership, dependencies and perception failures.',
    'rebuild_region': 'Generate one regional text or whole-asset candidate using existing production safety gates.',
    'repair_region': 'Try a changed strategy after inspecting an actual failed candidate. Same strategy is rejected.',
    'validate_candidate': 'Verify candidate and run actual PowerPoint edits and source-owner residual checks.',
    'commit_region': 'Commit a verified region against its exact parent scene; independent regions are retained.',
    'compose_delivery': 'Compile the current committed scene using the fixed compiler.',
    'validate_office': 'Validate final PPTX with actual PowerPoint and return required visual evidence IDs.',
    'get_evidence': 'Receive actual source/candidate/Office images or diagnostic JSON by evidence ID.',
    'record_visual_review': 'Record your visual review after viewing every required image and hard-check receipt.',
    'finalize_job': 'Finalize only with current actual editing checks and hash-bound OpenCode visual review.'}
DESCRIPTION.update({
    'inspect_image':'Read actual image dimensions, mode and sampled RGB color frequencies. Use get_evidence to view image content.',
    'crop_image':'Crop an immutable image evidence reference, optionally enlarge its preview, and return measured RGB frequencies from the original crop.',
    'list_fonts':'List actual installed CJK and common Latin font faces, including real weight availability. Full verified font metadata is returned as evidence.',
    'fit_background':'Fit a continuous background from an image evidence_id and xywh bbox, or from a scene text object. This produces a hypothesis and does not change the source.',
    'complete_text_mask':'SCENE MODE: supply page, object_id, background_id; OMIT original_mask_id unless you already have a genuine binary Mask. The tool derives initial strokes automatically. INDEPENDENT MODE: supply evidence_id, original_mask_id, background_id, bbox, glyph_height. Returns hypotheses, never erase authority.',
    'create_scene':'Initialize a page once. An existing page returns SCENE_EXISTS; use update_scene for revisions. Preserve exact OCR wording and bind text.members to OCR line IDs; inspect unresolved owners in the response.',
    'update_scene':'Revise scene evidence and add draft objects. Existing accepted objects and their styles are preserved; revising those requires dependency-local rollback.',
    'create_candidate':'Construct an unapproved text preview from explicit mask, background and optional layout evidence. Returns exploratory artifacts even when ownership or visual quality remains uncertain.',
    'segment_object':'Obtain SAM evidence for a complete icon/logo/illustration asset, not a text object. Supply either an existing scene asset ID or full object_evidence. Text strokes use complete_text_mask with page, object_id and background_id; original_mask_id is optional. SAM does not authorize erasure.',
    'fit_text_layout':'Compute a native text frame using installed font metrics. The result is a proposal; source ownership and candidate validation remain separate.',
    'run_perception':'Optional high-cost shortcut invoking three model branches and LayerD. Basic evidence tools can be used independently.'})
DESCRIPTION.update({
    'validate_mask':'Measure binary mask size and supplied protection/reference overlap. Returns measurements and uncertainty without approving source ownership.',
    'unmix_foreground':'Compute RGBA from explicit source evidence_id, background_id and mask_id. Does not choose a recovery strategy.',
    'validate_rgba':'Measure an RGBA evidence image, optional source recomposition and protection overlap. No staged IR is required.',
    'compile_pptx':'Compile candidate_id as an unapproved native preview, or compile the current committed scene when candidate_id is omitted.',
    'validate_candidate':'For exploratory candidates, prepare native preview artifacts. Legacy staged candidates retain formal submission validation.',
    'validate_office':'With candidate_id, return actual preview edit diagnostics without approval. With no candidate_id, validate final composition.',
    'record_observation':'Record Agent findings, uncertainty and next action on arbitrary task evidence, including failed experiments.',
    'commit_candidate':'Submit a candidate through source-owner, content and actual-edit safety checks. Exploratory preview approval never bypasses these checks.'})


def schemas():
    from ..parallel_contracts import TextEvidence, AssetEvidence, SurfaceEvidence
    string = {'type': 'string'}
    page = {'type': 'integer', 'minimum': 1}
    strings = {'type': 'array', 'items': string}
    region = {'page': page, 'object_id': string}
    fields = {
        'list_evidence': {'role':string,'limit':{'type':'integer','minimum':1,'maximum':100}},
        'inspect_scene': {'page': {'type': 'integer', 'minimum': 1}},
        'rebuild_region': {'page': {'type': 'integer', 'minimum': 1}, 'object_id': string,
                           'strategy': {'type': 'string', 'enum': ['auto', 'next_background', 'verified-backing', 'source-halo', 'sam']}},
        'repair_region': {'candidate_id': string, 'strategy': string},
        'validate_candidate': {'candidate_id': string},
        'commit_region': {'candidate_id': string},
        'get_evidence': {'evidence_ids': {'type': 'array', 'items': string, 'minItems': 1, 'maxItems': 8}},
        'record_visual_review': {'review': VisualReview.model_json_schema()}}
    fields.update({
        'inspect_image': {'evidence_id': string},
        'crop_image': {'evidence_id': string, 'bbox': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4}, 'scale': {'type': 'number', 'minimum': .1, 'maximum': 4}},
        'compare_images': {'first': string, 'second': string},
        'run_ocr': {'page': page}, 'analyze_text_regions': {'page': page},
        'analyze_image': {'evidence_id': string, 'question': string},
        'decompose_layers': {'page': page}, 'segment_object': dict(region),
        'extract_background': {**region, 'mask_id': string},
        'fit_background': dict(region), 'inpaint_background': {**region, 'mask_id': string},
        'complete_text_mask': dict(region), 'validate_mask': {**region, 'mask_id': string},
        'unmix_foreground': {**region, 'strategy': string},
        'validate_rgba': {'candidate_id': string}, 'fit_text_layout': dict(region),
        'create_scene': {'page': page, 'evidence': {'type': 'object'}},
        'update_scene': {'page': page, 'evidence': {'type': 'object'},'remove_object_ids':strings},
        'create_candidate': {**region, 'mask_id': string, 'background_id': string},
        'commit_candidate': {'candidate_id': string},
        'rollback_region': {'page': page, 'object_ids': strings, 'reason': string},
        'run_image_script': {'script': string, 'inputs': strings}})
    fields['fit_background'].update(evidence_id=string,bbox={'type':'array','items':{'type':'integer'},'minItems':4,'maxItems':4},protected_id=string)
    fields['fit_background']['context_width']={'type':'integer','minimum':8,'maximum':128,
        'description':'Unprotected source-strip sampling width in pixels. A changed width supplies different background evidence. All original ownership and fit validation thresholds still apply.'}
    fields['complete_text_mask'].update(evidence_id=string,original_mask_id=string,background_id=string,protected_id=string,
        bbox={'type':'array','items':{'type':'integer'},'minItems':4,'maxItems':4},glyph_height={'type':'number','exclusiveMinimum':0})
    fields['complete_text_mask']['original_mask_id']={'type':'string','description':'Optional existing binary grayscale Mask evidence. Never use a source/background image ID here. Omit in scene mode to derive original strokes automatically.'}
    fields['unmix_foreground'].update(object_evidence={'type':'object'},protected_id=string,alpha_id=string)
    fields['unmix_foreground'].update(evidence_id=string,background_id=string,mask_id=string)
    fields['validate_mask'].update(evidence_id=string,protected_id=string,reference_mask_id=string)
    fields['validate_rgba'].update(evidence_id=string,source_id=string,background_id=string,protected_id=string)
    fields['compile_pptx']={'candidate_id':string}
    fields['validate_office']={'candidate_id':string}
    fields['inpaint_background']['evidence_id']=string
    fields['record_observation']={'evidence_ids':strings,'findings':{'type':'array','items':{'type':'object'}},'next_action':string}
    fields['segment_object']=dict(fields['segment_object'],object_evidence={'type':'object'})
    frame={'type':'array','items':{'type':'number','exclusiveMinimum':0},'minItems':2,'maxItems':2}
    fields['fit_text_layout']['minimum_size']=frame
    fields['create_candidate']['frame_minimum']=frame
    fields['create_candidate']['layout_id']=string
    fields['create_candidate']['frame_minimum']['description']='Optional transparent native text frame [width,height] in source pixels. Keeps source origin and erase Mask fixed. Use measured Office bounds when testing a layout repair.'
    from ..object_contracts import ClosedObject
    object_schema=ClosedObject.model_json_schema()
    definitions = object_schema.pop('$defs',{})
    fields['segment_object']['object_evidence']=object_schema
    branches = {}
    for branch, contract in [('text', TextEvidence), ('asset', AssetEvidence), ('surface', SurfaceEvidence)]:
        schema = contract.model_json_schema()
        definitions.update(schema.pop('$defs', {}))
        for name in ('source_sha256', 'image_size', 'transform', 'method_version', 'schema'):
            schema['properties'].pop(name, None)
        schema['required'] = [name for name in schema.get('required', []) if name in schema['properties']]
        if branch=='text':
            schema['properties']['members']['description']='Map text object IDs to exact OCR line IDs from run_ocr. Content contradictions remain raster. Unique exact wording with matching geometry can be bound by the host.'
        branches[branch] = schema
    for name in ('create_scene', 'update_scene'):
        fields[name]['evidence'] = {'type': 'object', 'properties': branches, 'additionalProperties': False}
    result = []
    for name in TOOLS:
        properties = {'job_id': string, **fields.get(name, {})}
        required = list(properties)
        if name=='list_evidence':required=['job_id']
        if name=='update_scene':required.remove('remove_object_ids')
        if name in ('rebuild_region', 'unmix_foreground'):
            required.remove('strategy')
        if name=='segment_object':
            required.remove('object_id')
            required.remove('object_evidence')
        if name in ('fit_background','complete_text_mask'):
            required=['job_id']
        if name=='unmix_foreground':
            required=['job_id','evidence_id','background_id','mask_id']
        if name=='validate_mask':required=['job_id','mask_id']
        if name=='validate_rgba':required=['job_id','evidence_id']
        if name in ('compile_pptx','validate_office'):required=['job_id']
        if name=='inpaint_background':required=['job_id','mask_id']
        if name=='record_observation':required.remove('next_action')
        if name=='fit_text_layout':required.remove('minimum_size')
        if name=='create_candidate':
            required.remove('frame_minimum')
            required.remove('layout_id')
        if name == 'crop_image':
            required.remove('scale')
        if name not in READ_ONLY:
            properties.update(expected_version={'type': 'integer', 'minimum': 0},
                              idempotency_key={'type':'string','minLength':8,'maxLength':100,'pattern':'^[A-Za-z0-9_-]+$'})
            required.append('idempotency_key')
            if name in ('commit_candidate', 'commit_region'):
                required.append('expected_version')
        result.append(Tool(name=name, description=DESCRIPTION.get(name, name.replace('_', ' ')+'. Independent evidence tool; outputs do not authorize pixel erasure.'), inputSchema={
            'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False,
            **({'$defs': definitions} if name in ('create_scene', 'update_scene', 'segment_object') else {})}))
    return result


def create_server(root, bound_job=None):
    service = RuntimeService(root)
    server = Server('image2ppt')
    def catalog():
        profile='all'
        if bound_job:
            request=json.loads((service.state(bound_job).job/'request.json').read_text('utf8'))
            profile=request.get('tool_profile','all')
        shortcuts={'run_perception','run_parallel_pipeline','rebuild_region','repair_region'} if profile=='basic' else set()
        return [tool for tool in schemas() if tool.name not in shortcuts]

    @server.list_tools()
    async def list_tools():
        return catalog()

    @server.call_tool()
    async def call_tool(name, arguments):
        arguments = dict(arguments or {})
        if bound_job and arguments.get('job_id') != bound_job:
            raise ValueError('tool is bound to another job')
        from jsonschema import validate
        schema = next(t.inputSchema for t in catalog() if t.name == name)
        validate(arguments, schema)
        result = await asyncio.to_thread(service.call, name, **arguments)
        if name != 'get_evidence':
            return [TextContent(type='text', text=json.dumps(result, ensure_ascii=False))]
        content = []
        for path, reference in result:
            if path.suffix.lower() == '.png':
                content.append(TextContent(type='text', text=json.dumps(reference)))
                content.append(ImageContent(type='image', mimeType='image/png',
                                            data=base64.b64encode(path.read_bytes()).decode('ascii')))
            else:
                content.append(TextContent(type='text', text=json.dumps(reference)+'\n'+path.read_text('utf8')))
        # A supplied-image receipt records transport provision, not proof of model cognition.
        state = service.state(arguments['job_id'])
        for identifier in arguments['evidence_ids']:
            state.evidence(identifier, delivered=True)
        return content

    return server


async def serve(root, bound_job=None):
    configure()
    os.environ.setdefault('IMAGE2PPT_LAYERD_TIMEOUT','180')
    server = create_server(root, bound_job)
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--job', default=os.environ.get('IMAGE2PPT_BOUND_JOB'))
    options = parser.parse_args()
    asyncio.run(serve(options.root, options.job))
