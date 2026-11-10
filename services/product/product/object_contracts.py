"""Private object inventory; coordinates are full-page pixels, never crop-local."""
from typing import Literal
from pydantic import Field, model_validator
from .contracts import Strict


class LogicalObject(Strict):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,60}$')
    description: str = Field(min_length=1, max_length=600)
    prompt_en: str = Field(min_length=1, max_length=300)
    bbox: list[int] = Field(min_length=4, max_length=4, description='Full-page x,y,width,height pixels; include complete fine lines/shadow')
    layers: list[str] = Field(min_length=1, max_length=12, description='Exact layer filenames containing visible object pixels')
    text_policy: Literal['exclude_independent_text', 'intrinsic_label']
    intrinsic_text: str = Field(default='', max_length=200)
    replaces: list[str] = Field(default_factory=list, max_length=30, description='Existing graphic crop/shape IDs fully owned by this object; never text IDs')
    components: list[str] = Field(min_length=1, max_length=30, description='Semantic fragments belonging together, including fine connectors')


class ObjectPlan(Strict):
    schema_version: Literal['LogicalObjects/1'] = Field(alias='schema')
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    image_size: list[int] = Field(min_length=2, max_length=2)
    objects: list[LogicalObject] = Field(max_length=100)
    limitations: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode='after')
    def coherent(self):
        w,h=self.image_size
        if min(w,h)<=0 or max(w,h)>8192: raise ValueError('invalid canvas')
        ids=set(); owned=set()
        for obj in self.objects:
            x,y,bw,bh=obj.bbox
            if min(x,y)<0 or min(bw,bh)<=0 or x+bw>w or y+bh>h: raise ValueError('object outside canvas: '+obj.id)
            if obj.id in ids: raise ValueError('duplicate object id')
            if len(set(obj.layers))!=len(obj.layers): raise ValueError('duplicate layer')
            if len(set(obj.replaces))!=len(obj.replaces) or owned.intersection(obj.replaces): raise ValueError('duplicate ownership')
            if any('/' in name or '\\' in name or not name.endswith('.png') for name in obj.layers): raise ValueError('invalid layer filename')
            if obj.text_policy=='exclude_independent_text' and obj.intrinsic_text: raise ValueError('intrinsic text contradicts policy')
            ids.add(obj.id);owned.update(obj.replaces)
        return self


OBJECT_PROMPT = '''You plan complete graphic objects for faithful screenshot-to-editable-PPT restoration.
Return JSON matching the supplied schema. Images are source page followed by LayerD contact sheet and individual layers rendered over a checkerboard. LayerD connectivity is NOT semantic object identity.
Find every illustration, logo, complex icon and meaningful arrow/connector that should move as a complete object. Include disconnected fragments and thin lines. Separate neighboring icons. Do not list ordinary text, card backgrounds, simple rectangles or separators already represented by native shapes. Do not redesign or infer hidden pixels.
Use full-page integer x,y,width,height coordinates, not 0..1000 and not xyxy. Boxes enclose the COMPLETE visible object including white parts and shadows. Inspect actual edges, not symmetry assumptions. Select exact layer filenames using their previews; do not indiscriminately select all layers.
Independent text/sequence numbers remain native and must be excluded. Intrinsic lettering in a logo/icon may remain raster only with text_policy intrinsic_label and its actual wording. Preserve text and geometry baseline; do not change fonts or layout.
The supplied graphic baseline IDs may be declared in replaces ONLY if their complete visible content belongs to this object. Never claim a text node or large card background. Missing objects may have replaces=[]; do not omit them just because baseline missed them. components describes logical pieces, not connected-component numbers.
No other planner outputs are supplied. Report limitations honestly; your inventory is a proposal, not acceptance.'''


class ObjectPart(Strict):
    name: str = Field(min_length=1, max_length=120)
    bbox: list[int] = Field(min_length=4, max_length=4)
    layers: list[str] = Field(min_length=1, max_length=12)


class ClosedObject(LogicalObject):
    kind: Literal['icon', 'logo', 'illustration', 'decoration', 'shadow', 'glow']
    parts: list[ObjectPart] = Field(min_length=1, max_length=30)
    exclusions: list[ObjectPart] = Field(default_factory=list, max_length=30)
    required_owners: list[str] = Field(default_factory=list, max_length=30)


class ClosedPlan(ObjectPlan):
    schema_version: Literal['LogicalObjects/2'] = Field(alias='schema')
    objects: list[ClosedObject] = Field(max_length=100)

    @model_validator(mode='after')
    def regions(self):
        w,h=self.image_size
        for obj in self.objects:
            if not set(obj.required_owners)<=set(obj.replaces):
                raise ValueError('required owner missing: '+obj.id)
            for part in [*obj.parts,*obj.exclusions]:
                x,y,bw,bh=part.bbox
                if min(x,y)<0 or min(bw,bh)<=0 or x+bw>w or y+bh>h:
                    raise ValueError('part outside canvas')
                if part in obj.parts and not set(part.layers)<=set(obj.layers):raise ValueError('part layer outside object')
        return self


def read_plan(raw):
    """Explicit legacy adapter. Never rewrite the raw model reply."""
    if raw.get('schema')=='LogicalObjects/2':return ClosedPlan.model_validate(raw),False
    old=ObjectPlan.model_validate(raw)
    payload=old.model_dump();payload['schema']='LogicalObjects/2'
    payload['objects']=[{**o.model_dump(),'kind':'icon','parts':[{'name':'legacy localization; requires review',
        'bbox':o.bbox,'layers':o.layers}],'exclusions':[],'required_owners':o.replaces} for o in old.objects]
    return ClosedPlan.model_validate(payload),True


class ObjectVerdict(Strict):
    id: str
    candidate_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    pixels: Literal['PASS','FAIL','UNKNOWN']
    complete_object: Literal['PASS','FAIL','UNKNOWN']
    replacement: Literal['PASS','FAIL','UNKNOWN']
    background_restoration: Literal['PASS','FAIL','UNKNOWN'] = 'UNKNOWN'
    white_content_assembly: Literal['PASS','FAIL','UNKNOWN'] = 'UNKNOWN'
    native_owner_scope: Literal['PASS','FAIL','UNKNOWN'] = 'UNKNOWN'
    source_unmix: Literal['PASS','FAIL','UNKNOWN'] = 'UNKNOWN'
    reasons: list[str] = Field(max_length=20)


class ObjectReview(Strict):
    schema_version: Literal['ObjectReview/1'] = Field(alias='schema')
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    scene_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    objects: list[ObjectVerdict] = Field(max_length=100)


class ObjectRepairs(Strict):
    schema_version: Literal['ObjectRepairs/1'] = Field(alias='schema')
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    image_size: list[int] = Field(min_length=2,max_length=2)
    objects: list[ClosedObject] = Field(max_length=100,description='Only changed target objects; omit unchanged objects')
    limitations: list[str] = Field(default_factory=list,max_length=30)


def apply_repairs(plan,raw,targets):
    repair=ObjectRepairs.model_validate(raw)
    if repair.source_sha256!=plan.source_sha256 or repair.image_size!=plan.image_size:raise ValueError('repair binding mismatch')
    changed={o.id:o for o in repair.objects}
    if len(changed)!=len(repair.objects) or not set(changed)<=set(targets):raise ValueError('repair changed non-target inventory')
    payload=plan.model_dump();payload['objects']=[changed.get(o.id,o).model_dump() for o in plan.objects]
    return ClosedPlan.model_validate(payload)


CLOSED_OBJECT_PROMPT=OBJECT_PROMPT+'''
Use LogicalObjects/2. Classify kind, list required parts with their own source-coordinate seed boxes and layers; exclusions identify independent labels, neighbors, platform/base, connectors not belonging to the object. These boxes are localization evidence, never masks or permission to cut strokes. required_owners must include every old fragment, even fragments with a different name. Be precise about the visible content of old crops; unknown ownership is a limitation.
For a cloud-server symbol, cloud outline and internal server bars/dots form ONE object; a separately typeset caption, caption plate, platform and external connectors remain separate. Include ALL old cloud halves in replaces and required_owners.
On repair, change only the objects identified in feedback, and explain changed evidence. Do not repeat identical layer/part/prompt definitions.'''

OBJECT_REVIEW_PROMPT='''Independently review screenshot-to-PPT object candidates. Return ObjectReview/1, bound exactly to supplied source/scene/PNG hashes. Images show original, actual PowerPoint slide and per-object detail strips: transparent PNG, original region, actual region after deleting ONLY that object and saving/reopening. For EVERY candidate decide separately: pixels (soft alpha, white content, holes, fine lines, background and independent-text leakage); complete_object (all declared parts, no neighbors); replacement (all old fragments accounted for, correct overlap/z, preserved neighbors). If background_assets is nonempty, explicitly judge background_restoration: verify that the deletion image shows only the correct page/card background, without ANY remaining icon stroke, halo fragment, glyph or neighbor embedded in the backing. Colour patches or invented backing must FAIL. This is a proposed restoration, NOT verified background. Missing or uncertain evidence is UNKNOWN, never PASS. A candidate mask is NOT its own ground truth. Do not judge font slant. Keep human acceptance separate. Never approve a cropped half-object just because it matches that half of the source.'''
OBJECT_REVIEW_PROMPT+='''
Background review applies ONLY inside actual OLD OWNER rectangles, shown with RED outlines in SOURCE/AFTER DELETE panels. The panels also include surrounding context. Independent text, cards, connectors and neighbors OUTSIDE the red rectangles MUST remain; their presence is correct preservation, not a ghost or leakage. Red borders are private annotations, not source/product pixels. For native_owner_scope_hypothesis, additionally return native_owner_scope PASS only after checking the ORIGINAL content INSIDE EVERY old rectangle belongs exclusively to this object or evidenced native backing. Unknown old content fails; context outside old rectangles is not evidence of an owner conflict.'''
OBJECT_REVIEW_PROMPT+='''
Diagnostic counts are NOT semantic decisions. owner_proof_before_restoration describes OLD opaque rectangle pixels outside the foreground Alpha BEFORE proposed background restoration. A positive unexplained/unverified count commonly represents background padding: independently RESOLVE its content using the ORIGINAL source and actual deletion view; the count alone proves neither failure nor success. retained_old_alpha_pixels denotes preserved existing transparent-asset samples (possibly real white/soft edges), not proven background leakage. Recomposition error is auxiliary: nonzero error alone is not a defect and low error alone is not fidelity. Judge visible completeness, holes, Alpha and ownership from images. With no background_assets, an old transparent asset required no new backing patch; do not demand a nonexistent restoration asset. source_unmix PASS requires affirmative visual evidence, UNKNOWN when genuinely unresolved; never infer PASS merely from diagnostics or a clean-looking candidate in isolation.'''
OBJECT_REVIEW_PROMPT+='''
native_background_comparison compares sampled ORIGINAL backing with a PREVIOUSLY GENERATED PPT card/page fill, which may itself be inaccurate. It is not comparison against a known original background. A discrepancy alone cannot establish invented backing or wrong scope. Compare the ORIGINAL source and actual after-delete region for wrong colour, seams, texture, lost neighbors and residual strokes; reject visible mismatches or unresolved source evidence.'''
