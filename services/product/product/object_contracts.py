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
