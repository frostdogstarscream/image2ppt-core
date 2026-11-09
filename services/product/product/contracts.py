"""Private model contract. Public SlideIR and the fixed compiler stay unchanged."""
from __future__ import annotations

import math
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)

    def model_dump(self, **kwargs):
        kwargs.setdefault('by_alias',True)
        return super().model_dump(**kwargs)


class Style(Strict):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,60}$')
    role: str = Field(min_length=1, max_length=120)
    ea: str
    latin: str
    size_px: float = Field(ge=4, le=400)
    bold: bool
    color: str = Field(pattern=r'^[0-9A-Fa-f]{6}$')


class Span(Strict):
    text: str = Field(min_length=1, max_length=5000)
    style: str


class Text(Strict):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,60}$')
    bbox: list[float] = Field(min_length=4, max_length=4)
    z: int = Field(ge=0, le=10000)
    style: str
    text: str = Field(min_length=1, max_length=5000)
    align: Literal['left', 'center', 'right'] = 'left'
    line_spacing_px: float = Field(default=0, ge=0, le=600)
    spans: list[Span] = Field(default_factory=list, max_length=200)


class GradientStop(Strict):
    offset: int = Field(ge=0,le=100000)
    color: str = Field(pattern=r'^[0-9A-Fa-f]{6}$')


class Gradient(Strict):
    angle_deg: float
    stops: list[GradientStop] = Field(min_length=2,max_length=8)

    @model_validator(mode='after')
    def ordered(self):
        offsets=[s.offset for s in self.stops]
        if offsets[0]!=0 or offsets[-1]!=100000 or sorted(set(offsets))!=offsets:
            raise ValueError('gradient must have increasing stops from 0 to 100000')
        return self


class Shape(Strict):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,60}$')
    bbox: list[float] = Field(min_length=4, max_length=4)
    z: int = Field(ge=0, le=10000)
    geometry: Literal['rect','roundRect','ellipse','line','chevron','pentagon',
                      'rightArrow','leftArrow','trapezoid','parallelogram','hexagon','diamond','triangle']
    fill: str = Field(pattern=r'^(none|[0-9A-Fa-f]{6})$')
    line_color: str = Field(default='FFFFFF', pattern=r'^[0-9A-Fa-f]{6}$')
    line_width_pt: float = Field(default=0, ge=0, le=20)
    corner_radius_px: float | None = Field(default=None,ge=0)
    fill_gradient: Gradient | None = None


class Crop(Strict):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,60}$')
    bbox: list[float] = Field(min_length=4, max_length=4)
    z: int = Field(ge=0, le=10000)
    description: str = Field(min_length=1, max_length=300)


class Repeated(Strict):
    """Explicit repeated component roles, not inferred from old font fields."""
    role: str
    members: list[str] = Field(min_length=2, max_length=100,description='IDs from texts only; do not include shapes or crops')
    alignment: Literal['none', 'left', 'center', 'right'] = Field(default='none',description='Shared X coordinate across all members: left edge, center X or right edge. For horizontally distributed labels use none; this is NOT text alignment within each box.')


class Page(Strict):
    schema_version: Literal['VisualPage/1'] = Field(alias='schema')
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    image_size: list[int] = Field(min_length=2, max_length=2)
    background: str = Field(pattern=r'^[0-9A-Fa-f]{6}$')
    styles: list[Style] = Field(min_length=1, max_length=100)
    texts: list[Text] = Field(min_length=1, max_length=400)
    shapes: list[Shape] = Field(default_factory=list, max_length=400)
    crops: list[Crop] = Field(default_factory=list, max_length=100)
    repeated: list[Repeated] = Field(default_factory=list, max_length=100)
    limitations: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode='after')
    def coherent(self):
        w, h = self.image_size
        if not 1 <= w <= 8192 or not 1 <= h <= 8192:
            raise ValueError('image dimensions unsupported')
        nodes = [*self.texts, *self.shapes, *self.crops]
        if len({n.id for n in nodes}) != len(nodes):
            raise ValueError('duplicate node id')
        styles = {s.id: s for s in self.styles}
        if len(styles) != len(self.styles):
            raise ValueError('duplicate style id')
        for n in nodes:
            x,y,bw,bh = n.bbox
            if not all(math.isfinite(v) for v in n.bbox) or min(x,y)<0 or min(bw,bh)<=0 or x+bw>w+.01 or y+bh>h+.01:
                raise ValueError(f'bbox outside image: {n.id} bbox={n.bbox}; image_size={[w,h]}; keep x+width <= {w} and y+height <= {h}; wrap text into source lines rather than extend outside the page')
        for t in self.texts:
            if any(not line for line in t.text.split('\n')):
                raise ValueError('blank text lines must use spacing instead: '+t.id)
            if t.style not in styles or any(s.style not in styles for s in t.spans):
                raise ValueError('unknown text style: '+t.id)
            if t.spans and ''.join(s.text for s in t.spans) != t.text:
                raise ValueError('emphasis spans must preserve text: '+t.id)
        # A source crop cannot contain the native text that will be drawn on top.
        for crop in self.crops:
            if crop.bbox[2]*crop.bbox[3] > w*h*.18:
                raise ValueError('crop exceeds 18% of page: '+crop.id)
            conflicts=[{'id':t.id,'bbox':t.bbox} for t in self.texts if intersection(crop.bbox,t.bbox)>1]
            if conflicts:
                raise ValueError(f'crop overlaps editable text: {crop.id} bbox={crop.bbox}; text_boxes={conflicts}; split or trim crop around these boxes, preserve native text')
        by_id = {t.id:t for t in self.texts}
        candidates = [t for t in self.texts if re.match(r'^0[1-9](?:$|[^0-9])',t.text)]
        columns=[]
        for t in sorted(candidates,key=lambda t:t.bbox[0]):
            if not columns or abs(t.bbox[0]-columns[-1][0].bbox[0])>max(12,w*.015):
                columns.append([])
            columns[-1].append(t)
        for numbers in columns:
            if len(numbers)<2:continue
            if any(not re.fullmatch(r'0[1-9]',t.text) for t in numbers):
                raise ValueError('repeated sequence numbers must be separate text objects')
            if not any(set(t.id for t in numbers).issubset(g.members) for g in self.repeated):
                raise ValueError('sequence numbers need an explicit repeated role')
        for group in self.repeated:
            if len(set(group.members)) != len(group.members) or any(i not in by_id for i in group.members):
                raise ValueError('repeated members must be unique text IDs only; not shapes/crops: '+group.role)
            members = [by_id[i] for i in group.members]
            if len({t.style for t in members}) != 1:
                raise ValueError('repeated role must share a style: '+group.role)
            if group.alignment != 'none':
                factor = {'left':0, 'center':.5, 'right':1}[group.alignment]
                positions = [t.bbox[0]+factor*t.bbox[2] for t in members]
                if max(positions)-min(positions)>3:
                    raise ValueError('repeated role alignment mismatch: '+group.role)
        return self


def intersection(a,b):
    return max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))


class Finding(Strict):
    category: Literal['wording','structure','font','emphasis','layout','graphics','other']
    description: str = Field(min_length=1, max_length=1000)
    node_ids: list[str] = Field(default_factory=list, max_length=100)
    aspect: Literal['italic_only','other'] = 'other'
    source_bbox: list[float] | None = Field(default=None,min_length=4,max_length=4)
    suggested_fix: str = Field(default='',max_length=1500)


class Review(Strict):
    schema_version: Literal['VisualReview/1'] = Field(alias='schema')
    source_sha256: str
    plan_sha256: str
    preview_sha256: str
    passed: bool
    checked: list[Literal['wording','structure','font','emphasis','layout','graphics']]
    findings: list[Finding] = Field(max_length=100)

    @model_validator(mode='after')
    def complete(self):
        if set(self.checked) != {'wording','structure','font','emphasis','layout','graphics'}:
            raise ValueError('review must check all six dimensions')
        if self.passed == bool(self.findings):
            raise ValueError('review verdict contradicts findings')
        return self
