"""Private r10 evidence contracts. None of these extend public SlideIR."""
from typing import Literal
from pydantic import Field, model_validator
from .contracts import Strict, Style, Text, Shape
from .object_contracts import ClosedObject,ClosedPlan


class Binding(Strict):
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    image_size: list[int] = Field(min_length=2, max_length=2)
    transform: list[float] = Field(default_factory=lambda:[1.,0.,0.,1.,0.,0.],min_length=6,max_length=6)
    method_version: str
    references: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def dimensions(self):
        if min(self.image_size)<=0 or max(self.image_size)>8192:raise ValueError('invalid evidence canvas')
        return self


class TextEvidence(Binding):
    schema_version: Literal['TextEvidence/1'] = Field(alias='schema')
    styles: list[Style] = Field(default_factory=list,max_length=100)
    texts: list[Text] = Field(default_factory=list,max_length=400)
    # OCR member IDs make content disagreement explicit rather than rewriting it.
    members: dict[str,list[str]] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def coherent_text(self):
        styles={s.id for s in self.styles};ids={t.id for t in self.texts}
        if len(styles)!=len(self.styles) or len(ids)!=len(self.texts):raise ValueError('duplicate text/style identity')
        if not set(self.members)<=ids:raise ValueError('OCR members reference unknown text')
        w,h=self.image_size
        for t in self.texts:
            if any(not line for line in t.text.split('\n')):raise ValueError('blank text lines must use explicit spacing')
            if t.style not in styles or any(s.style not in styles for s in t.spans):raise ValueError('unknown shared text style')
            if t.spans and ''.join(s.text for s in t.spans)!=t.text:raise ValueError('emphasis rewrote text')
            # Coordinates are proposals. Fusion retains invalid candidates as
            # UNOWNED rather than clipping them or discarding the entire branch.
        return self


class AssetEvidence(Binding):
    schema_version: Literal['AssetEvidence/1'] = Field(alias='schema')
    objects: list[ClosedObject] = Field(default_factory=list,max_length=100)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def coherent_assets(self):
        if len({o.id for o in self.objects})!=len(self.objects):raise ValueError('duplicate asset identity')
        return self


class SurfaceEvidence(Binding):
    schema_version: Literal['SurfaceEvidence/1'] = Field(alias='schema')
    background: str = Field(pattern=r'^[0-9A-Fa-f]{6}$')
    shapes: list[Shape] = Field(default_factory=list,max_length=400)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def coherent_surfaces(self):
        if len({s.id for s in self.shapes})!=len(self.shapes):raise ValueError('duplicate surface identity')
        return self


class Relation(Strict):
    source: str
    target: str
    kind: Literal['contains','intrinsic_text','label_for','occludes','part_of','connects']
    status: Literal['RESOLVED','AMBIGUOUS','UNOWNED']
    evidence: list[str]


class SceneGraph(Binding):
    schema_version: Literal['SceneGraph/1'] = Field(alias='schema')
    text: TextEvidence
    asset: AssetEvidence
    surface: SurfaceEvidence
    relations: list[Relation] = Field(default_factory=list)
    ownership: dict[str,Literal['RESOLVED','AMBIGUOUS','UNOWNED']]
    selected: dict[str,dict] = Field(default_factory=dict)
    transactions: list[dict] = Field(default_factory=list)
    raster_backing: dict | None = None
    human_acceptance: bool = False

    @model_validator(mode='after')
    def bindings(self):
        for b in (self.text,self.asset,self.surface):
            if b.source_sha256!=self.source_sha256 or b.image_size!=self.image_size or b.transform!=self.transform:
                raise ValueError('branch snapshot binding mismatch')
        ids=[x.id for x in [*self.text.texts,*self.asset.objects,*self.surface.shapes]]
        if len(set(ids))!=len(ids):raise ValueError('duplicate scene identity')
        if set(self.ownership)!=set(ids):raise ValueError('incomplete ownership')
        if any(r.source not in ids or r.target not in ids for r in self.relations):raise ValueError('dangling relation')
        return self


def mode(value='parallel_review'):
    if value not in ('parallel_review','parallel_shadow','parallel_auto'):raise ValueError('serial and unknown modes are not supported')
    if value=='parallel_auto':raise ValueError('parallel_auto release qualification has not passed')
    return value


def strategy(value='default'):
    if value in ('default','auto'):return 'auto'
    if value not in ('off','alpha','sam3','sam3.1'):raise ValueError('unknown asset strategy')
    return value
