"""Explicit branch-level adapter for consistently misdeclared asset boxes."""
from .parallel_contracts import AssetEvidence
from .parallel_store import key
from .materialize import digest


def normalize(evidence, store):
    """Interpret a coherent xyxy reply without granting pixel ownership.

    At least two impossible xywh object boxes and a completely consistent xyxy
    inventory are required. Mixed/ambiguous replies retain their original boxes.
    The raw response stays immutable and every change is bound in an adapter
    receipt. Fusion and source-derived candidate gates must still decide safety.
    """
    width,height=evidence.image_size
    def xywh(box):
        x,y,w,h=box
        return min(x,y)>=0 and min(w,h)>0 and x+w<=width and y+h<=height
    def xyxy(box):
        left,top,right,bottom=box
        return 0<=left<right<=width and 0<=top<bottom<=height
    invalid=[obj.id for obj in evidence.objects if not xywh(obj.bbox)]
    boxes=[box for obj in evidence.objects for box in
           [obj.bbox,*[p.bbox for p in obj.parts],*[p.bbox for p in obj.exclusions]]]
    report={'schema':'AssetCoordinateAdapter/1','source_sha256':evidence.source_sha256,
            'adapter_sha256':digest(__file__),
            'input_evidence_sha256':key(evidence.model_dump()),'invalid_xywh_objects':invalid,
            'all_boxes_valid_xyxy':bool(boxes) and all(xyxy(box) for box in boxes),
            'pixel_ownership_granted':False,'changes':[],'status':'UNCHANGED'}
    if len(invalid)<2 or not report['all_boxes_valid_xyxy']:
        return evidence,report
    result=evidence.model_copy(deep=True)
    for obj in result.objects:
        for kind,item in [('object',obj),*[('part',p) for p in obj.parts],*[('exclusion',p) for p in obj.exclusions]]:
            left,top,right,bottom=item.bbox
            after=[left,top,right-left,bottom-top]
            report['changes'].append({'object_id':obj.id,'kind':kind,'before':item.bbox,'after':after})
            item.bbox=after
    report.update(status='ADAPTED_XYXY_TO_XYWH',
                  reason='multiple out-of-canvas xywh proposals and a consistent whole-branch xyxy inventory',
                  adapted_geometry_sha256=key([obj.model_dump() for obj in result.objects]))
    result.references.append(store.evidence(report))
    return AssetEvidence.model_validate(result.model_dump()),report
