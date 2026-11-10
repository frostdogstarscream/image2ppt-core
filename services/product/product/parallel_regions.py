"""Local replacement closures. Containment is a dependency, typography is not.

Foreground subregions can replace their ink while retaining a raster card.
A card replacement depends on all contained/intersecting foreground closures;
it cannot erase unknown ink, nor does its failure invalidate those subregions.
"""
from .parallel_store import key
from .contracts import intersection


def regions(graph):
    nodes={n.id:n for n in [*graph.text.texts,*graph.asset.objects,*graph.surface.shapes]}
    surfaces={s.id for s in graph.surface.shapes}
    parent={i:i for i in nodes}
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    # Only explicit resolved indivisible relationships merge transactions.
    for r in graph.relations:
        if r.status=='RESOLVED' and r.kind in ('part_of','occludes') and r.source not in surfaces and r.target not in surfaces:
            parent[root(r.source)]=root(r.target)
    groups={}
    for i in nodes:groups.setdefault(root(i),[]).append(i)
    records=[];by_id={}
    for members in groups.values():
        members.sort();boxes=[nodes[i].bbox for i in members]
        x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
        bbox=[x,y,max(b[0]+b[2] for b in boxes)-x,max(b[1]+b[3] for b in boxes)-y]
        rid='region-'+key({'source':graph.source_sha256,'members':members})[:20]
        record={'schema':'RegionTransaction/1','region_id':rid,'source_sha256':graph.source_sha256,
                'bbox':bbox,'member_ids':members,'dependencies':[],'owner_pixel_evidence':[],
                'status':'PENDING','fallback':'SOURCE_PLATE_LOCAL_PIXELS'}
        records.append(record)
        for i in members:by_id[i]=record
    aliases={r.source:r.target for r in graph.relations if r.kind=='part_of' and r.status=='RESOLVED' and r.target in surfaces}
    for s in graph.surface.shapes:
        # An intersecting rectangle is a conservative erasure veto, never a
        # reason to merge neighboring logical objects or discard their pixels.
        deps={by_id[n.id]['region_id'] for n in [*graph.text.texts,*graph.asset.objects]
              if intersection(s.bbox,n.bbox)>0 and aliases.get(n.id)!=s.id}
        by_id[s.id]['dependencies']=sorted(deps)
    return records,by_id


def retain_atomic(candidates, records):
    by_id={c['id']:c for c in candidates};rejected={}
    for r in records:
        present=set(r['member_ids'])&by_id.keys()
        if present and present!=set(r['member_ids']):
            for i in present:rejected[i]='REGION_MEMBER_FAILURE'
    return [c for c in candidates if c['id'] not in rejected],rejected


def finish_regions(records, report):
    staged={r['id']:r for r in report if r['state']=='STAGED'}
    fallback={r['id']:r for r in report if r['state']=='FALLBACK'}
    for r in records:
        r['status']='STAGED' if all(i in staged for i in r['member_ids']) else 'LOCAL_FALLBACK'
        r['owner_pixel_evidence']=[staged[i].get('mask_sha256') for i in r['member_ids'] if staged.get(i,{}).get('mask_sha256')]
        r['reasons']=[{'id':i,'code':fallback[i].get('reason_code'),'reason':fallback[i].get('reason')} for i in r['member_ids'] if i in fallback]
    return records
