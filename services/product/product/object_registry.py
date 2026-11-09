"""Immutable assets plus compare-and-swap scene publication after four gates."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from .materialize import digest,write

ALGORITHM='visual-object-closure/2'
SAM_BINDING={'version':'sam3','checkpoint_sha256':'9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',
    'code_revision':'f1647e8d0c1d280991f4d1b7d974c2704537fe66'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def pixel_key(source,layers,obj,model=None):
    # Identity, prose and old owner names do not affect extracted pixels.
    parts=lambda values:sorted([{'bbox':p.bbox,'layers':sorted(p.layers)} for p in values],key=fingerprint)
    return fingerprint({'source':source,'layers':layers,'bbox':obj.bbox,'parts':parts(obj.parts),
        'exclusions':parts(obj.exclusions),'text_policy':obj.text_policy,'intrinsic_text':obj.intrinsic_text,
        'algorithm':ALGORITHM,'model':model,'sam_prompt':obj.prompt_en if model else None})


def scene_key(ir):
    return fingerprint({'nodes':ir['nodes'],'groups':ir.get('groups',[]),'background':ir.get('background'),
        'assets':[{k:a.get(k) for k in ('id','sha256')} for a in ir['assets']]})


class Registry:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.pointer=self.root/'stable.json'

    def current(self):
        return json.loads(self.pointer.read_text(encoding='utf-8')) if self.pointer.exists() else {'generation':0,'objects':{}}

    def asset(self,entry):
        file=self.root/'assets'/(entry['sha256']+'.png')
        if digest(file)!=entry['sha256']:raise ValueError('stable asset hash mismatch')
        return file

    def publish(self,staged_ir,records,evidence,*,expected_generation):
        """One pointer update publishes the entire object set; failures keep parent."""
        lock=(self.root/'publish.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            lock.close();raise ValueError('object transaction writer busy') from None
        try:return self._publish(staged_ir,records,evidence,expected_generation=expected_generation)
        finally:lock.close()

    def _publish(self,staged_ir,records,evidence,*,expected_generation):
        parent=self.current()
        if parent['generation']!=expected_generation:raise ValueError('stale object transaction')
        staged_ir=Path(staged_ir);scene=digest(staged_ir)
        required=('pixels','object','replacement','product')
        for r in records:
            if r.get('status')!='STAGED':continue
            e=evidence.get(r['id'],{})
            if any(e.get(g)!='PASS' for g in required):raise ValueError('incomplete object gates: '+r['id'])
            if e.get('scene_sha256')!=scene or e.get('candidate_sha256')!=r['asset_sha256']:
                raise ValueError('object gate binding mismatch')
            for key in ('pptx','preview','edit_probe'):
                item=e.get(key,{})
                if not item.get('path') or digest(item['path'])!=item.get('sha256'):raise ValueError('product evidence missing or changed')
            probe=json.loads(Path(e['edit_probe']['path']).read_text(encoding='utf-8'))
            if probe.get('status')!='PASS_APP_AUTOMATED' or not probe.get('owner_residual_checked') or not probe.get('compiled_assets_checked'):
                raise ValueError('unverified edit/residual probe')
            if probe.get('pptx_sha256')!=e['pptx']['sha256'] or scene not in probe.get('scene_sha256s',[]):
                raise ValueError('edit probe belongs to another product')
            for action in ('move-scale','delete'):
                if not any(p.get('page')==e.get('page') and p.get('action')==action and 'extracted-'+r['id'] in p.get('object_ids',[])
                           for p in probe.get('records',[])):raise ValueError('object missing from edit probe')
            verdict=e.get('review_verdict',{})
            if verdict.get('candidate_sha256')!=r['asset_sha256'] or any(verdict.get(k)!='PASS' for k in ('pixels','complete_object','replacement')):
                raise ValueError('independent review evidence incomplete')
            if r.get('background_assets') and verdict.get('background_restoration')!='PASS':raise ValueError('background restoration unverified')
            if r.get('source_alpha_hole_hypothesis') and verdict.get('white_content_assembly')!='PASS':raise ValueError('white content ownership unverified')
            if any(b.get('native_owner_scope_hypothesis') for b in r.get('background_assets',[])) and verdict.get('native_owner_scope')!='PASS':
                raise ValueError('old crop scope ownership unverified')
        updated=copy.deepcopy(parent)
        for r in records:
            if r.get('status')!='STAGED':continue
            src=staged_ir.parent/r['asset']
            if digest(src)!=r['asset_sha256']:raise ValueError('staged asset changed')
            target=self.root/'assets'/(r['asset_sha256']+'.png');target.parent.mkdir(exist_ok=True)
            if target.exists():
                if digest(target)!=r['asset_sha256']:raise ValueError('immutable asset corrupted')
            else:shutil.copyfile(src,target)
            for bg in r.get('background_assets',[]):
                file=staged_ir.parent/bg['file'];destination=self.root/'assets'/(bg['sha256']+'.png')
                if digest(file)!=bg['sha256']:raise ValueError('background asset changed')
                if destination.exists():
                    if digest(destination)!=bg['sha256']:raise ValueError('stable background corrupted')
                else:shutil.copyfile(file,destination)
            updated['objects'][r['pixel_key']]={'sha256':r['asset_sha256'],'bbox':r['bbox'],'object_id':r['id'],
                'owners':r['replaces'],'scene_key':r['scene_key'],'quality':'AUTOMATED_PASS','human_acceptance':False,
                'background_assets':r.get('background_assets',[]),
                'source_alpha_hole_hypothesis':r.get('source_alpha_hole_hypothesis',False),
                'evidence':evidence[r['id']]}
        # Last check immediately before atomic replace; caller serializes job writes.
        if self.current()['generation']!=expected_generation:raise ValueError('concurrent object commit')
        updated.update(generation=expected_generation+1,scene_sha256=scene)
        write(self.pointer,updated)
        return updated

    def lookup(self,key):
        value=self.current()['objects'].get(key)
        if value:self.asset(value)  # corruption never silently falls through
        return value


def asset_refs(irfile):
    """Internal sidecar only. Public VisualPage and SlideIR remain unchanged."""
    irfile=Path(irfile);ir=json.loads(irfile.read_text(encoding='utf-8'));assets={a['id']:a for a in ir['assets']}
    return {n['id']:{'path':str((irfile.parent/assets[n['asset_id']]['path']).resolve()),
        'sha256':assets[n['asset_id']]['sha256'],'bbox':n['bbox'],'source_sha256':ir['source_sha256']}
        for n in ir['nodes'] if n['type']=='image' and n['id'].startswith(('extracted-','restored-bg-'))}


def pin_graphics(page,previous,records):
    stable=[r for r in records if r['status']=='STAGED']
    ids={'extracted-'+r['id'] for r in stable}|{b['id'] for r in stable for b in r.get('background_assets',[])};owners={i for r in stable for i in r['replaces']}
    pinned=[n for n in previous.crops if n.id in ids]
    if len(pinned)!=len(ids):raise ValueError('stable graphic definition missing')
    return page.model_copy(update={'crops':[n for n in page.crops if n.id not in ids|owners]+pinned,
        'shapes':[n for n in page.shapes if n.id not in owners]})
