"""Post-compile transaction gate. Any missing QA keeps the complete parent page."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import numpy as np
from PIL import Image,ImageDraw
from .materialize import digest,write
from .object_contracts import ObjectReview,OBJECT_REVIEW_PROMPT
from .object_registry import Registry,fingerprint
from .object_stage import effective_page


def object_pass(row,verdict):
    return (all(verdict.get(k)=='PASS' for k in ('pixels','complete_object','replacement')) and
        (not row.get('background_assets') or verdict.get('background_restoration')=='PASS') and
        (not row.get('source_alpha_hole_hypothesis') or verdict.get('white_content_assembly')=='PASS') and
        (not any(b.get('native_owner_scope_hypothesis') for b in row.get('background_assets',[])) or verdict.get('native_owner_scope')=='PASS'))


def review_objects(source,preview,irfile,report,directory,ask,*,review_model=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    staged=[r for r in report['objects'] if r['status']=='STAGED'];verdicts={}
    for start in range(0,len(staged),6):
        rows=staged[start:start+6];images=[source,preview]
        for r in rows:
            image=Image.open(Path(irfile).parent/r['asset']).convert('RGBA');w,h=image.size
            yy,xx=np.indices((h,w));rgb=np.repeat(np.where(((xx//12+yy//12)%2)[...,None],210,245).astype('uint8'),3,2)
            check=Image.fromarray(rgb).convert('RGBA');check.alpha_composite(image)
            panel=check.convert('RGB')
            edit=report.get('edit_evidence_directory')
            if edit:
                x,y,bw,bh=r['bbox'];padding=32;box=(max(0,x-padding),max(0,y-padding),x+bw+padding,y+bh+padding)
                delete=Path(edit)/(f"{report['edit_page']:02d}-extracted-{r['id']}-delete.png")
                before=Image.open(source).convert('RGB').crop(box);after=Image.open(delete).convert('RGB').crop(box)
                for bg in r.get('background_assets',[]):
                    ox,oy,ow,oh=bg['bbox'];oldbox=(ox-box[0],oy-box[1],ox+ow-box[0]-1,oy+oh-box[1]-1)
                    # Actual old scene bounds, not a candidate mask/ground truth.
                    # Context outside remains visible and MUST be preserved.
                    for part in (before,after):ImageDraw.Draw(part).rectangle(oldbox,outline=(220,20,20),width=1)
                # Equal context and explicit labels keep deletion evidence from
                # being mistaken for the candidate PNG (especially tiny icons).
                cy,cx=np.indices((before.height,before.width));checks=np.repeat(np.where(((cx//12+cy//12)%2)[...,None],210,245).astype('uint8'),3,2)
                framed=Image.fromarray(checks).convert('RGBA');framed.alpha_composite(image,(x-box[0],y-box[1]));panel=framed.convert('RGB')
                height=max(panel.height,before.height)+18;strip=Image.new('RGB',(panel.width+before.width+after.width,height),'white')
                offset=0
                for title,part in zip(('PNG / checker','SOURCE','AFTER DELETE'),(panel,before,after)):
                    strip.paste(part,(offset,18));ImageDraw.Draw(strip).text((offset+2,2),title,fill='black');offset+=part.width
                panel=strip
            file=directory/(r['id']+'-detail.png');panel.save(file);images.append(file)
        review_rows=[{'id':r['id'],'candidate_sha256':r['asset_sha256'],'bbox':r['bbox'],'replaces':r['replaces'],
            'definition':r.get('definition'),'method':r.get('method'),'background_assets':r.get('background_assets',[]),
            'source_alpha_hole_hypothesis':r.get('source_alpha_hole_hypothesis',False),
            'owner_proof':next((c['owner_proof'] for c in r['candidates'] if c['index']==r['selected_index']),{})} for r in rows]
        context={'source_sha256':digest(source),'scene_sha256':digest(irfile),'objects':review_rows,
            'binding_rule':'Copy ONLY the objects[i].candidate_sha256 values. These are the displayed PNG hashes. Do not substitute a raw mask/NPZ hash.',
            'white_content_rule':'For source_alpha_hole_hypothesis=true, explicitly return white_content_assembly=PASS only if restored source-layer pixels belong to real white/solid object content. Real transparent holes must remain holes. Unknown or indiscriminate filling must fail.',
            'native_owner_scope_rule':'If ANY background asset has native_owner_scope_hypothesis=true, independently inspect the ORIGINAL source over EVERY old owner bbox. Return native_owner_scope=PASS ONLY if ALL pixels in those old rectangles belong to this complete object OR its independently evidenced native page/card backing. Other icon strokes, independent text, connector/platform/label pixels, shadows belonging to neighbors, or any unknown content MUST FAIL or UNKNOWN. Those non-PNG pixels are an UNVERIFIED scope hypothesis, not known background. The after-delete view must also pass background_restoration. Missing this verdict blocks publication.',
            'detail_context_rule':'Red outlines show actual old-owner rectangles on SOURCE and AFTER DELETE. The detail image includes 32px of CONTEXT OUTSIDE those rectangles. Independent text/icons/connectors OUTSIDE red rectangles MUST remain and are NOT residuals. Judge backing/restoration only INSIDE old-owner rectangles, excluding the red annotation itself; compare context for preservation. These annotations are private review aids, never part of the PPT.',
            'image_order':['source','compiled_slide']+[r['id'] for r in rows]}
        cache=None
        if review_model and report.get('state_directory'):
            semantic=[{k:r.get(k) for k in ('id','definition','asset_sha256','replaces','bbox','method','source_alpha_hole_hypothesis')}
                for r in rows]
            key=fingerprint({'source':context['source_sha256'],'scene':context['scene_sha256'],'objects':semantic,
                'images':[digest(p) for p in images],'model':review_model,'prompt':OBJECT_REVIEW_PROMPT,'contract':ObjectReview.model_json_schema()})
            cache=Path(report['state_directory'])/'review-cache'/key;cache.mkdir(parents=True,exist_ok=True)
        if cache and (cache/'raw.json').exists():raw=json.loads((cache/'raw.json').read_text(encoding='utf-8'))
        else:
            if cache and (cache/'reserved.json').exists():raise RuntimeError('identical review already reserved; parent retained')
            if cache:write(cache/'reserved.json',{'state':'RESERVED','model':review_model,'key':key})
            raw=ask(context,images,ObjectReview)
            if cache:write(cache/'raw.json',raw)
        write(directory/f'review-{start:03d}-raw.json',raw)
        repair_file=None;review=ObjectReview.model_validate(raw)
        if review.source_sha256!=context['source_sha256'] or review.scene_sha256!=context['scene_sha256']:raise ValueError('object review binding mismatch')
        if len(review.objects)!=len(rows) or {r.id for r in review.objects}!={r['id'] for r in rows}:raise ValueError('incomplete object review')
        if any(v.candidate_sha256!=next(r['asset_sha256'] for r in rows if r['id']==v.id) for v in review.objects):
            # Historical input exposed NPZ hashes beside PNG hashes. Only this
            # proven within-object reference mistake is eligible for repair.
            for v in review.objects:
                row=next(r for r in rows if r['id']==v.id)
                if v.candidate_sha256!=row['asset_sha256'] and v.candidate_sha256 not in {c['raw_sha256'] for c in row['candidates']}:
                    raise ValueError('object review unknown PNG binding')
            expected=review.model_dump()
            for v in expected['objects']:v['candidate_sha256']=next(r['asset_sha256'] for r in rows if r['id']==v['id'])
            repair_file=(cache/'ref-raw.json') if cache else directory/f'binding-repair-{start:03d}-raw.json'
            if repair_file.exists():fixed=json.loads(repair_file.read_text(encoding='utf-8'))
            else:
                reserved=repair_file.with_name('ref-reserved.json')
                if reserved.exists():raise ValueError('binding repair already reserved')
                write(reserved,{'state':'RESERVED','original_response_sha256':fingerprint(raw)})
                fixed=ask({'task':'Reference-only repair. Return the same review, changing ONLY candidate_sha256 to the supplied PNG binding. Preserve IDs, order, decisions and reasons. No visual re-review.',
                    'source_sha256':context['source_sha256'],'scene_sha256':context['scene_sha256'],
                    'original_response':raw,'png_bindings':{r['id']:r['asset_sha256'] for r in rows}},[],ObjectReview)
                write(repair_file,fixed)
            repaired=ObjectReview.model_validate(fixed)
            if repaired.model_dump()!=expected:raise ValueError('binding repair changed visual decisions')
            write(directory/f'binding-repair-{start:03d}.json',{'adapter':'ObjectReviewBindingRepair/1',
                'original_response_sha256':fingerprint(raw),'repair_response_sha256':digest(repair_file),'visual_decisions_unchanged':True})
            review=repaired
        for v in review.objects:
            row=next(r for r in rows if r['id']==v.id)
            if v.candidate_sha256!=row['asset_sha256']:raise ValueError('object review PNG binding mismatch')
            verdicts[v.id]=v.model_dump()
        if cache:write(cache/'validated.json',{'source_sha256':context['source_sha256'],'scene_sha256':context['scene_sha256'],'raw_sha256':digest(cache/'raw.json'),
            'accepted_response_sha256':fingerprint(review.model_dump()),'binding_repair_sha256':digest(repair_file) if repair_file else None})
    return verdicts


def finalize(sources,irs,parents,reports,plans,output,compiler,ask,*,repair=None,reuse_office=None):
    """At most six assembly hypotheses; unresolved pages retain their parent.

    Compiler failures propagate, never becoming a quality pass. Go/service/Office
    failures roll back visual proposals before producing a safe deliverable.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    seen=[{digest(p)} for p in irs]
    for attempt in range(6):
        try:artifacts=compiler(irs,output/f'build-{attempt}',plans)
        except (RuntimeError,ValueError,OSError) as exc:
            if not any(r and any(o['status']=='STAGED' for o in r['objects']) for r in reports):raise
            write(output/f'compile-rollback-{attempt}.json',{'error_type':type(exc).__name__,'parent_preserved':True})
            for i,report in enumerate(reports):
                if report:
                    for row in report['objects']:
                        if row['status']=='STAGED':row.update(status='FALLBACK',reasons=['candidate_compile_failed'])
                    report['publication']='ROLLED_BACK'
                plans[i]=effective_page(plans[i],parents[i])
            return compiler(parents,output/'parent-compile',plans),parents,plans
        previews=sorted(Path(artifacts['preview']).glob('*.png'));verdicts={};failed=set()
        scenes=[{**(r or {'objects':[]}),'slide_ir':str(irs[i])} for i,r in enumerate(reports)];scenefile=output/f'scenes-{attempt}.json';write(scenefile,scenes)
        probe=output/f'office-{attempt}'
        try:
            cached=None
            if reuse_office:
                candidate=Path(reuse_office)
                old=json.loads((candidate/'report.json').read_text(encoding='utf-8'))
                expected={(i+1,'extracted-'+r['id'],action) for i,s in enumerate(scenes) for r in s['objects'] if r['status']=='STAGED' for action in ('move-scale','delete')}
                observed={(r['page'],name,r['action']) for r in old.get('records',[]) for name in r['object_ids']}
                media_audit=candidate.parent/'saved-media-audit-0.json'
                audit=json.loads(media_audit.read_text(encoding='utf-8')) if media_audit.exists() else {}
                if old.get('status')=='PASS_APP_AUTOMATED' and old.get('compiled_assets_checked') and old.get('owner_residual_checked') and old.get('pptx_sha256')==digest(artifacts['pptx']) and old.get('scene_sha256s')==[s.get('scene_sha256') for s in scenes] and observed==expected and audit.get('status')=='PASS':
                    if all(digest(candidate/f['file'])==f['sha256'] for f in audit['files']) and len(audit['files'])==len(expected)//2:
                        cached=candidate
                        write(output/f'office-cache-{attempt}.json',{'schema':'BoundOfficeProbeReuse/1','report_sha256':digest(candidate/'report.json'),
                            'media_audit_sha256':digest(media_audit),'pptx_sha256':digest(artifacts['pptx']),'exact_scene_and_target_match':True})
            if cached:probe=cached
            else:
                with (output/f'office-{attempt}.log').open('w') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'office_objects.py'),str(artifacts['pptx']),str(probe),
                        '--scenes',str(scenefile),'--per-object'],stdout=log,stderr=log,check=True,timeout=1800,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            actual=output/f'actual-preview-{attempt}';actual.mkdir()
            for i in range(len(plans)):shutil.copyfile(probe/f'page-{i+1:02d}-before.png',actual/f'page-{i+1:02d}.png')
            previews=sorted(actual.glob('*.png'));artifacts['preview']=str(actual)
            artifacts['powerpoint_checked']=True;artifacts['edit_probe']=str(probe/'report.json')
        except (OSError,ValueError,subprocess.SubprocessError):
            failed={i for i,r in enumerate(reports) if r and any(v['status']=='STAGED' for v in r['objects'])}
        for i,report in enumerate(reports):
            if i in failed or not report or not any(r['status']=='STAGED' for r in report['objects']):continue
            try:
                report['edit_evidence_directory']=str(probe);report['edit_page']=i+1
                verdicts[i]=ask(i,sources[i],previews[i],irs[i],report,output/f'review-{attempt}-{i:02d}')
                if any(not object_pass(r,verdicts[i][r['id']]) for r in report['objects'] if r['status']=='STAGED'):failed.add(i)
            except (ValueError,RuntimeError,OSError):failed.add(i)
        if not failed:
            try:
                for i,values in verdicts.items():
                    report=reports[i];evidence={}
                    for row in report['objects']:
                        if row['status']!='STAGED':continue
                        evidence[row['id']]={'pixels':'PASS','object':'PASS','replacement':'PASS','product':'PASS',
                            'scene_sha256':digest(irs[i]),'candidate_sha256':row['asset_sha256'],
                            'page':i+1,'review_verdict':values[row['id']],
                            **{k:{'path':str(p),'sha256':digest(p)} for k,p in {'pptx':artifacts['pptx'],'preview':previews[i],'edit_probe':probe/'report.json'}.items()}}
                    registry=Registry(Path(report['state_directory'])/'registry')
                    registry.publish(irs[i],report['objects'],evidence,expected_generation=registry.current()['generation'])
                    report['publication']='STABLE'
                    report['automatic_quality']='REVIEW_REQUIRED' if report.get('inventory_missing_owner_ids') or any(r['status']=='FALLBACK' for r in report['objects']) else 'AUTOMATED_PASS'
                return artifacts,irs,plans
            except (OSError,ValueError,subprocess.SubprocessError):
                failed={i for i,r in enumerate(reports) if r and any(v['status']=='STAGED' for v in r['objects'])}
        for i in failed:
            values=verdicts.get(i)
            report=reports[i]
            if values and attempt<5 and report.get('plan_file'):
                # Try the next existing assembly hypothesis as a WHOLE object.
                # No paid segmentation is repeated here and no partial owner
                # set can survive a failed member.
                from .object_closure import extract_closed
                offsets={r['id']:r['selected_index']+(0 if object_pass(r,values[r['id']]) else 1)
                    for r in report['objects'] if r['status']=='STAGED'}
                try:
                    ir,updated=extract_closed(report['plan_file'],report['layer_directory'],parents[i],output/f'candidate-{attempt}-{i:02d}',
                        sam_dir=report.get('sam_directory'),registry=Path(report['state_directory'])/'registry',candidate_offsets=offsets,rejected=report.get('rejected_objects'))
                    updated['state_directory']=report['state_directory'];updated['publication']='STAGED_ONLY'
                    exhausted=[k for k,v in values.items() if not object_pass(next(r for r in report['objects'] if r['id']==k),v) and
                        not any(r['id']==k and r['status']=='STAGED' for r in updated['objects'])]
                    if repair and exhausted:
                        try:
                            ri,rp,rr=repair(i,report,[values[k] for k in exhausted],output/f'moma-repair-{attempt}-{i:02d}')
                            if digest(ri) not in seen[i]:ir,updated,plans[i]=ri,rr,rp
                        except (ValueError,RuntimeError,OSError):pass
                    if digest(ir) in seen[i]:raise ValueError('object candidate cycle; retain parent')
                    seen[i].add(digest(ir))
                    reports[i]=updated;irs[i]=ir;plans[i]=effective_page(plans[i],ir)
                    continue
                except (ValueError,OSError):pass
            irs[i]=parents[i];plans[i]=effective_page(plans[i],parents[i])
            for row in reports[i]['objects']:
                if row['status']=='STAGED':row.update(status='FALLBACK',quality='REVIEW_REQUIRED',reasons=['post_compile_transaction_rejected'])
            reports[i]['publication']='ROLLED_BACK'
            write(output/f'rollback-{attempt}-{i:02d}.json',reports[i])
    # Final safe build is never published as a stable independent-object asset.
    return compiler(irs,output/'safe-build',plans),irs,plans
