"""Independent visual review of the five workbench-08 desktop PowerPoint editing trials."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/p) for p in ('services/product', 'services/planner/src', 'services/windows',
                                     'ppt-rebuild/src', 'services/vision/src')]
from product.materialize import digest, write

Verdict = Literal['PASS', 'FAIL', 'INSUFFICIENT_EVIDENCE']


class VisualReview(BaseModel):
    card: Verdict
    title: Verdict
    delete_on_new_fill: Verdict
    main_title: Verdict
    building_caption: Verdict
    observations: str


# The review panel is built from the sealed Office exports. Each contrast is an
# actual reopened edit; missing files are displayed explicitly, never simulated.
TRIALS = {
    'card': ('before.png', 'recolored.png'),
    'title': ('before.png', 'title-edited.png', 'title-removed.png'),
    'delete_on_new_fill': ('recolored.png', 'recolored-deleted.png'),
    'main_title': ('before.png', 'main-title/edited.png', 'main-title/without-text.png'),
    'building_caption': ('before.png', 'building-caption/edited.png', 'building-caption/without-text.png',
                         'building-caption/without-building.png'),
}
PANEL_SPEC = {
    'whole-slide': ((0,0,1672,941), 1, (('Whole slide: before edit','before.png'),
                                       ('Whole slide: after edit','recolored.png'))),
    'middle': ((568,161,1112,606), 2, (('source',None), ('exported','before.png'),
                ('card recolored','recolored.png'), ('then upper-left photo removed','recolored-deleted.png'),
                ('upper-right title edited','title-edited.png'), ('upper-right title erased','title-removed.png'))),
    'header': ((32,12,1408,108), 1, (('source',None), ('exported','before.png'),
                ('shorter main title','main-title/edited.png'), ('main title erased','main-title/without-text.png'))),
    'caption': ((573,233,1112,603), 2, (('source',None), ('exported','before.png'),
                ('upper-left caption edited','building-caption/edited.png'),
                ('upper-left caption erased','building-caption/without-text.png'),
                ('photo removed with edited caption retained','building-caption/without-building.png'))),
}
PROMPT = """Inspect these source and reopened PowerPoint images. Report each visible trial independently.
The first image shows the full reopened slides before and after the card edit at the same scale.
For effects outside the middle card, compare only these two full-slide views. Cropped panels provide
local detail only. If the full views are missing or unclear, report insufficient evidence for a
claim about other parts of the slide.
card: recolor the middle card mint; all photos and labels should remain in place.
title: replace the upper-right photo caption with 会议材料归档; the erased copy should show a clean strip.
delete_on_new_fill: remove the building after recolor; the photo area must show the new fill while its blue strip and neighboring photos survive.
main_title: shorten the top title to 应用展望 and also inspect its erased copy. Check for old strokes, page number, vertical divider, and underline damage.
building_caption: shorten the upper-left caption to 公文写作, erase the text on a separate copy, and inspect a copy with the building removed but the edited caption and its blue strip retained. Check neighboring photos.
Use INSUFFICIENT_EVIDENCE when an operation image is unavailable or illegible. Accept small font and gradient variations. Describe any visible defect in Chinese."""


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def prepare(source, deck, office, output):
    receipt = read(office/'region-report.json')
    reference = next(p for p in read(ROOT/'services/experiments/native_edits/workbench.json')['pages']
                     if p['id'] == 'workbench-08')
    if digest(source) != reference['sha256'] or digest(source) != receipt['source_sha256']:
        raise ValueError('Source differs from the evaluated workbench-08 image.')
    if digest(deck) != receipt['pptx_sha256'] or digest(office/'before.png') != receipt['before_render_sha256']:
        raise ValueError('Deck or before image differs from the Office receipt.')
    sealed = receipt['office_artifacts_sha256']
    required_frames = set(name for files in TRIALS.values() for name in files)
    for name in required_frames | {'report.json'}:
        path = office/name
        if path.is_file() != (name in sealed) or (path.is_file() and digest(path) != sealed[name]):
            raise ValueError('Missing, stale or unsealed PowerPoint artifact: '+name)
    old = read(office/'report.json')
    if old['pptx_sha256'] != receipt['pptx_sha256'] or old['source_sha256'] != reference['sha256']:
        raise ValueError('Three-edit and region-edit Office receipts use different inputs.')
    output.mkdir(parents=True, exist_ok=False)
    panels = []
    for key, (bounds, cols, frames) in PANEL_SPEC.items():
        width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
        board = Image.new('RGB',(cols*(width+4), ((len(frames)+cols-1)//cols)*(height+22)), 'white')
        draw = ImageDraw.Draw(board)
        for index, (label, name) in enumerate(frames):
            x, y = (index%cols)*(width+4), (index//cols)*(height+22)
            draw.text((x+2,y+2),label,fill='black')
            path = source if name is None else office/name
            if not path.is_file():
                draw.text((x+2,y+34),'No reopened Office image for this operation.',fill='black')
                continue
            with Image.open(path) as im:
                if im.size != (1672,941):
                    raise ValueError('Unexpected PowerPoint preview extent.')
                board.paste(im.convert('RGB').crop(bounds),(x,y+19))
        panel = output/(key+'-panel.png')
        board.save(panel)
        panels.append(panel)
    missing = {key:[f for f in files if f not in sealed] for key,files in TRIALS.items()}
    evidence = dict(source_sha256=reference['sha256'], pptx_sha256=receipt['pptx_sha256'],
                    office_receipt_sha256=digest(office/'region-report.json'),
                    prompt_sha256=hashlib.sha256(PROMPT.encode('utf8')).hexdigest(),
                    office_images={name:sealed[name] for name in sorted(required_frames) if name in sealed},
                    panels={p.name:digest(p) for p in panels}, missing_frames=missing)
    write(output/'review-inputs.json',evidence)
    return evidence, panels


def independent_review(source, deck, office, output, profile_path, prepare_only=False):
    evidence, panels = prepare(source, deck, office, output)
    if prepare_only:
        return 'PREPARED'
    from product.runtime.moma_review import MomaReviewer

    target = output/'independent-moma.json'
    record = {**evidence, 'status':'NOT_SENT'}
    try:
        profile = read(profile_path)
        if not os.environ.get('MOMA_API_KEY'):
            from credentials import crypt
            key_path = (Path(os.environ['LOCALAPPDATA'])/'image2ppt/credentials/native-editable-moma-free'
                        /profile['credential_set']/f"moma-{int(profile['default_key_slot'])}.dpapi")
            os.environ['MOMA_API_KEY'] = crypt(key_path.read_bytes(), True).decode('ascii').strip()
        actor = MomaReviewer(profile['model'])
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        record.update(status='UNKNOWN', provider_call='NOT_SENT', error_type=type(exc).__name__)
        write(target,record)
        return 'UNKNOWN'
    record.update(model_requested=actor.model, session=actor.session, status='RESERVED')
    write(target,record)
    try:
        answer = VisualReview.model_validate(actor.ask(PROMPT,{},panels,VisualReview)).model_dump()
    except Exception as exc:
        # The transport may already have consumed the call. Keep the unknown disposition
        # and do not retry. Exception text from HTTP transports can contain credentials.
        record.update(status='UNKNOWN', error_type=type(exc).__name__)
        write(target,record)
        return 'UNKNOWN'
    verdicts = {key:('INSUFFICIENT_EVIDENCE' if evidence['missing_frames'][key] else answer[key])
                for key in TRIALS}
    record.update(model=actor.reported_model, usage=actor.last_usage, answer=answer, verdicts=verdicts,
                  status='PASS' if all(v=='PASS' for v in verdicts.values()) else 'REVIEW_REQUIRED')
    write(target,record)
    return record['status']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','deck','office','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--profile',type=Path,default=ROOT/'.runtime/moma-profile.json')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    print(independent_review(args.source,args.deck,args.office,args.output,args.profile,args.prepare_only))
