"""Desktop PowerPoint acceptance for editable text regions on the workbench-08 development page."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import win32com.client

from probe_card_caption import opened, probe as probe_card, region, render, select_target
from probe_fields import name_text, named_shapes, normalized
from product.resource_lease import lease


# Evaluation coordinates on the frozen 1672 x 941 source; never sent to production planning.
HEADER = [1, '应用成效与未来展望', [158, 24, 398, 43], 'text', '应用展望']
CAPTION = [1, '公文写作辅助', [606, 376, 127, 25], 'text', '公文写作']
PHOTO = [1, '左上方的建筑照片', [588, 249, 247, 117], 'delete', None]
PAGE_NUMBER = [50, 23, 61, 47]
VERTICAL_RULE = [135, 21, 11, 51]
DIVIDER = [53, 88, 1320, 8]
BLUE_BAND = [591, 370, 240, 32]
NEIGHBORS = ([850, 251, 239, 149], [590, 429, 238, 153], [851, 429, 239, 153])


def native_shape(pres, name):
    return next((shape for key, shape in named_shapes(pres.Slides(1).Shapes) if key == name), None)


def changed_fraction(before, after, bbox):
    return float((abs(region(before,bbox) - region(after,bbox)).max(axis=2) > 24).mean())


def ink(region_rgb, light_text):
    red, _, blue = np.moveaxis(region_rgb, 2, 0)
    if light_text:
        return region_rgb.min(axis=2) > 180
    return (blue - red > 20) & (red < 150)


def backing_metrics(erased, source, case, light_text):
    # Measure visible lettering at the original viewing size. Large coherent remnants
    # must not be hidden by averaging the result over a long text frame.
    box = case[2]
    field = region(erased, box)
    original_ink = ink(region(source, box), light_text)
    remaining = ink(field, light_text)
    retained = float(remaining.sum() / max(1, original_ink.sum()))
    h, w = remaining.shape
    starts = [*range(0, w-h+1, max(1, h//2)), max(0, w-h)]
    peak = max(float(remaining[:, i:i+h].mean()) for i in starts)
    if light_text:
        red, green, blue = np.moveaxis(field, 2, 0)
        background = (blue - red > 25) & (green > red) & (red < 175)
    else:
        background = field.min(axis=2) > 175
    background_fraction = float(background.mean())
    return {'remaining_ink_vs_source': retained, 'peak_ink_fraction': peak,
            'expected_backing_fraction': background_fraction,
            'clean': retained < .06 and peak < .04 and background_fraction > .92}


def edit_field(app, deck, directory, case, before, source, *, building=False):
    directory.mkdir()
    edited_file = directory/'edited.pptx'
    shutil.copyfile(deck, edited_file)
    with opened(app, edited_file) as pres:
        target = select_target(pres, case)
        if target is None:
            return {'status':'NO_TARGET_OBJECT', 'expected':case[1]}
        _, name, shape = target
        initial = name_text(shape)
        shape.TextFrame.TextRange.Text = case[4]
        pres.Save()
    with opened(app, edited_file) as pres:
        new_shape = native_shape(pres, name)
        saved = name_text(new_shape) if new_shape is not None else None
        edited = render(pres, directory/'edited.png')

    removed_file = directory/'without-text.pptx'
    shutil.copyfile(edited_file, removed_file)
    with opened(app, removed_file) as pres:
        target = native_shape(pres, name)
        if target is None:
            return {'status':'REVIEW_REQUIRED', 'reason':'text shape disappeared after reopening'}
        target.Delete()
        pres.Save()
    with opened(app, removed_file) as pres:
        text_gone = native_shape(pres, name) is None
        erased = render(pres, directory/'without-text.png')
    backing = backing_metrics(erased, source, case, building)
    neighbors = [PHOTO[2], *NEIGHBORS] if building else [PAGE_NUMBER, VERTICAL_RULE, DIVIDER]
    neighbor_change = max(changed_fraction(before, step, b) for step in (edited, erased) for b in neighbors)
    initial_match = normalized(initial) == normalized(case[1])
    result = {'object':name, 'initial':initial, 'matches_reference':initial_match, 'reopened_text':saved,
              'backing':backing, 'max_neighbor_changed_fraction':neighbor_change,
              'text_object_removed':text_gone}
    passed = (normalized(saved or '') == normalized(case[4]) and changed_fraction(before, edited, case[2]) > .02
              and changed_fraction(edited, erased, case[2]) > .02 and backing['clean']
              and neighbor_change < .02 and text_gone)

    if not building:
        # An unchanged but already missing divider/page number is not a pass.
        # Compare the visible dark-blue areas to the source at normal size.
        for key, bounds in [('page_number', PAGE_NUMBER), ('title_vertical_rule', VERTICAL_RULE),
                            ('header_divider', DIVIDER)]:
            reference = ink(region(source, bounds), False).sum()
            visible = float(ink(region(erased, bounds), False).sum() / max(1, reference))
            result[key+'_relative_ink'] = visible
            passed = passed and .70 < visible < 1.30

    if building:
        # Start from the copy with the edited caption. Its independent textbox
        # must survive photo removal, and the original blue band must stay on the card.
        delete_photo_file = directory/'without-building.pptx'
        shutil.copyfile(edited_file, delete_photo_file)
        with opened(app, delete_photo_file) as pres:
            photo = select_target(pres, PHOTO)
            if photo is None:
                return {**result, 'status':'REVIEW_REQUIRED', 'reason':'independent building photograph missing'}
            _, photo_name, picture = photo
            picture.Delete()
            pres.Save()
        with opened(app, delete_photo_file) as pres:
            surviving = native_shape(pres, name)
            preserved_text = name_text(surviving) if surviving is not None else None
            photo_gone = native_shape(pres, photo_name) is None
            after = render(pres, directory/'without-building.png')
        band_change = changed_fraction(edited, after, BLUE_BAND)
        image_change = changed_fraction(edited, after, PHOTO[2])
        adjacent_change = max(changed_fraction(edited, after, b) for b in NEIGHBORS)
        result.update(photo_gone=photo_gone, caption_after_photo_delete=preserved_text,
                      changed_band_fraction=band_change, changed_photo_fraction=image_change,
                      changed_adjacent_fraction=adjacent_change)
        passed = (passed and photo_gone and normalized(preserved_text or '') == normalized(case[4])
                  and band_change < .02 and image_change > .30 and adjacent_change < .02)
    result['edit_status'] = 'PASS' if passed else 'REVIEW_REQUIRED'
    result['status'] = 'PASS' if passed and initial_match else 'REVIEW_REQUIRED'
    return result


def probe(source, deck, out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory to keep previous Office receipts.')
    legacy = probe_card(source, deck, out)
    with Image.open(out/'before.png') as im:
        before = np.asarray(im.convert('RGB'), dtype=np.int16)
    with Image.open(source) as im:
        original = np.asarray(im.convert('RGB'), dtype=np.int16)
    app = win32com.client.DispatchEx('PowerPoint.Application')
    try:
        header = edit_field(app, deck, out/'main-title', HEADER, before, original)
        caption = edit_field(app, deck, out/'building-caption', CAPTION, before, original, building=True)
    finally:
        if not app.Presentations.Count:
            app.Quit()
    tests = {'main_title':header, 'building_caption':caption,
             **{key:legacy[key] for key in ('card','title','delete_on_new_fill')}}
    passed = sum(case.get('edit_status',case['status']) == 'PASS' for case in tests.values())
    matches = {key:value.get('matches_reference', value.get('matches_frozen_reading'))
               for key,value in tests.items() if key in ('main_title','building_caption','title')}
    receipt = {'source_sha256':legacy['source_sha256'], 'pptx_sha256':legacy['pptx_sha256'],
               'before_render_sha256':hashlib.sha256((out/'before.png').read_bytes()).hexdigest(),
               'office_artifacts_sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(out.rglob('*')) if p.is_file()},
               'cases':tests, 'operations_passed':passed, 'operations_tested':len(tests),
               'initial_matches_reference':matches,
               'status':'PASS' if passed == len(tests) and all(matches.values()) else 'REVIEW_REQUIRED',
               'human_wps':'UNKNOWN'}
    (out/'region-report.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), 'utf8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('source','deck','out'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    with lease('office'):
        receipt = probe(args.source, args.deck, args.out)
    print(json.dumps({'status':receipt['status'], 'operations_passed':receipt['operations_passed'],
                      'operations_tested':receipt['operations_tested']}))
