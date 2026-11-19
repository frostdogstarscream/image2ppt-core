"""Exercise the three frozen workbench-08 edits in desktop PowerPoint, including recolor then delete."""

import argparse
import hashlib
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import win32com.client

from probe_fields import box, intersects, named_shapes, name_text, normalized

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'services/product'))
from product.resource_lease import lease

SIZE = (1672, 941)
COLOR = (112, 211, 165)
OFFICE_COLOR = COLOR[0] + 256 * COLOR[1] + 65536 * COLOR[2]


@contextmanager
def opened(app, path):
    presentation = app.Presentations.Open(str(path.resolve()), False, False, False)
    try:
        yield presentation
    finally:
        presentation.Close()


def select_target(presentation, case):
    _, label, target, action, _ = case
    matches = []
    for name, shape in named_shapes(presentation.Slides(1).Shapes):
        b = box(shape, presentation.PageSetup.SlideWidth, presentation.PageSetup.SlideHeight)
        overlap = intersects(b, target)
        ratio = overlap / max(1, b[2] * b[3] + target[2] * target[3] - overlap)
        right_type = (
            (action == 'fill' and shape.Type == 1)
            or (action == 'delete' and shape.Type == 13)
            or (action == 'text' and bool(name_text(shape)))
        )
        if right_type and ratio > (0.45 if action == 'text' else 0.65):
            matches.append((ratio, name, shape))
    return max(matches, default=None, key=lambda m: m[0])


def render(pres, path):
    pres.Slides(1).Export(str(path.resolve()), 'PNG', *SIZE)
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)


def region(image, coords, inset=0):
    x, y, w, h = map(round, coords)
    return image[y + inset : y + h - inset, x + inset : x + w - inset]


def changed_outside(a, b, target):
    diff = np.max(np.abs(a - b), axis=2) > 24
    x, y, w, h = map(round, target)
    diff[max(0, y - 5) : y + h + 5, max(0, x - 5) : x + w + 5] = False
    return int(diff.sum())


def probe(source, deck, out):
    row = next(
        r
        for r in json.loads((ROOT / 'services/experiments/native_edits/workbench.json').read_text('utf8'))[
            'pages'
        ]
        if r['id'] == 'workbench-08'
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Source differs from frozen edit cases')
    card, title, photo = row['cases']
    out.mkdir(parents=True, exist_ok=True)
    report = {
        'source_sha256': row['sha256'],
        'pptx_sha256': hashlib.sha256(deck.read_bytes()).hexdigest(),
        'card': {'status': 'NO_TARGET_OBJECT'},
        'title': {'status': 'NO_TARGET_OBJECT'},
        'delete_on_new_fill': {'status': 'PRECONDITION_FAILED'},
        'human_wps': 'UNKNOWN',
    }
    both = out / 'card-and-photo.pptx'
    text = out / 'title.pptx'
    shutil.copyfile(deck, both)
    shutil.copyfile(deck, text)
    app = win32com.client.DispatchEx('PowerPoint.Application')
    try:
        with opened(app, both) as pres:
            original = render(pres, out / 'before.png')
            target = select_target(pres, card)
            if target:
                _, card_name, shape = target
                shape.Fill.Solid()
                shape.Fill.ForeColor.RGB = OFFICE_COLOR
                pres.Save()
        if target:
            with opened(app, both) as pres:
                colored = render(pres, out / 'recolored.png')
                box_after = region(colored, card[2])
                colored_fraction = float((abs(box_after - COLOR).max(2) < 20).mean())
                outside = changed_outside(original, colored, card[2])
                photo_change = float(
                    (abs(region(colored, photo[2], 5) - region(original, photo[2], 5)).max(2) > 24).mean()
                )
                title_change = float(
                    (abs(region(colored, title[2], 3) - region(original, title[2], 3)).max(2) > 24).mean()
                )
                passed = (
                    pres.Slides(1).Shapes(card_name).Fill.ForeColor.RGB == OFFICE_COLOR
                    and colored_fraction > 0.025
                    and outside < 350
                    and max(photo_change, title_change) < 0.025
                )
                report['card'] = dict(
                    status='PASS' if passed else 'REVIEW_REQUIRED',
                    shape=card_name,
                    new_fill_fraction=colored_fraction,
                    outside=outside,
                    photo_changed=photo_change,
                    title_changed=title_change,
                )
                target_photo = select_target(pres, photo)
                if target_photo:
                    _, photo_name, shape = target_photo
                    shape.Delete()
                    pres.Save()
            if target_photo:
                with opened(app, both) as pres:
                    removed = render(pres, out / 'recolored-deleted.png')
                    revealed = float((abs(region(removed, photo[2], 5) - COLOR).max(2) < 24).mean())
                    outside = changed_outside(colored, removed, photo[2])
                    shape_gone = all(n != photo_name for n, _ in named_shapes(pres.Slides(1).Shapes))
                    passed = (
                        report['card']['status'] == 'PASS' and revealed > 0.90 and outside < 350 and shape_gone
                    )
                    report['delete_on_new_fill'] = dict(
                        status='PASS' if passed else 'REVIEW_REQUIRED',
                        photo=photo_name,
                        exposed_fill_fraction=revealed,
                        changed_outside_photo=outside,
                        shape_gone=shape_gone,
                    )
        with opened(app, text) as pres:
            target = select_target(pres, title)
            if target:
                _, text_name, shape = target
                initial = name_text(shape)
                shape.TextFrame.TextRange.Text = title[4]
                pres.Save()
        if target:
            with opened(app, text) as pres:
                edited = render(pres, out / 'title-edited.png')
                saved = normalized(name_text(pres.Slides(1).Shapes(text_name))) == normalized(title[4])
            # An extra copy exposes any old white caption still visible behind the text object.
            blank = out / 'title-removed.pptx'
            shutil.copyfile(text, blank)
            with opened(app, blank) as pres:
                pres.Slides(1).Shapes(text_name).Delete()
                pres.Save()
            with opened(app, blank) as pres:
                erased = render(pres, out / 'title-removed.png')
            cleared = region(erased, title[2], 2)
            old_ink = float((cleared.min(2) > 199).mean())
            outside = changed_outside(original, edited, title[2])
            passed = (
                saved
                and outside < 350
                and old_ink < 0.04
                and abs(region(edited, title[2]) - region(original, title[2])).max() > 24
            )
            report['title'] = dict(
                status='PASS' if passed else 'REVIEW_REQUIRED',
                object=text_name,
                initial=initial,
                matches_frozen_reading=normalized(initial) == normalized(title[1]),
                reopened_text=title[4] if saved else None,
                residual_light_fraction=old_ink,
                changed_outside=outside,
            )
    finally:
        if not app.Presentations.Count:
            app.Quit()
    report['operations_passed'] = sum(
        report[k]['status'] == 'PASS' for k in ('card', 'title', 'delete_on_new_fill')
    )
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), 'utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'deck', 'out'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    with lease('office'):
        print(json.dumps(probe(args.source, args.deck, args.out), ensure_ascii=True))
