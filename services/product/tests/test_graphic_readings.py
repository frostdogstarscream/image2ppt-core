from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from product.materialize import digest
from product.runtime.graphic_readings import bind_graphic_readings
from product.runtime.host_baseline import delivery_defects


def test_fragmentary_ocr_requires_bound_alpha_and_solid_delete_on_both_parts(tmp_path, monkeypatch):
    matte = Image.new('L', (40, 60), 0)
    draw = ImageDraw.Draw(matte)
    draw.ellipse((10, 3, 27, 20), outline=255, width=3)
    draw.arc((4, 27, 32, 56), 145, 335, fill=255, width=4)
    paper = Image.new('RGB', (80, 80), 'white')
    source = paper.copy()
    source.paste((40, 80, 170), (11, 8, 51, 68), matte)
    source.save(tmp_path / 'source.png')
    hidden = tmp_path / 'page-01-text-hidden.png'
    source.save(hidden)
    delete = tmp_path / '01-icon-delete.png'
    paper.save(delete)
    rgba = Image.new('RGBA', matte.size, (40, 80, 170))
    rgba.putalpha(matte)
    rgba.save(tmp_path / 'ink.png')
    mask = np.zeros((80, 80), bool)
    mask[8:68, 11:51] = np.array(matte) > 0
    box = [11, 8, 40, 60]
    np.savez(tmp_path / 'owner.npz', mask=mask, restore_bbox=box)
    sha = digest(tmp_path / 'source.png')
    proof = digest(tmp_path / 'owner.npz')
    scene = {
        'asset': {'objects': [{'id': 'icon', 'kind': 'icon'}]},
        'selected': {
            'icon': {
                'bbox': box,
                'source_sha256': sha,
                'move_safety': 'MOVABLE',
                'complete_owner_scope': True,
                'badge_evidence_ids': ['detector-icon', 'measured-disc'],
                'asset_reference': {
                    'path': 'ink.png',
                    'sha256': digest(tmp_path / 'ink.png'),
                    'source_sha256': sha,
                },
                'owner_evidence': {'file': 'owner.npz', 'sha256': proof, 'method': 'measured-badge-ink'},
            }
        },
    }
    receipt = {
        'id': 'icon',
        'source_sha256': sha,
        'reference_sha256': proof,
        'delete_sha256': digest(delete),
        'residual_check_status': 'PASS',
        'checked_owner_pixels': int(mask.sum()),
        'residual_pixels_confirmed': 0,
    }
    parts = [
        {'ocr_id': None, 'status': 'plate_only', 'seen_on_named_plate': False, 'text': word, 'bbox': b}
        for word, b in [('O', [20, 10, 21, 27]), ('C', [14, 34, 31, 32])]
    ]
    rereads = []
    monkeypatch.setattr(
        'product.runtime.graphic_readings.source_reading',
        lambda *_a, **_k: {'crop_sha256': 'crop-hash', 'readings': rereads},
    )

    def run(rows=None, lines=(), res=None):
        audit = {
            'status': 'MEASURED',
            'rows': deepcopy(parts if rows is None else rows),
            'source_sha256': sha,
            'text_hidden_sha256': digest(hidden),
        }
        bind_graphic_readings(
            audit,
            job=tmp_path,
            page=1,
            scene=scene,
            backing=tmp_path,
            source_path=tmp_path / 'source.png',
            hidden_path=hidden,
            lines=lines,
            editable_ids=['icon'],
            residuals=[receipt if res is None else res],
            directory=tmp_path,
        )
        defects = delivery_defects(
            {'verified': True, 'pages': [{'page': 1, 'selected_editables': ['icon'], 'plate_audit': audit}]}
        )
        return [r['status'] for r in audit['rows']], bool(defects)

    assert run() == (['intrinsic_graphic', 'intrinsic_graphic'], False)
    assert run(parts[:1]) == (['plate_only'], True)
    assert run([{**parts[0],'bbox':[20,10,21,21]},parts[1]]) == (['plate_only','plate_only'], True)
    assert run(lines=[{'text': 'O', 'bbox': parts[0]['bbox']}]) == (['plate_only', 'plate_only'], True)
    for text in ('0', '3%', 'O/C', '万元'):
        assert run([{**parts[0], 'text': text}, parts[1]]) == (['plate_only', 'plate_only'], True)
    assert run([{**parts[0], 'seen_on_named_plate': True}, parts[1]]) == (['plate_only', 'plate_only'], True)
    assert run(res={**receipt, 'delete_sha256': 'wrong'}) == (['plate_only', 'plate_only'], True)
    rereads.append({'text': 'C', 'score': 0.99})
    assert run() == (['plate_only', 'plate_only'], True)
    rereads.clear()
    rgba.putalpha(Image.new('L', matte.size, 0))
    rgba.save(tmp_path / 'ink.png')
    scene['selected']['icon']['asset_reference']['sha256'] = digest(tmp_path / 'ink.png')
    assert run() == (['plate_only', 'plate_only'], True)
