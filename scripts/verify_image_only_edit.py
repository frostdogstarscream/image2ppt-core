"""Verify the current generated page in PowerPoint using disposable copies."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import win32com.client

from image_only_loop import ROOT, text_mask_evidence, write_json
from model_slideir_loop import digest
from verify_powerpoint_edit import shapes


def geometry(shape):
    return [shape.Left, shape.Top, shape.Width, shape.Height]


def verify(run_directory, output):
    state = json.loads((run_directory / 'result.json').read_text(encoding='utf-8'))
    ir = json.loads(Path(state['best_ir']).read_text(encoding='utf-8'))
    source = Path(state['best_build']['pptx'])
    source_hash = digest(source)
    output.mkdir(parents=True, exist_ok=False)
    edited = output / 'edit-check.pptx'
    no_text = output / 'no-native-text.pptx'
    no_images = output / 'no-independent-images.pptx'
    shutil.copy2(source, edited)
    shutil.copy2(source, no_text)
    shutil.copy2(source, no_images)
    app = win32com.client.DispatchEx('PowerPoint.Application')
    existing = app.Presentations.Count
    presentation = None
    result = {'source_pptx': str(source), 'source_sha256': source_hash,
              'human_acceptance': False, 'wps_checked': False,
              'application': 'Microsoft PowerPoint COM'}
    try:
        presentation = app.Presentations.Open(str(edited.resolve()), ReadOnly=False,
                                              Untitled=False, WithWindow=False)
        slide = presentation.Slides.Item(1)
        initial = shapes(slide)
        expected = {n['id'] for n in ir['nodes']}
        assert expected <= set(initial), 'IR object missing from final PowerPoint page'
        text_ids = [n['id'] for n in ir['nodes'] if n['type'] == 'text']
        image_ids = [n['id'] for n in ir['nodes'] if n['type'] == 'image']
        assert text_ids, 'no editable text'
        before = {name: geometry(initial[name]) for name in initial}
        edited_text = text_ids[0]
        original_text = initial[edited_text].TextFrame.TextRange.Text
        initial[edited_text].TextFrame.TextRange.Text = original_text + '【编辑验证】'
        movable = image_ids[0] if image_ids else text_ids[-1]
        initial[movable].Left += 12
        initial[movable].Width *= 1.1
        deletion_candidates = image_ids if len(image_ids) > 1 else text_ids
        deleted = next((name for name in reversed(deletion_candidates)
                       if name not in {edited_text, movable}), None)
        if deleted is not None:
            initial[deleted].Delete()
        presentation.Save()
        presentation.Close()
        presentation = app.Presentations.Open(str(edited.resolve()), ReadOnly=True,
                                              Untitled=False, WithWindow=False)
        after = shapes(presentation.Slides.Item(1))
        assert deleted is None or deleted not in after
        assert after[edited_text].TextFrame.TextRange.Text == original_text + '【编辑验证】'
        assert abs(after[movable].Left - before[movable][0] - 12) < .1
        assert abs(after[movable].Width - before[movable][2] * 1.1) < .1
        unchanged = set(initial) - {deleted, movable}
        for name in unchanged:
            assert all(abs(a-b) < .1 for a, b in zip(geometry(after[name]), before[name])), name
        presentation.Slides.Item(1).Export(str((output / 'edited-preview.png').resolve()),
                                          'PNG', *ir['canvas']['source_px'])
        presentation.Close()
        presentation = app.Presentations.Open(str(no_text.resolve()), ReadOnly=False,
                                              Untitled=False, WithWindow=False)
        slide = presentation.Slides.Item(1)
        for name in text_ids:
            shapes(slide)[name].Delete()
        presentation.Save()
        preview = output / 'no-native-text.png'
        slide.Export(str(preview.resolve()), 'PNG', *ir['canvas']['source_px'])
        presentation.Close()
        presentation = None
        sys.path.insert(0, str(ROOT / 'ppt-rebuild/src'))
        from background_models import precise_text_masks, inspect_residue
        original = np.asarray(Image.open(run_directory / 'source/page.png').convert('RGB'))
        actual = np.asarray(Image.open(preview).convert('RGB'))
        residuals = []
        text_footprint = np.zeros(original.shape[:2], bool)
        for node in ir['nodes']:
            if node['type'] != 'text':
                continue
            line, _ = text_mask_evidence(original, node)
            masks = precise_text_masks(original, line, np.zeros(original.shape[:2], bool))
            text_footprint |= masks['repair']
            residuals.append({'id': node['id'], **inspect_residue(
                original, actual, masks['core'], masks['inspection'])})
        presentation = app.Presentations.Open(str(no_images.resolve()), ReadOnly=False,
                                              Untitled=False, WithWindow=False)
        slide = presentation.Slides.Item(1)
        for name in image_ids:
            shapes(slide)[name].Delete()
        presentation.Save()
        image_preview = output / 'no-independent-images.png'
        slide.Export(str(image_preview.resolve()), 'PNG', *ir['canvas']['source_px'])
        presentation.Close()
        presentation = None
        import cv2
        rendered = np.asarray(Image.open(image_preview).convert('RGB'))
        source_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype('float32')
        target_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY).astype('float32')
        original_edges = np.abs(source_gray - cv2.GaussianBlur(source_gray, (0, 0), 2)) > 15
        remaining_edges = np.abs(target_gray - cv2.GaussianBlur(target_gray, (0, 0), 2)) > 6
        image_residuals = []
        for node in ir['nodes']:
            if node['type'] != 'image':
                continue
            asset = state['assets'][node['asset_id']]
            x, y, width, height = map(int, asset['bbox'])
            local = np.asarray(Image.open(run_directory / asset['path']).convert('RGBA'))
            reference = original[y:y+height, x:x+width]
            target = rendered[y:y+height, x:x+width]
            evidence = ((local[:, :, 3] > 128) & original_edges[y:y+height, x:x+width]
                        & ~text_footprint[y:y+height, x:x+width]
                        & (np.abs(local[:, :, :3].astype('int16') - reference.astype('int16')).max(axis=2) <= 6))
            count = int(evidence.sum())
            matches = (remaining_edges[y:y+height, x:x+width]
                       & (np.abs(target.astype('int16') - reference.astype('int16')).max(axis=2) <= 12))
            fraction = float(matches[evidence].mean()) if count else None
            image_residuals.append({'id': node['id'], 'checked_source_stroke_pixels': count,
                'source_stroke_matches_after_deletion_fraction': fraction,
                'status': 'UNKNOWN' if count < 60 else 'PASS' if fraction <= .01 else 'REVIEW_REQUIRED'})
        assert digest(source) == source_hash, 'original output changed'
        result.update(status='PASS', edited_text=edited_text, moved_and_scaled=movable,
                      deleted=deleted, sibling_geometries_checked=len(unchanged),
                      source_native_texts=len(text_ids), independent_images=len(image_ids),
                      saved_and_reopened=True, text_residual_checks=residuals,
                      text_residual_status='PASS' if all(r['status'] == 'PASS' for r in residuals)
                      else 'REVIEW_REQUIRED', edited_copy=str(edited), no_text_copy=str(no_text),
                      no_text_preview=str(preview), image_residual_checks=image_residuals,
                      image_residual_status='PASS' if all(r['status'] == 'PASS' for r in image_residuals)
                      else 'REVIEW_REQUIRED', no_images_copy=str(no_images), no_images_preview=str(image_preview))
    except Exception as exc:
        result.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
    finally:
        if presentation is not None:
            presentation.Close()
        if existing == 0 and app.Presentations.Count == 0:
            app.Quit()
    write_json(output / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = verify(arguments.run.resolve(), arguments.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
