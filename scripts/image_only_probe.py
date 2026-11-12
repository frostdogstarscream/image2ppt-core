"""Render an actual PPTX copy with its native text objects removed."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile


def without_native_text(source, target, text_ids):
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    ns = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
    removed = []
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as result:
        for info in original.infolist():
            data = original.read(info.filename)
            if info.filename.startswith('ppt/slides/slide') and info.filename.endswith('.xml'):
                root = ET.fromstring(data)
                tree = root.find('.//p:spTree', ns)
                if tree is not None:
                    for shape in list(tree):
                        props = shape.find('./p:nvSpPr/p:cNvPr', ns)
                        if props is not None and props.get('name') in text_ids:
                            removed.append(props.get('name'))
                            tree.remove(shape)
                data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            result.writestr(info, data)
    if set(removed) != set(text_ids):
        raise ValueError('native-text diagnostic could not locate every expected text object')
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError('original PPTX changed during probe')
    return {'source_pptx_sha256': before, 'removed_text_ids': removed}


def render_probe(pptx, ir, folder, root, node, source=None, preview=None):
    target = folder / 'no-native-text.pptx'
    report = without_native_text(pptx, target, {n['id'] for n in ir['nodes'] if n['type'] == 'text'})
    script = "import {renderPptx} from './scripts/render.mjs'; console.log(JSON.stringify(await renderPptx(process.argv[1],process.argv[2])));"
    run = subprocess.run([node, '--input-type=module', '-e', script, str(target), str(folder / 'no-text-render')],
                         cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=1800)
    if run.returncode:
        raise ValueError((run.stderr or run.stdout)[-3000:])
    report['render'] = json.loads(run.stdout[run.stdout.find('{'):])
    if source is not None:
        import sys
        import numpy as np
        from PIL import Image
        from image_only_loop import text_mask_evidence
        sys.path.insert(0, str(root / 'ppt-rebuild/src'))
        from background_models import precise_text_masks, inspect_residue
        original = np.asarray(Image.open(source).convert('RGB'))
        with Image.open(report['render']['pages'][0]['path']) as image:
            actual = np.asarray(image.convert('RGB').resize((original.shape[1], original.shape[0])))
        residuals = []
        for text in ir['nodes']:
            if text['type'] != 'text':
                continue
            line, _ = text_mask_evidence(original, text)
            masks = precise_text_masks(original, line, np.zeros(original.shape[:2], bool))
            residuals.append({'id': text['id'], **inspect_residue(
                original, actual, masks['core'], masks['inspection'])})
        report['text_residuals'] = residuals
        report['text_residual_status'] = 'PASS' if all(r['status'] == 'PASS' for r in residuals) else 'REVIEW_REQUIRED'
        if preview is not None:
            with Image.open(preview) as image:
                full = image.convert('RGB').resize((original.shape[1], original.shape[0]))
            nodes = {n['id']: n for n in ir['nodes']}
            failed = sorted((r for r in residuals if r['status'] != 'PASS'),
                            key=lambda r: r.get('residual_fraction', 0) * r.get('checked_pixels', 0), reverse=True)
            comparisons = []
            for index, residual in enumerate(failed[:8]):
                text = nodes[residual['id']]
                x, y, width, height = text.get('source_text_bbox', text['bbox'])
                box = (max(0, int(x)-12), max(0, int(y)-12),
                       min(original.shape[1], int(x+width)+12), min(original.shape[0], int(y+height)+12))
                parts = [Image.fromarray(original).crop(box), full.crop(box), Image.fromarray(actual).crop(box)]
                comparison = Image.new('RGB', (parts[0].width * 3, parts[0].height), 'white')
                for column, part in enumerate(parts):
                    comparison.paste(part, (column * part.width, 0))
                path = folder / f'no-text-comparison-{index:02}.png'
                comparison.save(path)
                comparisons.append({'id': text['id'], 'path': str(path),
                                    'columns': ['original', 'final PPTX', 'same PPTX without native text']})
            report['comparison_crops'] = comparisons
    return report
