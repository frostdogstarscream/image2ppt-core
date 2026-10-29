"""Bind the ordered deck to the selected pages and recheck its actual renders."""
from pathlib import Path

import numpy as np
from PIL import Image

from image2ppt.compiler import ArtifactBackend
from image2ppt.quality import inspect_pptx
from ..storage import digest
from .loop_checks import issue, measured_error


def compose_delivery(root, names, pages, *, project_root):
    from .quality_loop import read, write
    root = Path(root)
    result = {'schema': 'LayeredDelivery/1', 'status': 'FAILED', 'page_order': names,
              'human_acceptance': False, 'office_acceptance': False, 'issues': [], 'pages': []}
    if any(not pages[n].get('pptx') for n in names):
        result['issues'].append(issue('missing_page_artifact', confirmed=True, severity='blocking'))
        write(root/'delivery.json', result)
        return result
    try:
        irs = [root/'pages'/n/pages[n]['proposal']/'slideir.json' for n in names]
        backend = ArtifactBackend(root=project_root, timeout=max(600, 120*len(names)))
        composed = backend.invoke('core_bridge.mjs', 'compose', str(root/'delivery'), *map(str, irs))
        pptx = Path(composed['pptx']); inspection = inspect_pptx(pptx)
        manifest = read(composed['quality'])
        if inspection['slide_count'] != len(names) or len(manifest['pages']) != len(names):
            raise ValueError('composed page count mismatch')
        for index, (name, ir_file, record) in enumerate(zip(names, irs, manifest['pages'])):
            ir = read(ir_file)
            if record['ir_sha256'] != digest(ir_file) or record['source_sha256'] != ir['source_sha256']:
                raise ValueError('composed page order or source binding mismatch')
            objects = {o['name']: o for o in inspection['slides'][index]['objects']}
            expected = {n['id'] for n in ir['nodes']}
            if not expected <= objects.keys():
                raise ValueError('composed page lost objects: '+name)
            preview = pptx.parent/record['preview']
            with Image.open(preview) as im: rendered = np.asarray(im.convert('RGB'))
            with Image.open(root/'pages'/name/pages[name]['preview']) as im: single = np.asarray(im.convert('RGB'))
            error = measured_error(single, rendered, [])
            if error > .05:
                result['issues'].append(issue('composition_render_changed', target=name, difference=error))
            result['pages'].append({'page': name, 'slide': index+1, 'source_sha256': ir['source_sha256'],
                                    'ir_sha256': digest(ir_file), 'preview': str(preview.relative_to(root)),
                                    'preview_sha256': digest(preview), 'single_page_difference': error})
        result.update(pptx=str(pptx.relative_to(root)), pptx_sha256=digest(pptx), slide_count=len(names),
                      status='FAILED' if any(pages[n]['status'] == 'FAILED' for n in names) else 'REVIEW_REQUIRED',
                      preview_origin='reimported_final_pptx')
    except Exception as error:
        result['issues'].append(issue('composition_failed', confirmed=True, severity='blocking', detail=str(error)[:2000]))
    write(root/'delivery.json', result)
    return result
