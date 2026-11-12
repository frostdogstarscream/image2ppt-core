"""Draft computation and measurements, independent of delivery approval."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..parallel_store import key
from .primitives import PrimitiveTools


class ExplorationTools(PrimitiveTools):
    def complete_text_mask(self, state, current, directory, *, page=None, object_id=None,
                           evidence_id=None, original_mask_id=None, background_id=None,
                           protected_id=None, bbox=None, glyph_height=None):
        if page is None or object_id is None:
            return super().complete_text_mask(state, current, directory, evidence_id=evidence_id,
                original_mask_id=original_mask_id, background_id=background_id,
                protected_id=protected_id, bbox=bbox, glyph_height=glyph_height)
        if not background_id:
            return {'status': 'MISSING_INPUT', 'required': ['background_id'],
                    'reason': 'Choose explicit background evidence for stroke comparison.'}
        row = self.page(current, page)
        graph = self.graph(row)
        text = next(t for t in graph.text.texts if t.id == object_id)
        style = next(s for s in graph.text.styles if s.id == text.style)
        source = np.array(Image.open(row['snapshot']['path']).convert('RGB'))
        background = self.image(state, background_id)
        if source.shape != background.shape:
            raise ValueError('background dimensions differ from source')
        from ..parallel_fusion import glyph_evidence
        from ..parallel_text_candidates import peer_protection
        from ..recovery.text_mask_completion import TextMaskCompletion
        x, y, w, h = map(round, text.bbox)
        box = [max(0,x-3), max(0,y-3), min(source.shape[1],x+w+3)-max(0,x-3),
               min(source.shape[0],y+h+3)-max(0,y-3)]
        protected = peer_protection(text, graph, source, np.zeros(source.shape[:2],bool),
                                   baseline_directory=Path(row['ir']).parent)
        if original_mask_id:
            mask_path, mask_reference = state.evidence(original_mask_id)
            with Image.open(mask_path) as mask_image:
                pixels=np.array(mask_image)
                if mask_image.mode not in ('1','L') or not np.isin(pixels,[0,1,255]).all():
                    raise ValueError('original_mask_id must reference a binary grayscale Mask, not a background or source image. In scene mode omit original_mask_id to derive strokes from source and background_id.')
            original = self.image(state, original_mask_id, 'L') > 0
        else:
            bx, by, bw, bh = box
            ink, _ = glyph_evidence(source, box, style.color, background[by:by+bh,bx:bx+bw])
            original = np.zeros(source.shape[:2],bool)
            original[by:by+bh,bx:bx+bw] = ink
        excluded_seed_pixels=int((original & protected).sum())
        if not original_mask_id:
            original &= ~protected  # derived hypotheses must never seed known neighbor pixels
        if not original.any():
            report={'derived_seed_protected_pixels_excluded':excluded_seed_pixels,'remaining_seed_pixels':0}
            write(directory/'report.json',report)
            return {'status':'NO_UNPROTECTED_SEED','masks':{},'report':report,'ownership_authority':False}
        completion = TextMaskCompletion.analyze(source, original, protected, box, background,
                                                glyph_height=style.size_px)
        refs = {}
        for name, pixels in {**completion.evidence, **completion.masks}.items():
            file = directory/(name+'.png')
            Image.fromarray(pixels.astype('uint8')*255).save(file)
            refs[name] = self.artifact(state,file,'mask_candidate')
        completion.report['derived_seed_protected_pixels_excluded']=excluded_seed_pixels if not original_mask_id else 0
        write(directory/'report.json',completion.report)
        return {'status':'COMPUTED','masks':refs,'report':completion.report,'ownership_authority':False}

    def validate_mask(self, state, current, directory, *, mask_id, page=None,
                      object_id=None, evidence_id=None, protected_id=None, reference_mask_id=None):
        mask = self.image(state, mask_id, 'L')
        source = self.image(state, evidence_id) if evidence_id else np.array(
            Image.open(self.page(current, page)['snapshot']['path']).convert('RGB'))
        if mask.shape != source.shape[:2]:
            raise ValueError('mask dimensions differ from source')
        wanted = mask > 0
        measurements = {'binary': bool(np.isin(mask, [0, 255]).all()),
                        'mask_pixels': int(wanted.sum())}
        for identifier, name in ((protected_id, 'protected_overlap_pixels'),
                                 (reference_mask_id, 'outside_reference_pixels')):
            if identifier:
                reference = self.image(state, identifier, 'L') > 0
                if reference.shape != mask.shape:
                    raise ValueError('comparison mask dimensions differ')
                overlap = wanted & (reference if name.startswith('protected') else ~reference)
                measurements[name] = int(overlap.sum())
        return {'status': 'MEASURED', 'measurements': measurements,
                'ownership_authority': False,
                'uncertainties': ['Pixel ownership requires independent evidence at submission.']}

    def create_candidate(self, state, current, directory, *, page, object_id, mask_id,
                         background_id, frame_minimum=None, layout_id=None):
        """Construct a private preview through the existing scene adapter."""
        row = self.page(current, page)
        graph = self.graph(row).model_copy(deep=True)
        text = next((t for t in graph.text.texts if t.id == object_id), None)
        if text is None:
            raise ValueError('scene text object required')
        mask = self.image(state, mask_id, 'L') > 0
        background = self.image(state, background_id)
        if mask.shape != tuple(reversed(graph.image_size)) or background.shape != (*mask.shape, 3):
            raise ValueError('source-sized mask and RGB background required')
        fingerprint = key({'scene': digest(row['scene']), 'parent': digest(row['ir']),
            'mask': state.evidence(mask_id)[1]['sha256'], 'background': state.evidence(background_id)[1]['sha256'],
            'object': object_id, 'frame': frame_minimum, 'layout': layout_id, 'code': self.code_binding()})
        for prior in current['candidates'].values():
            if prior.get('fingerprint') == fingerprint:
                self.check_candidate(current, prior)
                return {**prior, 'cache_hit': True}
        limit = json.loads((state.job/'request.json').read_text('utf8')).get('adaptive_budget', {}).get('complex_region_candidates', 4)
        if sum(c['page'] == page and c['object_id'] == object_id for c in current['candidates'].values()) >= limit:
            return {'status': 'RESOURCE_LIMIT', 'reason': 'candidate budget exhausted'}
        folder = directory/'preview'
        shutil.copytree(Path(row['ir']).parent, folder)
        plate = np.array(Image.open(folder/graph.raster_backing['path']).convert('RGBA'))
        plate[..., :3][mask] = background[mask]
        Image.fromarray(plate).save(folder/graph.raster_backing['path'])
        graph.raster_backing['sha256'] = digest(folder/graph.raster_backing['path'])
        if layout_id:
            path, _ = state.evidence(layout_id)
            proposal = json.loads(path.read_text('utf8'))
            if proposal['id'] != text.id or proposal['text'] != text.text:
                raise ValueError('layout changes object identity or content')
            if proposal['bbox'][:2] != text.bbox[:2]:
                raise ValueError('update the scene to propose a different origin before fitting a layout')
            text.bbox = proposal['bbox']
        elif frame_minimum:
            text.bbox = [*text.bbox[:2], max(text.bbox[2], frame_minimum[0]), max(text.bbox[3], frame_minimum[1])]
        graph.selected[object_id] = {'id': object_id, 'kind': 'text', 'bbox': text.bbox,
            'state': 'EXPLORATORY', 'payload': text.model_dump(), 'quality': 'UNVERIFIED'}
        from ..parallel_rebuild import adapt
        write(folder/'scene.json', graph.model_dump())
        write(folder/'slideir.json', adapt(graph, folder))
        Image.fromarray(mask.astype('uint8')*255).save(directory/'mask.png')
        candidate = {'id': 'c-'+key(str(directory.relative_to(state.job)))[:24],
            'page': page, 'object_id': object_id, 'kind': 'text', 'status': 'EXPLORATORY',
            'exploratory': True, 'ir': str(folder/'slideir.json'),
            'parent_ir_sha256': digest(row['ir']), 'parent_scene_sha256': digest(row['scene']),
            'code': self.code_binding(), 'fingerprint': fingerprint,
            'proposal': {'page': page, 'object_id': object_id, 'mask_id': mask_id,
                         'background_id': background_id, 'frame_minimum': text.bbox[2:]},
            'files': {str(p): digest(p) for p in folder.rglob('*') if p.is_file()},
            'evidence': [self.artifact(state, folder/graph.raster_backing['path'], 'exploratory_backing')],
            'ownership_authority': False}
        current['candidates'][candidate['id']] = candidate
        return candidate

    def unmix_foreground(self, state, current, directory, *, page=None, object_id=None,
                         evidence_id=None, background_id=None, mask_id=None, **options):
        if not (evidence_id and background_id and mask_id):
            return {'status': 'MISSING_INPUT', 'required': ['evidence_id', 'background_id', 'mask_id'],
                    'reason': 'Source Unmix takes explicit source, backing and segmentation evidence.'}
        from reconstruction import extract_rgba
        source = self.image(state, evidence_id)
        background = self.image(state, background_id)
        mask = self.image(state, mask_id, 'L') > 0
        if source.shape != background.shape or mask.shape != source.shape[:2]:
            raise ValueError('unmix input dimensions differ')
        rgba = extract_rgba(source, background, mask)
        Image.fromarray(rgba).save(directory/'rgba.png')
        return {'status': 'COMPUTED', 'image': self.artifact(state, directory/'rgba.png', 'unverified_rgba'),
                'ownership_authority': False}

    def validate_rgba(self, state, current, directory, *, evidence_id=None, candidate_id=None,
                      source_id=None, background_id=None, protected_id=None):
        if not evidence_id:
            return {'status': 'MISSING_INPUT', 'required': ['evidence_id'],
                    'reason': 'Supply the actual RGBA image evidence to measure.'}
        path, _ = state.evidence(evidence_id)
        with Image.open(path) as image:
            has_alpha = 'A' in image.getbands()
            rgba = np.array(image.convert('RGBA'))
        alpha = rgba[..., 3:4].astype(float)/255
        values = {'has_alpha_channel': has_alpha, 'visible_pixels': int((alpha > 0).sum()),
                  'soft_alpha_pixels': int(((alpha > 0) & (alpha < 1)).sum())}
        if source_id and background_id:
            source, background = self.image(state, source_id), self.image(state, background_id)
            if source.shape != rgba[..., :3].shape or background.shape != source.shape:
                raise ValueError('recomposition dimensions differ')
            recomposed = rgba[..., :3]*alpha + background*(1-alpha)
            error = np.abs(recomposed-source)
            values.update(recomposition_mae=float(error.mean()), recomposition_p95=float(np.percentile(error, 95)))
            Image.fromarray(np.uint8(np.clip(recomposed, 0, 255))).save(directory/'recomposed.png')
        if protected_id:
            protected = self.image(state, protected_id, 'L') > 0
            if protected.shape != rgba.shape[:2]:
                raise ValueError('protection dimensions differ')
            values['protected_overlap_pixels'] = int(((alpha[..., 0] > 0) & protected).sum())
        return {'status': 'MEASURED', 'measurements': values, 'ownership_authority': False,
                'uncertainties': ['Object completeness requires source and component evidence.'],
                'evidence': [self.artifact(state, p, 'recomposition') for p in directory.glob('*.png')]}

    def record_observation(self, state, current, directory, *, evidence_ids, findings, next_action=None):
        references = [state.evidence(identifier)[1] for identifier in evidence_ids]
        record = {'evidence': references, 'findings': findings, 'next_action': next_action,
                  'scene_version': current['version'], 'author': 'agent', 'delivery_approval': False}
        write(directory/'observation.json', record)
        return {'status': 'RECORDED', 'evidence': self.artifact(state, directory/'observation.json', 'agent_observation')}
