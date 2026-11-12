"""Original screenshot input and a persistent visual model tool loop."""
from __future__ import annotations

import argparse
import copy
import getpass
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import requests
from PIL import Image

from model_slideir_loop import compiler, digest, image_data, rgb_page, seal, FORMAT_RULES
from moma_page_analysis import retry_after_seconds
from model_slideir_loop import patch
from image_only_typography import fit_model_text
from image_only_assets import restore_dependencies, dependency_conflicts, overlapping_ink, remove_text_plan
from image_only_probe import render_probe
from model_slideir_loop import node_cmd

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_FILES = {name: digest(ROOT / 'scripts' / name) for name in
                  ('image_only_loop.py', 'image_only_typography.py', 'image_only_assets.py', 'image_only_probe.py')}
EXECUTOR_SHA256 = hashlib.sha256(json.dumps(EXECUTOR_FILES, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """Commit each receipt atomically so interruptions retain a readable version."""
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


SYSTEM = """You reconstruct an original screenshot as editable PPTX. You own analysis,
planning and local corrections. Preserve wording, layout, inline bold/color and stacking.
Return one JSON action per turn with action and arguments. Available actions:
ocr {} obtains local text evidence.
inventory {objects:[{id,role,bbox,decision:'asset'|'backing'|'native',asset_id?,node_id?,reason?}]}
records your logical visual-object plan. Identify source icons, logos, photos,
panels and compound UI modules. Counts and regions come from this source. Update
individual IDs when revising decisions. Backing decisions need an uncertainty
reason. After OCR, create this inventory, then rebuild both text AND visual objects.
draft {texts:[{id:'ocr-0',color:'RRGGBB',bold:false,size_pt:24,runs:[{text,color,bold}],
  text?:corrected wording,bbox?:[x,y,w,h]}],objects:[asset IDs]} creates a full valid
SlideIR deterministically from YOUR selected OCR IDs and style/geometry decisions.
All texts must be explicitly selected. It returns the current IR and its path.
OCR is evidence, correct recognition mistakes using the source. Logo intrinsic letters
can remain raster, exclude them from native texts. Include every other real text line.
Objects must be extracted first. Select them by asset ID in draft. Native shapes may
be added with shapes:[complete SlideIR shape nodes] to draft. Use color in all text
entries, with runs for inline color/bold. Defaults use OCR geometry and 0.75*bbox height
points at source-pixel canvas, Microsoft YaHei. Override when the source needs it.
patch {patches:[{id,changes:{bbox,font,color,bold,runs,...}}],review_notes:[]} updates
the current IR locally. After changing text geometry/style, create a new plate.
patch additionally accepts source_text_bbox to correct the ORIGINAL glyph search
region. bbox changes only the editable text layout. Expanding bbox alone does
not expand the cleanup footprint. Use source_text_bbox for omitted glyph fringes.
fit {} uses existing source-stroke font measurement on the current IR and its
cleaned plate. It preserves your text and emphasis, returning geometry diagnostics.
view {bbox:[x,y,w,h],target?:'source'|'plate'|'preview'} returns an actual crop.
remove_text {ids:[node IDs],reason:string} returns uncertain or false-positive OCR
nodes to raster ownership in the backing. Keep real text native whenever possible.
For intrinsic UI screenshot lettering, remove_text can instead use
{within_inventory:logical_object_id,reason:string}. This returns native text fully
inside YOUR inventory region to raster ownership and deselects stale cleaned crops.
Regenerate plate then re-extract those modules. Standalone body text stays native.
extract_many {objects:[{id,bbox,mode,background_color?,from?},...]} extracts a
model-defined batch with independent per-object receipts. Prefer it for repeated
icons and photos. Extract identifiable logos, icons, photos and backing panels
as independently movable assets or native shapes. Use raster backing for the
remaining background and genuinely uncertain visual content.
extract {id,bbox:[x,y,w,h],mode:'rectangle'|'flat',background_color?:'FFFFFF'}
saves an independent object. Flat mode uses the given background color to make
transparent foreground pixels. Rectangle mode is for opaque photos. Objects must
be tightly bounded and contain no text selected for native reconstruction.
The full screenshot and text-bearing panels are not extractable objects. Keep
backing panels in the cleaned plate, or represent them as native shapes. Obtain
the first draft/plate/compile before extracting a long list of objects.
extract can set from:'plate' to crop from the current cleaned plate, allowing
an independently movable panel behind native text. First draft the texts with
objects:[], generate plate, then extract the text-free art from that plate.
The plate also removes SELECTED image objects. Before re-extracting a parent,
use objects to deselect that parent while keeping its children selected, then
regenerate plate. This preserves the parent's original artwork while clearing
child pixels. Never crop an image from a plate where that image was erased.
objects {ids:[asset IDs]} replaces ONLY the selected image nodes, preserving text
and shape nodes. Use this to remove rejected overlapping objects or add new
cleaned-plate crops, then plate and compile again.
For local revisions prefer objects {add_ids:[...],remove_ids:[...]}, preserving
every unrelated selected object. Replacing a few photos must retain selected icons.
An explicit objects.ids list defines image stacking from bottom to top. Put a
repaired parent below its independent children, while retaining unrelated IDs.
plate {ir?:SlideIR} uses existing text stroke masking and inpainting to create a
cleaned full-page backing for the text nodes in the supplied IR. Inspect it before
compiling. Omitting ir uses current IR. It also clears extracted objects selected
by draft. The tool updates background.asset_id and assets automatically.
The plate step deterministically calibrates native text with existing source-ink
font measurement, and shrinks measured single-line overflow. Text, emphasis and
colors remain your decisions. Read typography diagnostics and use compile {}.
Plate diagnostics include measured source ink colors, independent of your chosen
font color. Use them and source crops to preserve real inline emphasis and color.
compile {ir?:SlideIR} validates, compiles and returns the actual PPTX preview.
finish {issues:[strings],visual_status:'pass'|'needs_review'} ends after inspecting a compiled preview.
Use native text and shapes. Images may preserve uncertain visual content. Never place
native text over the same source text pixels. You can reject or amend OCR evidence.
No fixed object count, color, IDs or layout is required. Coordinates always use
ORIGINAL SOURCE PIXELS [x,y,width,height], also after viewing a crop. For source
W,H require x+w<=W,y+h<=H. Never repeat the same failed action. After OCR and at most
four useful crops, proceed to draft, plate, compile, then inspect and patch locally.
After the first compile, fulfill the visual inventory with extract_many and objects.
Do not repeatedly polish a single font issue while visual objects remain unhandled.
Small font/italic differences are secondary to complete content, cleanup and editing.
For nested visual assets, extract leaf objects from the text-clean plate first.
Select those leaves and generate a plate that removes them, while parents remain
deselected. Extract parent backing from that plate, then select parents below
their children in objects.ids. This preserves artwork without duplicating children.
finish requires your honest remaining issues. Do not execute code. """ + FORMAT_RULES + (
    " These SlideIR field rules apply only inside ir values. Always return an action JSON. "
    "Prefer draft instead of spelling out full SlideIR. For source text missed by OCR, "
    "draft texts may contain a new id, text, bbox, color and style. "
    "Compile returns the actual final preview followed by an actual-PPTX no-native-text preview. "
    "Failed-region comparison crops follow, each with ORIGINAL, FINAL and NO-NATIVE-TEXT columns. "
    "Inspect both and the no-text residual diagnostics. A fresh successful compile is required "
    "after every resume before finish. Ghosts hidden by native text still fail cleanup. Operational overlap errors "
    "are repairable. Preserve recognizable icons. A parent extracted after clearing children "
    "requires those children selected ABOVE it. To retain a child in backing, re-extract its "
    "parent from a fresh plate with that child deselected. Never drop children and leave holes.")


def local_ocr(source, models):
    import numpy as np
    from rapidocr import RapidOCR
    engine = RapidOCR(params={"Det.model_path": str(models / "det.onnx"),
                              "Rec.model_path": str(models / "rec.onnx"),
                              "Cls.model_path": str(models / "cls.onnx"),
                              "Global.log_level": "error"})
    result = engine(np.asarray(Image.open(source).convert("RGB"))[:, :, ::-1].copy())
    rows = zip(result.boxes, result.txts, result.scores) if result.boxes is not None else []
    evidence = []
    for index, (polygon, text, confidence) in enumerate(rows):
        xs, ys = zip(*polygon.tolist())
        evidence.append({"id": f"ocr-{index}", "kind": "text", "text": text,
                         "bbox": [float(min(xs)), float(min(ys)), float(max(xs)-min(xs)),
                                  float(max(ys)-min(ys))], "confidence": float(confidence)})
    return evidence


def safe_box(box, size):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("bbox must be [x,y,width,height]")
    x, y, width, height = map(int, box)
    if min(x, y) < 0 or min(width, height) <= 0 or x+width > size[0] or y+height > size[1]:
        raise ValueError(f"bbox {box} outside source {size}: use [x,y,width,height], "
                         f"require x+width <= {size[0]}, y+height <= {size[1]}")
    return x, y, x+width, y+height


def decode_action(content):
    """Normalize an unambiguous JSON envelope without changing model decisions."""
    if not isinstance(content, str):
        raise ValueError('model action content must be JSON text')
    cleaned = content.strip()
    if cleaned.startswith('```'):
        cleaned = '\n'.join(cleaned.splitlines()[1:-1])
    action = json.loads(cleaned)
    if not isinstance(action, dict) or not isinstance(action.get('action'), str):
        raise ValueError('return {"action":"tool name","arguments":{...}}')
    if 'arguments' in action:
        if set(action) - {'action', 'arguments'}:
            raise ValueError('put all tool parameters inside arguments, without duplicate top-level parameters')
        arguments = action['arguments']
    else:
        arguments = {key: value for key, value in action.items() if key != 'action'}
    if not isinstance(arguments, dict):
        raise ValueError('arguments must be an object')
    return action['action'], arguments


def draft_ir(arguments, state, source, ocr):
    """Expand a model-selected plan without making page-content decisions."""
    evidence = {row['id']: row for row in ocr}
    selected = set()
    ir = {'schema_version': '0.1', 'source_image': 'source/page.png',
          'source_sha256': digest(source), 'canvas': {'source_px': state['size'],
          'target_px': state['size']}, 'background': {'color': 'FFFFFF'},
          'review_status': 'needs_review', 'review_notes': [], 'evidence': [],
          'assets': [], 'groups': [], 'nodes': []}
    for item in arguments['texts']:
        identifier = item['id']
        if 'color' not in item:
            raise ValueError(f'text {identifier} requires explicit color as six RGB hex digits')
        if identifier in selected:
            raise ValueError('duplicate text selection: ' + identifier)
        selected.add(identifier)
        if identifier in evidence:
            row = copy.deepcopy(evidence[identifier])
        else:
            row = {'id': identifier, 'kind': 'text', 'text': item['text'], 'bbox': item['bbox']}
        row.pop('confidence', None)
        row['text'] = item.get('text', row['text'])
        row['bbox'] = item.get('bbox', row['bbox'])
        safe_box(row['bbox'], state['size'])
        ir['evidence'].append(row)
        node = {'id': 'text-' + identifier, 'type': 'text', 'text': row['text'],
                'bbox': row['bbox'], 'source_text_bbox': row['bbox'],
                'evidence_ids': [identifier], 'z': len(ir['nodes']) + 100,
                'font': {'latin': 'Arial', 'ea': item.get('font_ea', 'Microsoft YaHei'),
                         'size_pt': item.get('size_pt', row['bbox'][3] * .75)},
                'color': item['color'], 'bold': item.get('bold', False),
                'wrap': False, 'insets': [0, 0, 0, 0], 'vertical_anchor': 'middle'}
        for key in ('runs', 'paragraphs', 'align', 'italic'):
            if key in item:
                node[key] = item[key]
        ir['nodes'].append(node)
    for index, identifier in enumerate(arguments.get('objects', [])):
        if isinstance(identifier, dict):
            identifier = identifier.get('asset_id', identifier.get('id'))
        if identifier not in state['assets']:
            raise ValueError(f'asset {identifier} has not been extracted. Use extract_many, '
                             'or use objects:[] for the first text draft and add assets later.')
        asset = state['assets'][identifier]
        ir['assets'].append({k: asset[k] for k in ('id', 'path', 'sha256') } |
                            {'purpose': 'module', 'contains_evidence_ids': []})
        ir['nodes'].append({'id': 'object-' + identifier, 'type': 'image',
                            'bbox': asset['bbox'], 'asset_id': identifier,
                            'evidence_ids': [], 'z': index})
    ir['nodes'].extend(arguments.get('shapes', []))
    for identifier, row in evidence.items():
        if identifier not in selected:
            record = copy.deepcopy(row)
            record.pop('confidence', None)
            ir['evidence'].append(record)
    return ir


def source_ink_colors(pixels, bbox, text_color):
    """Measure source stroke colors without treating output font color as evidence."""
    import cv2
    import numpy as np
    x0, y0, x1, y1 = safe_box(bbox, [pixels.shape[1], pixels.shape[0]])
    crop = pixels[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype('float32')
    contrast = gray - cv2.GaussianBlur(gray, (0, 0), max(1, crop.shape[0] / 10))
    background = np.median(crop.reshape(-1, 3), axis=0)
    chosen = np.array([int(text_color[i:i+2], 16) for i in (0, 2, 4)])
    polarity = 1 if chosen.mean() > background.mean() else -1
    foreground = ((contrast * polarity > 20) & ((gray - np.median(gray)) * polarity > 35)
                  & (np.abs(crop.astype('float32') - background).max(axis=2) > 100))
    samples = crop[foreground]
    if not len(samples):
        return []
    bins = samples // 32
    keys = bins[:, 0].astype('int32') * 64 + bins[:, 1] * 8 + bins[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    ranked = sorted(zip(unique, counts), key=lambda pair: -pair[1])[:8]
    return [{'color': ''.join(f'{int(c):02X}' for c in np.median(samples[keys == key], axis=0)),
             'pixels': int(count)} for key, count in ranked if count >= 3]


def text_mask_evidence(pixels, node):
    """Keep modeled colors and add distinct, measured source emphasis colors."""
    line = {'bbox': node.get('source_text_bbox', node['bbox']),
            'style': {'color': node['color']}, 'runs': list(node.get('runs', []))}
    palette = source_ink_colors(pixels, line['bbox'], node['color'])
    import numpy as np
    colors = [node['color'], *[r['color'] for r in line['runs'] if r.get('color')]]
    x0, y0, x1, y1 = safe_box(line['bbox'], [pixels.shape[1], pixels.shape[0]])
    background = np.median(pixels[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    chosen = np.array([int(node['color'][i:i+2], 16) for i in (0, 2, 4)])
    chosen_contrast = np.abs(chosen - background).max()
    for entry in palette:
        candidate = [int(entry['color'][i:i+2], 16) for i in (0, 2, 4)]
        distances = [max(abs(channel - int(color[i:i+2], 16))
                        for channel, i in zip(candidate, (0, 2, 4))) for color in colors]
        source_contrast = np.abs(np.array(candidate) - background).max()
        if min(distances) > 65 and source_contrast >= chosen_contrast * .8:
            line['runs'].append({'color': entry['color']})
            colors.append(entry['color'])
    return line, palette


def build_clean_plate(ir, source, target, state, output):
    """Reuse bounded color-stroke masks, retaining pixels outside their union."""
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT / 'ppt-rebuild/src'))
    from background_models import precise_text_masks, inspect_residue, fit_surface, gradient_pixels
    pixels = np.asarray(Image.open(source).convert('RGB'))
    protected = np.zeros(pixels.shape[:2], bool)
    masks = []
    details = []
    palettes = []
    for node in ir['nodes']:
        if node['type'] != 'text':
            continue
        line, palette = text_mask_evidence(pixels, node)
        palettes.append({'id': node['id'], 'colors': palette})
        detail = precise_text_masks(pixels, line, protected)
        if detail['status'] != 'PASS':
            raise ValueError(f"no safe source glyph mask for {node['id']}, source bbox {line['bbox']}, "
                             f"color {node['color']}. View this source region, correct color or "
                             'source_text_bbox, or remove_text if the OCR is actually an icon.')
        # Complete the interiors of thick color-consistent strokes connected to
        # the measured glyph cores. The repair remains bounded by pixel support.
        colors = {node['color'], *[r['color'] for r in line['runs'] if r.get('color')]}
        distance = np.minimum.reduce([np.abs(pixels.astype('int16') -
            np.array([int(color[i:i+2], 16) for i in (0, 2, 4)])).max(axis=2) for color in colors])
        candidates = (distance <= 100) & detail['inspection']
        _, labels = cv2.connectedComponents(candidates.astype('uint8'), 8)
        touched = np.unique(labels[detail['core']])
        touched = touched[touched != 0]
        completed = np.isin(labels, touched) & candidates
        if completed.sum() <= detail['inspection'].sum() * .8:
            detail['repair'] |= cv2.dilate(completed.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool) & detail['inspection']
        masks.append(detail['repair'])
        details.append((node['id'], detail))
    for node in ir['nodes']:
        if node['type'] == 'image':
            asset = state['assets'][node['asset_id']]
            box = safe_box(asset['bbox'], state['size'])
            local = np.asarray(Image.open(output / asset['path']).convert('RGBA'))
            original = pixels[box[1]:box[3], box[0]:box[2]].astype('int16')
            source_matches = np.abs(local[:, :, :3].astype('int16') - original).max(axis=2) <= 6
            overlap = []
            for identifier, detail in details:
                core = detail['core'][box[1]:box[3], box[0]:box[2]] & (local[:, :, 3] > 128)
                count = int((core & source_matches).sum())
                if count >= 12 and count > int(core.sum()) * .01:
                    overlap.append(identifier)
            if overlap:
                raise ValueError('extracted object ' + node['id'] + ' retains original glyph strokes for native text ' +
                    ','.join(overlap) + '. Use objects with ids excluding this asset, then plate. '
                    'To preserve its artwork independently, re-extract from the cleaned plate.')
            local = local[:, :, 3] > 0
            mask = np.zeros(pixels.shape[:2], bool)
            mask[box[1]:box[3], box[0]:box[2]] = local
            masks.append(mask)
    if not masks:
        raise ValueError('select native text or extracted objects before plate')
    text_union = np.logical_or.reduce(masks[:len(details)]) if details else protected.copy()
    sample_exclusion = cv2.dilate(text_union.astype('uint8'), np.ones((5, 5), 'uint8')).astype(bool)
    image_union = np.logical_or.reduce(masks[len(details):]) if len(masks) > len(details) else protected.copy()
    planes = []
    for index, (identifier, detail) in enumerate(details):
        yy, xx = np.nonzero(detail['inspection'])
        box = [int(xx.min()), int(yy.min()), int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)]
        try:
            model, quality = fit_surface(pixels, box, ~sample_exclusion)
        except ValueError:
            continue
        if quality['mae'] > 2 or quality['p95'] > 5:
            continue
        x, y, width, height = box
        background = gradient_pixels(model, box)
        foreground = np.abs(pixels[y:y+height, x:x+width].astype('int16') - background.astype('int16')).max(axis=2) > 3
        grown = cv2.dilate(detail['repair'].astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
        detail['repair'][y:y+height, x:x+width] |= grown[y:y+height, x:x+width] & foreground
        masks[index] = detail['repair']
        planes.append((identifier, box, background, detail['repair'], quality))
    union = np.logical_or.reduce(masks)
    repaired = cv2.inpaint(pixels, union.astype('uint8') * 255, 4, cv2.INPAINT_TELEA)
    clean = pixels.copy()
    clean[union] = repaired[union]
    for _, box, background, repair_mask, _ in planes:
        x, y, width, height = box
        authorized = repair_mask[y:y+height, x:x+width] & ~image_union[y:y+height, x:x+width]
        clean[y:y+height, x:x+width][authorized] = background[authorized]
    Image.fromarray(clean).save(target)
    Image.fromarray(union.astype('uint8') * 255).save(target.with_suffix('.mask.png'))
    return {'outside_mask_changed_pixels': int(np.any(clean[~union] != pixels[~union], axis=1).sum()),
            'mask_pixels': int(union.sum()), 'source_ink_colors': palettes,
            'source_fitted_text_backgrounds': [{'id': identifier, 'bbox': box, 'quality': quality}
                                              for identifier, box, _, _, quality in planes],
            'text_residuals': [
                {'id': identifier, **inspect_residue(pixels, clean, detail['core'], detail['inspection'])}
                for identifier, detail in details]}


def extract_object(arguments, state, source, output, folder):
    """Extract one model-selected region and retain source-bound cleanup evidence."""
    identifier = arguments['id']
    if not isinstance(identifier, str) or not identifier.replace('-', '').replace('_', '').isalnum():
        raise ValueError('asset id must be alphanumeric')
    if identifier in state['assets']:
        raise ValueError('asset id already exists, choose a new revision id')
    box = safe_box(arguments['bbox'], state['size'])
    if list(box) == [0, 0, *state['size']]:
        raise ValueError('full screenshot cannot be an extracted object, use plate')
    crop_source = source
    cleaned_ids = []
    cleaned_boxes = {}
    cleared_ids = []
    if arguments.get('from') == 'plate':
        if state.get('plate_plan') != state.get('current_ir'):
            raise ValueError('generate plate for current text plan before extracting from plate')
        aid = state['current_ir']['background']['asset_id']
        cleared_ids = state['assets'][aid].get('cleared_asset_ids', [])
        crop_source = output / state['assets'][aid]['path']
        import numpy as np
        erased = np.zeros((box[3]-box[1], box[2]-box[0]), bool)
        for node in state['current_ir']['nodes']:
            if node['type'] != 'image':
                continue
            other = state['assets'][node['asset_id']]
            bounds = safe_box(other['bbox'], state['size'])
            x0, y0 = max(box[0], bounds[0]), max(box[1], bounds[1])
            x1, y1 = min(box[2], bounds[2]), min(box[3], bounds[3])
            if x1 <= x0 or y1 <= y0:
                continue
            with Image.open(output / other['path']) as image:
                alpha = np.asarray(image.convert('RGBA'))[:, :, 3] > 0
            erased[y0-box[1]:y1-box[1], x0-box[0]:x1-box[0]] |= alpha[
                y0-bounds[1]:y1-bounds[1], x0-bounds[0]:x1-bounds[0]]
        if erased.mean() > .5:
            raise ValueError('most of the requested artwork was already removed from this plate. '
                'Use objects to deselect the parent being extracted, keep its child objects selected, '
                'generate a fresh plate, then extract from that plate. Alternatively crop original source.')
        cleaned_ids = [r['id'] for r in state['assets'][aid]['inspection']['text_residuals']
                       if r['status'] == 'PASS']
        cleaned_boxes = {n['id']: n.get('source_text_bbox', n['bbox']) for n in state['current_ir']['nodes']
                         if n['id'] in cleaned_ids}
    with Image.open(crop_source) as image:
        crop = image.crop(box).convert('RGBA')
    if arguments.get('mode', 'rectangle') == 'flat':
        import numpy as np
        color = arguments.get('background_color', 'FFFFFF')
        background = np.array([int(color[i:i+2], 16) for i in (0, 2, 4)])
        rgba = np.asarray(crop).copy()
        difference = np.abs(rgba[:, :, :3].astype('int16') - background).max(axis=2)
        rgba[:, :, 3] = np.clip((difference - 5) * 255 / 20, 0, 255).astype('uint8')
        crop = Image.fromarray(rgba)
    asset = output / 'assets' / (identifier + '.png')
    crop.save(asset)
    result = {'id': identifier, 'path': 'assets/' + asset.name, 'sha256': digest(asset),
              'bbox': arguments['bbox'], 'cleaned_text_ids': cleaned_ids,
              'cleaned_text_boxes': cleaned_boxes, 'cropped_from_sha256': digest(crop_source)}
    result['requires_asset_ids'] = [other for other in cleared_ids
        if overlapping_ink(result, state['assets'][other], output) >= 12]
    state['assets'][identifier] = result
    write_json(folder / (identifier + '.json'), result)
    return result, asset


def duplicated_asset_strokes(ir, state, source, output):
    """Find source strokes present in two independently editable raster objects."""
    import cv2
    import numpy as np
    original = np.asarray(Image.open(source).convert('RGB'))
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype('float32')
    detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2)) > 15
    records = []
    for node in ir['nodes']:
        if node['type'] != 'image':
            continue
        asset = state['assets'][node['asset_id']]
        with Image.open(output / asset['path']) as image:
            rgba = np.asarray(image.convert('RGBA'))
        local_gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY).astype('float32')
        local_detail = np.abs(local_gray - cv2.GaussianBlur(local_gray, (0, 0), 2)) > 15
        records.append((node['id'], safe_box(asset['bbox'], state['size']), rgba, local_detail))
    conflicts = []
    for index, (left_id, left, left_pixels, left_detail) in enumerate(records):
        for right_id, right, right_pixels, right_detail in records[index+1:]:
            x0, y0 = max(left[0], right[0]), max(left[1], right[1])
            x1, y1 = min(left[2], right[2]), min(left[3], right[3])
            if x1 <= x0 or y1 <= y0:
                continue
            a = left_pixels[y0-left[1]:y1-left[1], x0-left[0]:x1-left[0]]
            b = right_pixels[y0-right[1]:y1-right[1], x0-right[0]:x1-right[0]]
            reference = original[y0:y1, x0:x1].astype('int16')
            shared = ((a[:, :, 3] > 128) & (b[:, :, 3] > 128) & detail[y0:y1, x0:x1]
                      & left_detail[y0-left[1]:y1-left[1], x0-left[0]:x1-left[0]]
                      & right_detail[y0-right[1]:y1-right[1], x0-right[0]:x1-right[0]]
                      & (np.abs(a[:, :, :3].astype('int16') - reference).max(axis=2) <= 8)
                      & (np.abs(b[:, :, :3].astype('int16') - reference).max(axis=2) <= 8))
            count = int(shared.sum())
            if count >= 12:
                conflicts.append({'objects': [left_id, right_id], 'source_overlap_bbox':
                                  [x0, y0, x1-x0, y1-y0], 'duplicated_detail_pixels': count})
    return conflicts


def pending_visual_objects(ir, inventory):
    assets = {node['asset_id'] for node in ir['nodes'] if node['type'] == 'image'}
    native = {node['id'] for node in ir['nodes'] if node['type'] != 'image'}
    return [item for item in inventory.values()
            if (item['decision'] == 'asset' and item.get('asset_id', item['id']) not in assets)
            or (item['decision'] == 'native' and item.get('node_id', item['id']) not in native)]


def run(args, key):
    """Allow exactly one writer for a task directory, including resumed runs."""
    output = args.output.resolve()
    if getattr(args, 'resume', False):
        if not output.is_dir():
            raise ValueError('resume requires an existing task directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / '.run.lock').open('a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError('task directory is already being processed') from None
        try:
            return run_locked(args, key)
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_locked(args, key):
    output = args.output.resolve()
    (output / "source").mkdir(exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    source = output / "source/page.png"
    if not source.exists():
        with Image.open(args.source) as original:
            rgb_page(original).save(source)
    with Image.open(source) as image:
        size = list(image.size)
    if not (output / 'input-manifest.json').exists():
        write_json(output / 'input-manifest.json', {'input_path': str(args.source.resolve()),
            'input_sha256': digest(args.source), 'normalized_source_sha256': digest(source), 'size': size})
    state = {"schema": "image-only-loop/1", "source_sha256": digest(source),
             "size": size, "assets": {}, "best_build": None, "history": [],
             "model": args.model, "status": "RUNNING", 'next_turn': 0, "human_acceptance": False,
             "powerpoint_checked": False, "wps_checked": False, 'started_at': time.time()}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
        {"type": "text", "text": json.dumps({"source_sha256": digest(source), "size": size})},
        {"type": "image_url", "image_url": {"url": image_data(source)}}]}]
    start_turn = 0
    if getattr(args, 'resume', False):
        checkpoint = output / 'checkpoint.json'
        saved = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
        state = saved['state'] if saved else json.loads((output / 'state.json').read_text(encoding='utf-8'))
        if digest(source) != state['source_sha256']:
            raise ValueError('source changed since checkpoint')
        messages = saved['messages'] if saved else json.loads((output / 'messages.json').read_text(encoding='utf-8'))
        start_turn = state['next_turn']
        for reservation in output.glob('turn-*/reservation.json'):
            receipt = json.loads(reservation.read_text(encoding='utf-8'))
            if receipt['status'] in ('RESPONSE_UNKNOWN', 'HTTP_RECEIVED') and not (reservation.parent / 'raw-response.json').exists():
                raise ValueError('unknown response requires reconciliation before resume')
        pending = output / f'turn-{start_turn:03}' / 'reservation.json'
        if pending.exists():
            receipt = json.loads(pending.read_text(encoding='utf-8'))
            if receipt['status'] in ('RATE_LIMITED', 'HTTP_ERROR'):
                remaining = receipt.get('retry_not_before', 0) - time.time()
                if remaining > 0:
                    time.sleep(remaining)
                start_turn += 1
        state['status'] = 'RUNNING'
        state.pop('model_review', None)
        state.pop('model_review_binding', None)
        restore_dependencies(state, output)
        messages[0] = {'role': 'system', 'content': SYSTEM}
        if state.get('current_ir'):
            state['pending_asset_conflicts'] = duplicated_asset_strokes(
                state['current_ir'], state, source, output)
        messages.append({'role': 'user', 'content': 'Resume reconstruction with the current tool '
                         'definitions. Correct the recorded remaining issues, create a new plate '
                         'and inspect a fresh compiled preview before finish. Prior valid builds are retained. '
                         'Current independent-asset duplicate stroke diagnostics: ' +
                         json.dumps(state.get('pending_asset_conflicts', []), ensure_ascii=False)})
        probe = state.get('best_probe', {})
        if probe.get('text_residual_status') == 'REVIEW_REQUIRED':
            messages.append({'role': 'user', 'content': 'The actual-PPTX no-native-text probe still '
                'has source glyph residuals. Inspect the failed-region comparisons from a fresh compile. '
                'Readable fragments or duplicated intrinsic UI captions need local correction. '
                'When tiny intrinsic UI lettering cannot be cleared safely, use remove_text with '
                'YOUR within_inventory module region and re-extract that module from a regenerated '
                'plate with its old image deselected. Keep standalone body text native. '
                'Changing to raster ownership must restore the original pixels and preserve content.'})
    state.setdefault('executor_versions', []).append({'first_turn': start_turn, 'sha256': EXECUTOR_SHA256})
    state['compiled_this_execution'] = False
    for turn in range(start_turn, start_turn + args.max_turns):
        folder = output / f"turn-{turn:03}"
        replay = folder.exists() and (folder / 'raw-response.json').exists()
        folder.mkdir(exist_ok=replay)
        if len(messages) > 24 and not replay:
            context = {'source_size': size, 'source_sha256': state['source_sha256'],
                       'current_ir': state.get('current_ir'), 'best_build': state.get('best_build'),
                       'pending_asset_conflicts': state.get('pending_asset_conflicts', []),
                       'visual_inventory': state.get('visual_inventory', {}),
                       'assets': [{k: a[k] for k in ('id', 'path', 'sha256', 'bbox', 'cleaned_text_ids') if k in a}
                                  for a in state['assets'].values()],
                       'recent_results': [row['result'] for row in state['history'][-3:]]}
            ocr_path = output / 'ocr.json'
            if ocr_path.exists():
                context['ocr'] = json.loads(ocr_path.read_text(encoding='utf-8'))
            write_json(folder / 'context-snapshot.json', context)
            content = [{'type': 'text', 'text': 'Continue from this exact runtime checkpoint. '
                        'All content below was produced in this run. ' + json.dumps(context, ensure_ascii=False)},
                       {'type': 'image_url', 'image_url': {'url': image_data(source)}}]
            if state['best_build']:
                content.append({'type': 'image_url', 'image_url': {
                    'url': image_data(Path(state['best_build']['preview']))}})
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
        state['next_turn'] = turn
        write_json(output / 'state.json', state)
        write_json(output / 'messages.json', messages)
        write_json(output / 'checkpoint.json', {'state': state, 'messages': messages})
        if replay:
            raw = json.loads((folder / 'raw-response.json').read_text(encoding='utf-8'))
            elapsed = json.loads((folder / 'reservation.json').read_text(encoding='utf-8')).get('elapsed_seconds', 0)
        else:
            payload = {"model": args.model, "messages": messages, "temperature": 0,
                       "max_tokens": 10000, "enable_thinking": False,
                       'response_format': {'type': 'json_object'}}
            call_receipt = {"status": "RESPONSE_UNKNOWN", "model": args.model,
                            'endpoint': args.base_url.rstrip('/') + '/chat/completions',
                            "created_at": time.time(), 'request_sha256': hashlib.sha256(
                                json.dumps(payload, ensure_ascii=False).encode()).hexdigest()}
            write_json(folder / 'reservation.json', call_receipt)
            started = time.monotonic()
            try:
                response = requests.post(args.base_url.rstrip('/') + '/chat/completions',
                    headers={"Authorization": "Bearer " + key},
                    json=payload, timeout=args.timeout)
            except requests.RequestException as exc:
                state.update(status="MODEL_RESPONSE_UNKNOWN", error_type=type(exc).__name__)
                break
            elapsed = time.monotonic()-started
            write_json(folder / "reservation.json", call_receipt | {"status": "HTTP_RECEIVED", "http": response.status_code,
                                                     "elapsed_seconds": elapsed})
            if response.status_code == 429:
                delay = retry_after_seconds(response.headers.get("Retry-After"))
                write_json(folder / "rate-limit.json", {"retry_after_seconds": delay})
                write_json(folder / 'reservation.json', call_receipt | {'status': 'RATE_LIMITED', 'http': 429,
                                                        'elapsed_seconds': elapsed,
                                                        'retry_not_before': time.time() + (delay or 0)})
                if delay is None:
                    state["status"] = "RATE_LIMITED"
                    break
                time.sleep(delay)
                continue
            if response.status_code != 200:
                write_json(folder / 'reservation.json', call_receipt | {'status': 'HTTP_ERROR', 'http': response.status_code,
                                                        'elapsed_seconds': elapsed})
                state.update(status="MODEL_HTTP_ERROR", http_status=response.status_code)
                break
            try:
                raw = response.json()
            except ValueError:
                state.update(status='MODEL_RESPONSE_UNKNOWN', error_type='InvalidResponseJSON')
                break
            write_json(folder / "raw-response.json", raw)
            write_json(folder / 'reservation.json', call_receipt | {'status': 'RESPONSE_RECORDED', 'http': 200,
                                                    'elapsed_seconds': elapsed,
                                                    'response_sha256': digest(folder / 'raw-response.json')})
        content = raw["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": content})
        images = []
        name = 'decode_action'
        try:
            name, arguments = decode_action(content)
            write_json(folder / 'action.json', {'action': name, 'arguments': arguments})
            if name == "ocr":
                result = local_ocr(source, args.models)
                write_json(output / "ocr.json", result)
            elif name == 'inventory':
                inventory = copy.deepcopy(state.get('visual_inventory', {}))
                for item in arguments['objects']:
                    safe_box(item['bbox'], size)
                    if (not isinstance(item['id'], str) or not item['id'] or not item.get('role')
                            or item['decision'] not in ('asset', 'backing', 'native')):
                        raise ValueError('inventory needs id, role, bbox and decision')
                    if item['decision'] == 'backing' and not item.get('reason'):
                        raise ValueError('backing decisions need an uncertainty reason')
                    inventory[item['id']] = item
                state['visual_inventory'] = inventory
                result = {'objects': list(inventory.values()),
                          'next_actions': ['draft', 'extract_many', 'plate', 'compile']}
            elif name == 'draft':
                if 'ir' in arguments or 'schema_version' in arguments:
                    state['current_ir'] = seal(arguments.get('ir', arguments), source, state['assets'])
                else:
                    if 'texts' not in arguments:
                        raise ValueError('draft requires arguments.texts as an explicit text plan array, '
                                         'and arguments.objects as extracted asset IDs, or arguments.ir as full SlideIR')
                    ocr = json.loads((output / 'ocr.json').read_text(encoding='utf-8'))
                    state['current_ir'] = draft_ir(arguments, state, source, ocr)
                state.pop('plate_plan', None)
                write_json(folder / 'draft.json', state['current_ir'])
                result = {'draft_path': str(folder / 'draft.json'),
                          'text_count': sum(n['type'] == 'text' for n in state['current_ir']['nodes']),
                          'node_ids': [n['id'] for n in state['current_ir']['nodes']],
                          'next_actions': ['plate', 'compile']}
            elif name == 'patch':
                local = copy.deepcopy(arguments)
                footprints = {}
                for entry in local['patches']:
                    if 'source_text_bbox' in entry['changes']:
                        footprint = entry['changes'].pop('source_text_bbox')
                        safe_box(footprint, size)
                        footprints[entry['id']] = footprint
                revised = patch(state['current_ir'], local)
                for node in revised['nodes']:
                    if node['id'] in footprints:
                        if node['type'] != 'text':
                            raise ValueError('source_text_bbox requires a text node')
                        node['source_text_bbox'] = footprints[node['id']]
                state['current_ir'] = revised
                state.pop('plate_plan', None)
                result = {'patched': True, 'next_actions': ['plate', 'compile']}
            elif name == 'fit':
                from model_text_layout import calibrate
                fitted, report = calibrate(state['current_ir'], source, output)
                state['current_ir'] = fitted
                state.pop('plate_plan', None)
                write_json(folder / 'text-layout.json', report)
                result = {'report': report, 'next_actions': ['plate', 'compile']}
            elif name == 'objects':
                ir = copy.deepcopy(state['current_ir'])
                previous_ids = [n['asset_id'] for n in ir['nodes'] if n['type'] == 'image']
                selected_ids = arguments.get('ids')
                if selected_ids is None:
                    selected_ids = [identifier for identifier in previous_ids
                                    if identifier not in arguments.get('remove_ids', [])]
                    selected_ids.extend(identifier for identifier in arguments.get('add_ids', [])
                                        if identifier not in selected_ids)
                ir['nodes'] = [n for n in ir['nodes'] if n['type'] != 'image']
                ir['assets'] = []
                ir['background'].pop('asset_id', None)
                for index, identifier in enumerate(selected_ids):
                    asset = state['assets'][identifier]
                    ir['assets'].append({k: asset[k] for k in ('id', 'path', 'sha256')} |
                                        {'purpose': 'module', 'contains_evidence_ids': []})
                    ir['nodes'].append({'id': 'object-' + identifier, 'type': 'image',
                                       'bbox': asset['bbox'], 'asset_id': identifier,
                                       'evidence_ids': [], 'z': index - len(selected_ids)})
                state['current_ir'] = ir
                state.pop('plate_plan', None)
                result = {'selected_objects': selected_ids,
                          'removed_objects': [identifier for identifier in previous_ids if identifier not in selected_ids],
                          'next_actions': ['plate', 'compile']}
            elif name == 'remove_text':
                ir, result = remove_text_plan(arguments, state)
                state['current_ir'] = ir
                state.pop('plate_plan', None)
            elif name in ('extract', 'extract_many'):
                objects = arguments['objects'] if name == 'extract_many' else [arguments]
                extracted, failures = [], []
                for item in objects:
                    try:
                        receipt, asset = extract_object(item, state, source, output, folder)
                        extracted.append(receipt)
                        images.append(asset)
                        write_json(output / 'state.json', state)
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        failures.append({'id': item.get('id'), 'error': str(exc)})
                result = {'objects': extracted, 'errors': failures}
                if failures and not extracted:
                    result['error'] = 'all requested extractions failed, correct the reported parameters'
            elif name == 'view':
                box = safe_box(arguments['bbox'], size)
                crop_source = source
                if arguments.get('target') == 'preview':
                    crop_source = Path(state['best_build']['preview'])
                if arguments.get('target') == 'plate':
                    aid = state['current_ir']['background']['asset_id']
                    crop_source = output / state['assets'][aid]['path']
                target = folder / 'crop.png'
                with Image.open(crop_source) as image:
                    image.crop(box).save(target)
                images.append(target)
                result = {'bbox': arguments['bbox'], 'target': arguments.get('target', 'source')}
            elif name == "plate":
                identifier = f"plate-{turn}"
                target = output / 'assets' / (identifier + '.png')
                ir = copy.deepcopy(arguments.get('ir', state.get('current_ir')))
                if not ir:
                    raise ValueError('draft or supply IR before plate')
                ir = seal(ir, source, state['assets'])
                report = build_clean_plate(ir, source, target, state, output)
                result = {"id": identifier, "path": 'assets/'+target.name,
                          "sha256": digest(target), 'inspection': report,
                          'cleared_asset_ids': [n['asset_id'] for n in ir['nodes'] if n['type'] == 'image']}
                state['assets'][identifier] = dict(result)
                ir['assets'] = [a for a in ir['assets'] if a['purpose'] != 'cleaned_background']
                owned = {eid for n in ir['nodes'] for eid in n['evidence_ids']}
                owned.update(eid for a in ir['assets'] for eid in a['contains_evidence_ids'])
                ir['assets'].append({'id': identifier, 'path': result['path'],
                                    'sha256': result['sha256'], 'purpose': 'cleaned_background',
                                    'contains_evidence_ids': [e['id'] for e in ir['evidence'] if e['id'] not in owned]})
                ir['background']['asset_id'] = identifier
                state['last_model_plan'] = copy.deepcopy(ir)
                ir, typography, summary = fit_model_text(ir, source, output)
                write_json(folder / 'typography.json', typography)
                result['typography'] = summary
                state['current_ir'] = ir
                state['plate_plan'] = ir
                write_json(folder / 'plate-inspection.json', report)
                images.append(target)
            elif name == "compile":
                ir = seal(arguments.get('ir', state.get('current_ir')), source, state["assets"])
                if ir == state.get('last_model_plan'):
                    ir = state['current_ir']
                if ir != state.get('plate_plan'):
                    raise ValueError('create a cleaned plate for this exact IR before compile')
                missing = dependency_conflicts(ir, state)
                if missing:
                    raise ValueError('Parent artwork has erased children: ' + json.dumps(missing) +
                        '. Select required children ABOVE parents, or deselect parents and those children, '
                        'regenerate plate and re-extract parents preserving source child artwork.')
                conflicts = duplicated_asset_strokes(ir, state, source, output)
                write_json(folder / 'asset-overlap.json', conflicts)
                if conflicts:
                    raise ValueError('Source strokes are duplicated across independent assets: ' +
                        json.dumps(conflicts[:8], ensure_ascii=False) +
                        '. Re-extract corrected object bounds, or extract a parent from a plate '
                        'where its child objects have already been removed. Deselect that parent before '
                        'making the plate. Use objects.ids in bottom-to-top order to place repaired '
                        'parents below children, preserving unrelated selected objects.')
                ir_path = folder / "slideir.json"
                write_json(ir_path, ir)
                shutil.copytree(output / 'source', folder / 'source', dirs_exist_ok=True)
                shutil.copytree(output / 'assets', folder / 'assets', dirs_exist_ok=True)
                result = compiler(ir_path, 'validate')
                build = compiler(ir_path, 'compile')
                images.append(Path(build["preview"]))
                result = {"validation": result, "build": build}
                probe = render_probe(Path(build['pptx']), ir, folder, ROOT, node_cmd(), source=source,
                                     preview=Path(build['preview']))
                write_json(folder / 'no-text-probe.json', probe)
                state["best_build"] = build
                state["best_ir"] = str(ir_path)
                state["current_ir"] = ir
                state['best_probe'] = probe
                state['compiled_this_execution'] = True
                state['pending_asset_conflicts'] = []
                write_json(folder / 'generation.json', {'ir_sha256': digest(ir_path), 'build': build,
                    'no_native_text_probe': probe, 'executor_sha256': EXECUTOR_SHA256})
                images.append(Path(probe['render']['pages'][0]['path']))
                images.extend(Path(row['path']) for row in probe.get('comparison_crops', []))
                result['no_native_text_probe'] = probe
                result['image_order'] = ['final PPTX preview', 'same PPTX with native text removed',
                    *[f"{row['id']}: original | final | no native text" for row in probe.get('comparison_crops', [])]]
                result['pending_visual_objects'] = pending_visual_objects(ir, state.get('visual_inventory', {}))
            elif name == "finish":
                if state["best_build"] is None:
                    raise ValueError("compile and inspect actual preview before finishing")
                if not state.get('compiled_this_execution'):
                    raise ValueError('Resume requires a fresh plate and successful compile with the current tools. '
                                     'Inspect both actual-PPTX previews before finish.')
                if (arguments.get('visual_status') not in ('pass', 'needs_review')
                        or not isinstance(arguments.get('issues'), list)
                        or not all(isinstance(issue, str) for issue in arguments['issues'])):
                    raise ValueError('finish requires visual_status and an issues string array')
                if state['current_ir'] != json.loads(Path(state['best_ir']).read_text(encoding='utf-8')):
                    raise ValueError('current plan changed since last valid compile, compile and inspect it first')
                state['pending_visual_objects'] = pending_visual_objects(
                    state['current_ir'], state.get('visual_inventory', {}))
                if state['pending_visual_objects']:
                    raise ValueError('planned visual objects need a final representation: ' +
                        json.dumps([item['id'] for item in state['pending_visual_objects']], ensure_ascii=False) +
                        '. Complete the accepted asset/native decisions, or revise genuinely uncertain '
                        'objects to backing with individual reasons in inventory. Other selected assets stay selected.')
                state.update(status="REVIEW_REQUIRED", model_review=arguments)
                state['editable_completion'] = ('UNKNOWN' if not state.get('visual_inventory') else
                    'PARTIAL' if state['pending_visual_objects'] or any(item['decision'] == 'backing'
                    for item in state['visual_inventory'].values()) else 'MODEL_INVENTORY_COVERED')
                state['model_review_binding'] = {'source_sha256': state['source_sha256'],
                    'pptx_sha256': digest(Path(state['best_build']['pptx'])),
                    'preview_sha256': digest(Path(state['best_build']['preview'])),
                    'no_native_text_preview_sha256': digest(Path(state['best_probe']['render']['pages'][0]['path']))}
                result = arguments
            else:
                raise ValueError('unknown action. Choose ocr, inventory, draft, view, extract, extract_many, '
                                 'objects, remove_text, patch, fit, plate, compile, or finish')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result = {"error": f'{name}: {exc}'[-4000:],
                      "instruction": 'Correct this tool call. Return {"action":"tool name","arguments":{...}}.'}
        state["history"].append({"turn": turn, "elapsed_seconds": elapsed,
                                  "usage": raw.get("usage", {}), "result": result})
        write_json(folder / 'tool-result.json', result)
        messages.append({"role": "user", "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)},
            *[{"type": "image_url", "image_url": {"url": image_data(path)}} for path in images]]})
        state['next_turn'] = turn + 1
        write_json(output / 'messages.json', messages)
        write_json(output / 'state.json', state)
        write_json(output / 'checkpoint.json', {'state': state, 'messages': messages})
        if state['status'] == 'REVIEW_REQUIRED':
            break
        recent = state['history'][-3:]
        if len(recent) == 3 and all(isinstance(h['result'], dict) and 'error' in h['result'] for h in recent):
            messages.append({'role': 'user', 'content': 'Three consecutive tool failures. Read the exact '
                             'diagnostics and change strategy. Reuse the current IR and valid assets. '
                             'Inspect the relevant source or preview region when needed, and avoid '
                             'repeating unchanged calls. Use actual tool names from next_actions.'})
            write_json(output / 'messages.json', messages)
        if len(state['history']) >= 6 and all(isinstance(h['result'], dict) and 'error' in h['result']
                                            for h in state['history'][-6:]):
            state['status'] = 'TOOL_FAILURE_STALLED'
            break
    if state["status"] == "RUNNING":
        state["status"] = "TURN_LIMIT_REACHED"
    state['updated_at'] = time.time()
    state['total_elapsed_seconds'] = state['updated_at'] - state.get('started_at', state['updated_at'])
    state['model_calls'] = len(list(output.glob('turn-*/reservation.json')))
    state['usage'] = {key: sum(row['usage'].get(key, 0) for row in state['history'])
                      for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    write_json(output / 'state.json', state)
    write_json(output / 'result.json', state)
    write_json(output / 'checkpoint.json', {'state': state, 'messages': messages})
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / ('image-only-' + str(time.time_ns())))
    parser.add_argument('--models', type=Path, default=Path(os.environ.get(
        'IMAGE2PPT_OCR_MODELS', str(ROOT / 'ppt-rebuild/models'))))
    parser.add_argument('--base-url', default='https://moma.cmecloud.cn/v1')
    parser.add_argument('--model', default='qwen/qwen3.8-max')
    parser.add_argument('--max-turns', type=int, default=120)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    credential = os.environ.get('MOMA_API_KEY') or getpass.getpass('MoMA key (memory only): ')
    if not credential:
        raise SystemExit('MoMA credential required')
    result = run(args, credential)
    print(json.dumps({"status": result['status'], "output": str(args.output)}, ensure_ascii=False))
