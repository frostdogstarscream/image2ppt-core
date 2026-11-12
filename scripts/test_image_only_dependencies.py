"""Regressions for missing nested artwork and intrinsic screenshot text."""
import tempfile
import unittest
import zipfile
import json
from pathlib import Path

from PIL import Image

from image_only_assets import dependency_conflicts, overlapping_ink, remove_text_plan, restore_dependencies
from image_only_probe import without_native_text
from image_only_loop import ROOT


class DependencyTest(unittest.TestCase):
    def test_recorded_plate_restores_legacy_dependency_without_changing_pixels(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp-test') as temporary:
            root = Path(temporary)
            for name in ['parent', 'child']:
                Image.new('RGBA', (10, 10), (0, 0, 255, 255)).save(root / f'{name}.png')
            before = (root / 'parent.png').read_bytes()
            folder = root / 'turn-001'
            folder.mkdir()
            (folder / 'action.json').write_text(json.dumps({'action': 'plate', 'arguments': {}}))
            state = {'assets': {
                'parent': {'id': 'parent', 'path': 'parent.png', 'bbox': [0, 0, 10, 10],
                           'sha256': 'parent-hash', 'cropped_from_sha256': 'plate-hash'},
                'child': {'id': 'child', 'path': 'child.png', 'bbox': [0, 0, 10, 10], 'sha256': 'child-hash'},
                'plate': {'id': 'plate', 'sha256': 'plate-hash'}}, 'history': [
                    {'turn': 0, 'result': {'selected_objects': ['child']}},
                    {'turn': 1, 'result': {'id': 'plate'}}]}
            first = root / 'turn-000'
            first.mkdir()
            (first / 'action.json').write_text(json.dumps({'action': 'objects'}))
            restore_dependencies(state, root)
            self.assertEqual(state['assets']['parent']['requires_asset_ids'], ['child'])
            self.assertEqual((root / 'parent.png').read_bytes(), before)

    def test_erased_child_must_be_present_above_parent(self):
        state = {'assets': {'parent': {'requires_asset_ids': ['child']}, 'child': {}}}
        ir = {'nodes': [{'type': 'image', 'asset_id': 'parent', 'z': 0}]}
        self.assertEqual(len(dependency_conflicts(ir, state)), 1)
        ir['nodes'].append({'type': 'image', 'asset_id': 'child', 'z': -1})
        self.assertEqual(len(dependency_conflicts(ir, state)), 1)
        ir['nodes'][1]['z'] = 1
        self.assertEqual(dependency_conflicts(ir, state), [])

    def test_transparent_holes_create_no_artwork_dependency(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp-test') as temporary:
            root = Path(temporary)
            Image.new('RGBA', (10, 10), (0, 0, 0, 0)).save(root / 'a.png')
            Image.new('RGBA', (10, 10), (0, 0, 0, 255)).save(root / 'b.png')
            self.assertEqual(overlapping_ink({'bbox': [0, 0, 10, 10], 'path': 'a.png'},
                {'bbox': [0, 0, 10, 10], 'path': 'b.png'}, root), 0)

    def test_model_region_returns_intrinsic_text_and_invalidates_cleaned_asset(self):
        state = {'visual_inventory': {'module': {'bbox': [10, 10, 30, 30]}},
                 'current_ir': {'review_notes': [], 'nodes': [
                     {'type': 'text', 'id': 'inside', 'bbox': [15, 15, 5, 5]},
                     {'type': 'text', 'id': 'outside', 'bbox': [50, 50, 5, 5]},
                     {'type': 'image', 'id': 'photo', 'asset_id': 'module'}]},
                 'assets': {'module': {'bbox': [10, 10, 30, 30], 'cleaned_text_ids': ['inside']}}}
        ir, result = remove_text_plan({'within_inventory': 'module', 'reason': 'intrinsic UI'}, state)
        self.assertEqual([n['id'] for n in ir['nodes']], ['outside'])
        self.assertEqual(result['deselected_stale_assets'], ['module'])
        self.assertEqual(len(state['current_ir']['nodes']), 3)

    def test_probe_removes_named_native_shape_and_preserves_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp-test') as temporary:
            root = Path(temporary)
            source = root / 'source.pptx'
            xml = '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr name="native"/></p:nvSpPr></p:sp><p:pic><p:nvPicPr><p:cNvPr name="photo"/></p:nvPicPr></p:pic></p:spTree></p:cSld></p:sld>'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('ppt/slides/slide1.xml', xml)
                archive.writestr('ppt/media/image1.png', b'original-image-bytes')
            before = source.read_bytes()
            report = without_native_text(source, root / 'probe.pptx', {'native'})
            self.assertEqual(report['removed_text_ids'], ['native'])
            self.assertEqual(source.read_bytes(), before)
            with zipfile.ZipFile(root / 'probe.pptx') as archive:
                self.assertNotIn(b'name="native"', archive.read('ppt/slides/slide1.xml'))
                self.assertEqual(archive.read('ppt/media/image1.png'), b'original-image-bytes')


if __name__ == '__main__':
    unittest.main()
