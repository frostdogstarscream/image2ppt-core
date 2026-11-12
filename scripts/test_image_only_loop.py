"""Regressions for the original-image entrypoint and durable tool loop."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from PIL import Image, ImageDraw, ImageFont

import image_only_loop as loop
import image_only_typography as typography


class ImageOnlyTest(unittest.TestCase):
    def test_inventory_tracks_accepted_assets_native_nodes_and_backing(self):
        inventory = {'a': {'id': 'a', 'decision': 'asset', 'asset_id': 'photo'},
                     'n': {'id': 'n', 'decision': 'native', 'node_id': 'panel'},
                     'b': {'id': 'b', 'decision': 'backing', 'reason': 'uncertain decoration'}}
        ir = {'nodes': [{'id': 'text', 'type': 'text'}]}
        self.assertEqual([r['id'] for r in loop.pending_visual_objects(ir, inventory)], ['a', 'n'])
        ir['nodes'].extend([{'id': 'p', 'type': 'image', 'asset_id': 'photo'},
                            {'id': 'panel', 'type': 'shape'}])
        self.assertEqual(loop.pending_visual_objects(ir, inventory), [])

    def test_containment_preserves_runs_and_wrapping_and_converts_insets(self):
        if not Path('C:/Windows/Fonts/arial.ttf').exists():
            self.skipTest('installed fonts required for deterministic containment')
        node = {'id': 't', 'type': 'text', 'text': 'Hello world', 'bbox': [5, 5, 60, 30],
                'font': {'latin': 'Arial', 'ea': 'Microsoft YaHei', 'size_pt': 42},
                'insets': [10, 6, 4, 4], 'runs': [{'text': 'Hello ', 'color': '112233'},
                                               {'text': 'world', 'bold': True, 'color': 'AA0000'}]}
        wrapped = {**copy.deepcopy(node), 'id': 'wrapped', 'wrap': True,
                   'font': {'latin': 'Arial', 'ea': 'Microsoft YaHei', 'size_pt': 20}}
        ir = {'canvas': {'source_px': [200, 100], 'target_px': [400, 200]},
              'nodes': [node, wrapped]}

        def measured_only(measured, source, assets):
            return copy.deepcopy(measured), {'rows': [], 'splits': []}

        with patch.object(typography, 'calibrate', side_effect=measured_only):
            fitted, report, _ = typography.fit_model_text(ir, Path('unused'), Path('unused'))
        by_id = {n['id']: n for n in fitted['nodes']}
        self.assertLess(by_id['t']['font']['size_pt'], 42)
        self.assertEqual(by_id['t']['runs'], node['runs'])
        self.assertEqual(by_id['wrapped']['font']['size_pt'], 20)
        self.assertEqual(report['containment'][0]['available_source_px'], [52, 26])
        self.assertEqual(node['font']['size_pt'], 42)

    def test_flat_action_envelope_is_losslessly_normalized(self):
        arguments = {'texts': [{'id': 'ocr-0', 'color': '0015A0'}], 'objects': []}
        wrapped = json.dumps({'action': 'draft', 'arguments': arguments})
        flat = json.dumps({'action': 'draft', **arguments})
        self.assertEqual(loop.decode_action(wrapped), loop.decode_action(flat))
        with self.assertRaisesRegex(ValueError, 'duplicate top-level'):
            loop.decode_action(json.dumps({'action': 'draft', 'arguments': arguments, 'texts': []}))

    def test_asset_text_collision_uses_actual_source_strokes(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            root = Path(tmp)
            (root / 'assets').mkdir()
            source = root / 'source.png'
            image = Image.new('RGB', (240, 100), 'white')
            ImageDraw.Draw(image).text((12, 12), 'Bold', fill='blue')
            image.save(source)
            asset = root / 'assets/parent.png'
            Image.new('RGBA', (100, 40), 'white').save(asset)
            ir = {'nodes': [{'id': 't', 'type': 'text', 'bbox': [10, 10, 60, 20],
                             'text': 'Bold', 'color': '333333'},
                            {'id': 'parent', 'type': 'image', 'asset_id': 'parent'}]}
            state = {'size': [240, 100], 'assets': {'parent': {
                'bbox': [5, 5, 100, 40], 'path': 'assets/parent.png'}}}
            # The pixels establish cleanup even without a prior PASS declaration.
            loop.build_clean_plate(ir, source, root / 'safe.png', state, root)
            image.crop((5, 5, 105, 45)).convert('RGBA').save(asset)
            with self.assertRaisesRegex(ValueError, 'retains original glyph strokes'):
                loop.build_clean_plate(ir, source, root / 'unsafe.png', state, root)

    def test_same_task_cannot_be_resumed_while_its_writer_is_active(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            args = self.args(Path(tmp))

            def while_locked(arguments, key):
                resumed = argparse.Namespace(**vars(arguments))
                resumed.resume = True
                with self.assertRaisesRegex(ValueError, 'already being processed'):
                    loop.run(resumed, key)
                return {'status': 'LOCK_VERIFIED'}

            with patch.object(loop, 'run_locked', side_effect=while_locked):
                result = loop.run(args, 'temporary-test-key')
            self.assertEqual(result['status'], 'LOCK_VERIFIED')

    def test_plate_cannot_be_used_to_extract_an_already_erased_parent(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            root = Path(tmp)
            (root / 'assets').mkdir()
            source = root / 'source.png'
            Image.new('RGB', (100, 100), 'white').save(source)
            Image.new('RGBA', (40, 40), 'blue').save(root / 'assets/object.png')
            Image.new('RGB', (100, 100), 'white').save(root / 'assets/plate.png')
            ir = {'background': {'asset_id': 'plate'}, 'nodes': [
                {'id': 'object', 'type': 'image', 'asset_id': 'photo'}]}
            state = {'size': [100, 100], 'current_ir': ir, 'plate_plan': ir,
                     'assets': {'photo': {'bbox': [20, 20, 40, 40], 'path': 'assets/object.png'},
                                'plate': {'path': 'assets/plate.png'}}}
            with self.assertRaisesRegex(ValueError, 'already removed'):
                loop.extract_object({'id': 'revision', 'bbox': [20, 20, 40, 40],
                                     'from': 'plate'}, state, source, root, root)

    def test_overlapping_crops_cannot_duplicate_independent_icon_strokes(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            root = Path(tmp)
            (root / 'assets').mkdir()
            image = Image.new('RGB', (100, 100), 'white')
            ImageDraw.Draw(image).ellipse((30, 30, 65, 65), fill='blue')
            source = root / 'source.png'
            image.save(source)
            image.convert('RGBA').save(root / 'assets/parent.png')
            image.crop((25, 25, 70, 70)).convert('RGBA').save(root / 'assets/icon.png')
            state = {'size': [100, 100], 'assets': {
                'parent': {'path': 'assets/parent.png', 'bbox': [0, 0, 100, 100]},
                'icon': {'path': 'assets/icon.png', 'bbox': [25, 25, 45, 45]}}}
            ir = {'nodes': [{'id': 'parent-node', 'type': 'image', 'asset_id': 'parent'},
                             {'id': 'icon-node', 'type': 'image', 'asset_id': 'icon'}]}
            conflicts = loop.duplicated_asset_strokes(ir, state, source, root)
            self.assertEqual(conflicts[0]['objects'], ['parent-node', 'icon-node'])
            Image.new('RGBA', (100, 100), (255, 255, 255, 255)).save(root / 'assets/parent.png')
            # Original white pixels near icon edges are not duplicate artwork
            # when the repaired parent itself has no edge at those pixels.
            self.assertEqual(loop.duplicated_asset_strokes(ir, state, source, root), [])
            # Transparent foreground omits the surrounding uniform background.
            import numpy as np
            icon = np.asarray(Image.open(root / 'assets/icon.png')).copy()
            icon[:, :, 3] = (icon[:, :, 0] == 0).astype('uint8') * 255
            Image.fromarray(icon).save(root / 'assets/icon.png')
            self.assertEqual(loop.duplicated_asset_strokes(ir, state, source, root), [])

    def test_recorded_pending_reply_replays_without_another_model_request(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            args = self.args(Path(tmp))
            args.max_turns = 0
            loop.run(args, 'temporary-test-key')
            pending = args.output / 'turn-000'
            pending.mkdir()
            loop.write_json(pending / 'raw-response.json', {'choices': [{'message': {
                'content': json.dumps({'action': 'ocr'})}}], 'usage': {}})
            loop.write_json(pending / 'reservation.json', {'status': 'RESPONSE_RECORDED'})
            args.resume = True
            args.max_turns = 1
            with patch.object(loop.requests, 'post') as request, \
                    patch.object(loop, 'local_ocr', return_value=[]):
                state = loop.run(args, 'temporary-test-key')
            request.assert_not_called()
            self.assertEqual(state['next_turn'], 1)
            self.assertEqual(len(state['history']), 1)

    def test_cleanup_uses_source_colors_for_unmodeled_inline_emphasis(self):
        import numpy as np
        font = Path('C:/Windows/Fonts/arialbd.ttf')
        if not font.exists():
            self.skipTest('installed Arial required for this stroke regression')
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            root = Path(tmp)
            source = root / 'source.png'
            image = Image.new('RGB', (300, 100), 'white')
            draw = ImageDraw.Draw(image)
            draw.text((15, 20), 'Bold', font=ImageFont.truetype(str(font), 45), fill='#0015A0')
            image.save(source)
            ir = {'nodes': [{'id': 't', 'type': 'text', 'bbox': [10, 15, 150, 70],
                              'text': 'Bold', 'color': '333333'}]}
            target = root / 'plate.png'
            report = loop.build_clean_plate(ir, source, target, {'size': [300, 100]}, root)
            original = np.asarray(image)
            actual = np.asarray(Image.open(target))
            ink = (original[:, :, 0] == 0) & (original[:, :, 2] == 160)
            unchanged_ink = (actual[ink] == original[ink]).all(axis=1).mean()
            self.assertLess(unchanged_ink, .01)
            self.assertEqual(report['outside_mask_changed_pixels'], 0)

    def args(self, root):
        source = root / 'input.png'
        image = Image.new('RGB', (240, 100), 'white')
        ImageDraw.Draw(image).text((20, 20), 'Hello', fill='black')
        image.save(source)
        return argparse.Namespace(source=source, output=root / 'run', models=root,
                                  model='test', base_url='https://test.invalid/v1',
                                  max_turns=6, timeout=10, resume=False)

    def test_bbox_error_reports_original_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'require x\\+width <= 100'):
            loop.safe_box([0, 80, 60, 40], [100, 100])

    def test_model_selects_arbitrary_text_count_and_overrides_ocr(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            args = self.args(Path(tmp))
            evidence = [{'id': 'ocr-87', 'kind': 'text', 'text': 'wrong',
                         'bbox': [20, 20, 50, 20], 'confidence': .9}]
            ir = loop.draft_ir({'texts': [{'id': 'ocr-87', 'text': 'Hello',
                                 'color': 'CC0011', 'bold': True}], 'objects': []},
                               {'size': [240, 100], 'assets': {}}, args.source, evidence)
            self.assertEqual(ir['nodes'][0]['text'], 'Hello')
            self.assertTrue(ir['nodes'][0]['bold'])
            self.assertNotIn('confidence', ir['evidence'][0])
            self.assertEqual(evidence[0]['text'], 'wrong')

    def test_unknown_response_cannot_be_retried_by_resume(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            args = self.args(Path(tmp))
            with patch.object(loop.requests, 'post', side_effect=requests.Timeout):
                state = loop.run(args, 'temporary-test-key')
            self.assertEqual(state['status'], 'MODEL_RESPONSE_UNKNOWN')
            reservation = json.loads((args.output / 'turn-000/reservation.json').read_text())
            self.assertEqual(reservation['status'], 'RESPONSE_UNKNOWN')
            args.resume = True
            with self.assertRaisesRegex(ValueError, 'unknown response'):
                loop.run(args, 'temporary-test-key')

    def test_full_loop_records_finish_and_keeps_last_valid_build(self):
        with tempfile.TemporaryDirectory(dir=loop.ROOT / '.tmp-test') as tmp:
            args = self.args(Path(tmp))
            actions = [{'action': 'ocr'}, {'action': 'draft', 'arguments': {'texts': [
                {'id': 'ocr-0', 'color': '000000'}]}}, {'action': 'plate'},
                {'action': 'compile'}, {'action': 'compile', 'arguments': {'ir': {}}},
                {'action': 'finish', 'arguments': {'issues': [], 'visual_status': 'needs_review'}}]
            responses = []
            for action in actions:
                response = unittest.mock.Mock(status_code=200)
                response.json.return_value = {'choices': [{'message': {'content': json.dumps(action)}}],
                                              'usage': {'total_tokens': 10}}
                responses.append(response)
            evidence = [{'id': 'ocr-0', 'kind': 'text', 'text': 'Hello', 'bbox': [20, 20, 50, 20]}]

            def plate(ir, source, target, state, output):
                Image.new('RGB', (240, 100), 'white').save(target)
                return {'outside_mask_changed_pixels': 0}

            def compiler(ir_path, action):
                if action == 'validate':
                    return {'valid': True}
                preview = ir_path.parent / 'preview.png'
                Image.new('RGB', (240, 100), 'white').save(preview)
                (ir_path.parent / 'slide.pptx').write_bytes(b'local compiler fixture')
                return {'preview': str(preview), 'pptx': str(ir_path.parent / 'slide.pptx')}

            with patch.object(loop.requests, 'post', side_effect=responses), \
                    patch.object(loop, 'local_ocr', return_value=evidence), \
                    patch.object(loop, 'build_clean_plate', side_effect=plate), \
                    patch.object(loop, 'render_probe', side_effect=lambda pptx, ir, folder, root, node, **kwargs:
                        {'render': {'pages': [{'path': str(folder / 'preview.png')}]}}), \
                    patch.object(loop, 'compiler', side_effect=compiler):
                state = loop.run(args, 'temporary-test-key')
            self.assertEqual(state['status'], 'REVIEW_REQUIRED')
            self.assertEqual(len(state['history']), 6)
            self.assertIn('turn-003', state['best_ir'])
            self.assertIn('error', state['history'][4]['result'])
            self.assertEqual(state['next_turn'], 6)
            self.assertTrue((args.output / 'messages.json').exists())
            self.assertNotIn('temporary-test-key', (args.output / 'messages.json').read_text())
            args.resume = True
            args.max_turns = 1
            old_build = state['best_build']
            with patch.object(loop.requests, 'post', return_value=responses[-1]):
                resumed = loop.run(args, 'temporary-test-key')
            self.assertEqual(resumed['status'], 'TURN_LIMIT_REACHED')
            self.assertIn('fresh plate', resumed['history'][-1]['result']['error'])
            self.assertEqual(resumed['best_build'], old_build)


if __name__ == '__main__':
    unittest.main()
