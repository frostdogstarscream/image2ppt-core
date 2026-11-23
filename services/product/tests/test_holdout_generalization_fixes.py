import json
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import httpx

from product import pipeline
from product.contracts import intersection
from product.materialize import digest
from product.parallel_text_candidates import peer_protection
from product.recovery.container_stage import attach_card_groups
from product.recovery.region_crop import _submask_any
from product.runtime.batch_backbone import _split_pill_divider_ocr_lines, _detect_embedded_ui_frames
from product.runtime.runner import OpenCodeClient


def test_compose_objects_handles_connector_shapes_without_batch_mismatch(tmp_path):
    """P0-1: `*Connector*` (`<p:cxnSp>`) bypasses batch merge mismatch cleanly via compose_objects.mjs."""
    w, h = 480, 270
    src_path = tmp_path / 'source.png'
    plate_path = tmp_path / 'source-plate.png'
    Image.new('RGB', (w, h), (254, 254, 254)).save(src_path)
    Image.new('RGBA', (w, h), (254, 254, 254, 255)).save(plate_path)
    ir = {
        'schema_version': '0.1',
        'source_image': 'source.png',
        'source_sha256': digest(src_path),
        'canvas': {'source_px': [w, h], 'target_px': [w, h]},
        'background': {'color': 'FEFEFE'},
        'review_status': 'needs_review',
        'review_notes': [],
        'evidence': [
            {'id': 'ev-plate', 'kind': 'shape', 'bbox': [0, 0, w, h]},
            {'id': 'ev-conn', 'kind': 'shape', 'bbox': [60, 120, 140, 20]},
        ],
        'assets': [
            {'id': 'a-plate', 'path': 'source-plate.png', 'purpose': 'module', 'sha256': digest(plate_path), 'contains_evidence_ids': []},
        ],
        'unresolved_evidence_ids': [],
        'nodes': [
            {'id': 'source-plate', 'type': 'image', 'bbox': [0, 0, w, h], 'z': 0, 'asset_id': 'a-plate', 'evidence_ids': ['ev-plate']},
            {'id': 's-001', 'type': 'shape', 'geometry': 'straightConnector1', 'bbox': [60, 120, 140, 4], 'z': 1,
             'fill': 'none', 'line': {'color': '2563EB', 'width_pt': 2.0, 'tail_end': 'triangle'}, 'evidence_ids': ['ev-conn']},
        ],
        'groups': [],
    }
    ir_file = tmp_path / 'slideir.json'
    manifest = tmp_path / 'manifest.json'
    out_dir = tmp_path / 'composed'
    ir_file.write_text(json.dumps(ir), encoding='utf-8')
    manifest.write_text(json.dumps([str(ir_file)]), encoding='utf-8')
    proc = subprocess.run(
        ['node', 'services/product/compose_objects.mjs', str(manifest), str(out_dir)],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(proc.stdout[proc.stdout.find('{'):])
    assert payload['slide_count'] == 1
    assert 'pptx' in payload


def test_runner_nudge_uses_opencode_prompt_endpoint(tmp_path):
    """P0-2: `_run` session nudge posts to `/prompt` (not GET-only `/message`)."""
    from product.materialize import write
    from product.runtime import runner
    from product.runtime.state import RuntimeState

    job = tmp_path / 'job'
    (job / 'inputs').mkdir(parents=True)
    img = job / 'inputs/01.png'
    Image.new('RGB', (64, 64), 'white').save(img)
    (job / 'request.json').write_text(json.dumps({
        'sources': ['01.png'],
        'host_stage0': True,
        'runtime_seconds': 300,
    }), encoding='utf-8')
    st = RuntimeState(job)
    write(st.root / 'state.json', {'version': 0, 'pages': [], 'candidates': {}, 'status': 'RUNNING'})
    ws = job / 'agent'
    ws.mkdir()

    calls = []

    class _FakeClient:
        def verify(self):
            return {'version': runner.PINNED_VERSION}

        def request(self, method, route, **kw):
            calls.append((method, route, kw))
            if route == '/api/mcp':
                return {'data': [{'name': 'image2ppt', 'status': {'status': 'connected'}}]}
            if route.endswith('/prompt') and 'paused' in (kw.get('json') or {}).get('text', ''):
                write(st.root / 'state.json', {'version': 1, 'pages': [], 'candidates': {}, 'status': 'SUCCEEDED'})
                (job / 'runtime/agent-result.json').write_text(json.dumps({'status': 'SUCCEEDED'}), encoding='utf-8')
                return {}
            return {}

    import unittest.mock as mock
    with mock.patch.object(runner, 'prepare_workspace', return_value=ws):
        res = runner._run(job, client=_FakeClient())
    assert res['status'] == 'SUCCEEDED'
    post_routes = [r for m, r, _ in calls if m == 'POST']
    assert all('/message' not in r for r in post_routes)
    nudge_posts = [(r, k['json']) for m, r, k in calls if m == 'POST' and r.endswith('/prompt') and 'paused' in k['json']['text']]
    assert len(nudge_posts) == 1
    assert 'compose_delivery -> audit_delivery -> finalize_job' in nudge_posts[0][1]['text']
    assert nudge_posts[0][1]['files'] == []


def test_pill_divider_ocr_split_preserves_marked_fields():
    """P1-1: `_split_pill_divider_ocr_lines` splits multi-segment pill bars separated by thin divider strokes."""
    arr = np.full((60, 320, 3), 240, dtype=np.uint8)
    # 3 dark ink glyph blocks separated by wide gaps and thin lighter vertical bars
    arr[18:42, 20:70] = 25
    arr[16:44, 95:97] = 150
    arr[18:42, 122:175] = 25
    arr[16:44, 200:202] = 150
    arr[18:42, 226:285] = 25
    ocr = {
        'lines': [
            {'id': 'l1', 'text': '应用展示丨身份认证丨待办汇聚', 'bbox': [16.0, 14.0, 275.0, 32.0], 'score': 0.96},
        ]
    }
    changed = _split_pill_divider_ocr_lines(arr, ocr)
    assert changed is True
    assert [r['text'] for r in ocr['lines']] == ['应用展示', '身份认证', '待办汇聚']
    assert [r['id'] for r in ocr['lines']] == ['l1__s1', 'l1__s2', 'l1__s3']
    # Verify tight sub-boxes avoid overlap with divider columns
    assert ocr['lines'][0]['bbox'][0] + ocr['lines'][0]['bbox'][2] <= 80
    assert ocr['lines'][1]['bbox'][0] >= 115


def test_active_resolved_peer_uses_tight_mask_instead_of_padded_rect(tmp_path):
    """P1-1: `peer_protection` exempts active RESOLVED peers from coarse pad=1 OCR blocking."""
    source = np.full((80, 140, 3), 245, dtype=np.uint8)
    source[15:30, 20:70] = 20
    source[34:49, 22:72] = 20
    (tmp_path / 'runtime-ocr-protection.json').write_text(json.dumps({
        'source_sha256': 'src',
        'lines': [
            {'id': 'l1', 'bbox': [18, 13, 54, 19], 'text': '燃气'},
            {'id': 'l2', 'bbox': [20, 32, 54, 19], 'text': '风险排查'},
        ],
    }), encoding='utf-8')
    graph = SimpleNamespace(
        source_sha256='src',
        text=SimpleNamespace(
            texts=[
                SimpleNamespace(id='t1', bbox=[18, 13, 54, 19]),
                SimpleNamespace(id='t2', bbox=[20, 32, 54, 19]),
            ],
            members={'t1': ['l1'], 't2': ['l2']},
        ),
        asset=SimpleNamespace(objects=[]),
        surface=SimpleNamespace(shapes=[]),
        ownership={'t1': 'RESOLVED', 't2': 'RESOLVED'},
        relations=[],
    )
    prot = peer_protection(graph.text.texts[0], graph, source, np.zeros((80, 140), bool), baseline_directory=tmp_path)
    # White gap pixel just outside l2's tight ink inside l2's padded rect (y=31, x=21) is NOT falsely blocked!
    assert prot[31, 21] == False
    # Actual ink of l2 (y=36, x=25) IS still protected!
    assert prot[36, 25] == True


def test_attach_card_groups_skips_multi_card_wrappers_and_groups_leaf_cards():
    """P2-2: `attach_card_groups` groups leaf cards (including 2-zone header+card) and leaves section wrappers ungrouped."""
    selected = {
        's-wrapper': {'kind': 'surface', 'bbox': [20, 150, 760, 240], 'payload': {'geometry': 'roundRect'}},
        's-card1': {'kind': 'surface', 'bbox': [30, 170, 340, 180], 'payload': {'geometry': 'roundRect'}},
        's-hdr1': {'kind': 'surface', 'bbox': [30, 170, 340, 48], 'payload': {'geometry': 'roundRect'}},
        'a-001': {'kind': 'asset', 'bbox': [45, 230, 40, 40]},
        't-001': {'kind': 'text', 'bbox': [95, 180, 180, 28]},
        's-card2': {'kind': 'surface', 'bbox': [400, 170, 340, 180], 'payload': {'geometry': 'roundRect'}},
        't-002': {'kind': 'text', 'bbox': [420, 190, 150, 28]},
    }
    attach_card_groups(selected, (1672, 941))
    assert 'move_members' not in selected['s-wrapper']
    assert 'move_members' not in selected['s-hdr1']
    assert set(selected['s-card1']['move_members']) == {'s-card1', 's-hdr1', 'a-001', 't-001'}
    assert set(selected['s-card2']['move_members']) == {'s-card2', 't-002'}


def test_compile_pages_retries_transient_vulkan_and_node_realloc(tmp_path, monkeypatch):
    """Compiler helper retries Vulkan AllocImageMemory and Node Realloc OOM up to 2 retries without sleeping in tests."""
    import pytest
    monkeypatch.setattr(pipeline, 'compiler_runtime', lambda: (tmp_path, 'node'))
    monkeypatch.setattr(pipeline, '_COMPILE_RETRY_PAUSE_S', 0.0)
    monkeypatch.setattr(pipeline, '_compiled_artifacts', lambda *args: {'pptx': 'synthetic.pptx', 'preview': 'synthetic-preview'})
    monkeypatch.setattr(pipeline, 'verify_package', lambda *args: None)

    calls = []
    errors = [
        'Failed vulkan call. Error: -2, skgpu::VulkanMemory::AllocImageMemory\n[Error: Could not encode as png]',
        "validator: 'first-party-import',\n  exitCode: 134,\n  stderr: 'Assertion failed: !(n > 0) || (ret != nullptr)'",
    ]

    def invoke(command, **kwargs):
        idx = len(calls)
        calls.append(command)
        if idx < len(errors):
            kwargs['stderr'].write(errors[idx])
            return SimpleNamespace(returncode=1)
        kwargs['stdout'].write('{}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, 'run', invoke)
    output = tmp_path / 'compile-transient'
    assert pipeline.compile_pages([], output, [])['pptx'] == 'synthetic.pptx'
    assert len(calls) == 3
    assert json.loads((output / 'compile-retry-0.json').read_text('utf8'))['reason'] == 'vulkan_image_memory'
    assert json.loads((output / 'compile-retry-1.json').read_text('utf8'))['reason'] == 'node_realloc_oom'
    assert (output / 'compiler-attempt-0.stderr').exists()
    assert (output / 'compiler-attempt-1.stderr').exists()

    bad_calls = []

    def invoke_bad(command, **kwargs):
        bad_calls.append(command)
        kwargs['stderr'].write('fatal schema mismatch in input IR')
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pipeline.subprocess, 'run', invoke_bad)
    bad_out = tmp_path / 'compile-nontransient'
    with pytest.raises(pipeline.PipelineError, match='fixed_compiler_failed'):
        pipeline.compile_pages([], bad_out, [])
    assert len(bad_calls) == 1
    assert not (bad_out / 'compile-retry-0.json').exists()


def test_submask_any_matches_full_canvas_slice():
    """Tight foreground sub-mask intersection helper matches full-canvas numpy mask slicing identically."""
    full = np.zeros((40, 60), dtype=bool)
    full[10:15, 20:30] = True
    entry = (20, 10, 10, 5, full[10:15, 20:30].copy())
    for box in ([0, 0, 20, 10], [30, 15, 10, 10], [18, 8, 2, 4], [25, 12, 8, 8], [22, 11, 2, 2]):
        x, y, w, h = box
        assert _submask_any(entry, box) is bool(full[y:y + h, x:x + w].any())


def test_office_bridge_unpack_streams_from_path_and_bytes(tmp_path):
    """Streaming Office packet unpack supports both Path inputs and in-memory bytes."""
    from product.runtime.office_bridge import pack, unpack

    src = tmp_path / 'src'
    (src / 'sub').mkdir(parents=True)
    (src / 'a.txt').write_text('alpha', 'utf8')
    (src / 'sub' / 'b.bin').write_bytes(b'\x00\x01\x02\x03' * 512)
    zpath = tmp_path / 'packet.zip'
    pack(src, zpath)

    out_path = tmp_path / 'out_from_path'
    unpack(zpath, out_path)
    out_bytes = tmp_path / 'out_from_bytes'
    unpack(zpath.read_bytes(), out_bytes)

    assert (out_path / 'a.txt').read_text('utf8') == 'alpha'
    assert (out_path / 'sub' / 'b.bin').read_bytes() == (src / 'sub' / 'b.bin').read_bytes()
    assert (out_bytes / 'a.txt').read_text('utf8') == 'alpha'
    assert (out_bytes / 'sub' / 'b.bin').read_bytes() == (src / 'sub' / 'b.bin').read_bytes()
