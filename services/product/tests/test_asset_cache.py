"""Cache invalidation tests use synthetic inputs and stub external Office."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from product.materialize import digest
from product.parallel_store import key
from product.recovery import asset_cache, single_asset
from test_r1205_agent_asset_stage import scene


def test_compiler_fingerprint_tracks_transitive_package_bytes(tmp_path, monkeypatch):
    engine = tmp_path/'modules/@oai/artifact-tool'
    engine.mkdir(parents=True)
    (engine/'index.js').write_text('export const version = 1')
    dependency = tmp_path/'modules/skia-canvas'
    dependency.mkdir()
    binary = dependency/'renderer.node'
    binary.write_bytes(b'renderer version one')
    executable = tmp_path/'runtime.exe'
    executable.write_bytes(b'fixture executable')
    runtime = {'modules': str(tmp_path/'modules'), 'skill': str(tmp_path/'skill'),
               'python': str(executable), 'version': 'fixture',
               'packages': [str(engine), str(dependency)], 'missing': []}
    monkeypatch.setattr('product.pipeline.compiler_runtime', lambda: (tmp_path, executable))
    monkeypatch.setattr(asset_cache.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps(runtime)))
    class RegistryHandle:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setitem(sys.modules, 'winreg', SimpleNamespace(
        HKEY_LOCAL_MACHINE=0, OpenKey=lambda *a: RegistryHandle(),
        QueryValue=lambda *a: str(executable)))
    one = asset_cache.compiler_runtime_fingerprint()
    assert str(binary.resolve()) in one['dependency_files']
    binary.write_bytes(b'renderer version two')
    two = asset_cache.compiler_runtime_fingerprint()
    assert one != two


def test_local_dependency_closure_includes_added_changed_and_removed_modules(tmp_path):
    helper = tmp_path/'services/product/product/transitive_helper.py'
    helper.parent.mkdir(parents=True)
    helper.write_text('VALUE = 1')
    original = asset_cache.code_fingerprint(tmp_path)
    helper.write_text('VALUE = 2')
    assert asset_cache.code_fingerprint(tmp_path) != original
    dependency = tmp_path/'ppt-rebuild/reconstruction.py'
    dependency.parent.mkdir()
    dependency.write_text('VALUE = 1')
    added = asset_cache.code_fingerprint(tmp_path)
    assert str(dependency.resolve()) in added
    dependency.unlink()
    assert asset_cache.code_fingerprint(tmp_path) != added


@pytest.mark.parametrize('change', ['dependency', 'layer_manifest'])
def test_l1_invalidates_and_reuses_identical_inputs(tmp_path, monkeypatch, change):
    graph, layers, baseline = scene(tmp_path)
    helper = tmp_path/'indirect.py'
    helper.write_text('VALUE = 1')
    monkeypatch.setattr(asset_cache, 'code_fingerprint', lambda: {str(helper): digest(helper)})
    output = tmp_path/'probe'
    first = single_asset.probe(graph, 'first', layers/'source.png', layers, baseline, output, strategy='source-halo')
    assert first['status'] == 'CANDIDATE_READY'
    assert single_asset.probe(graph, 'first', layers/'source.png', layers, baseline, output,
                              strategy='source-halo')['cache_hit']
    file = helper if change == 'dependency' else layers/'manifest.json'
    file.write_bytes(file.read_bytes()+b'\n')
    second = single_asset.probe(graph, 'first', layers/'source.png', layers, baseline, output, strategy='source-halo')
    assert second['status'] == 'CANDIDATE_READY' and not second['cache_hit']
    assert second['receipt'] != first['receipt']


@pytest.mark.parametrize('change', ['font_configuration', 'font_bytes', 'layer_manifest',
                                   'source', 'scene', 'dependency', 'compiler_runtime'])
def test_l2_fingerprint_covers_all_editing_dependencies(tmp_path, monkeypatch, change):
    graph, layers, baseline = scene(tmp_path)
    result = single_asset.probe(graph, 'first', layers/'source.png', layers, baseline,
                                tmp_path/'probe', strategy='source-halo')
    font = tmp_path/'font.ttf'
    font.write_bytes(b'font version 1')
    fonts = {'fonts': [{'file': str(font), 'family': 'Face', 'bold': False}]}
    dependency = tmp_path/'office_implementation.py'
    dependency.write_text('VERSION = 1')
    runtime = {'version': 'one'}
    monkeypatch.setattr(asset_cache, 'code_fingerprint', lambda **kwargs: {str(dependency): digest(dependency)})
    monkeypatch.setattr(asset_cache, 'compiler_runtime_fingerprint', lambda: dict(runtime))
    def binding():
        return asset_cache.l2_binding(result, graph, layers/'source.png', layers, baseline, fonts)
    before = key(binding())
    assert key(binding()) == before
    if change == 'font_configuration':
        fonts['fonts'][0]['bold'] = True
    elif change == 'font_bytes':
        font.write_bytes(b'font version 2')
    elif change == 'layer_manifest':
        manifest = layers/'manifest.json'
        manifest.write_bytes(manifest.read_bytes()+b'\n')
    elif change == 'source':
        source = layers/'source.png'
        source.write_bytes(source.read_bytes()+b'\n')
    elif change == 'scene':
        graph.ownership['second'] = 'AMBIGUOUS'
    elif change == 'dependency':
        dependency.write_text('VERSION = 2')
    else:
        runtime['version'] = 'two'
    assert key(binding()) != before


def test_l2_identical_inputs_hit_cache_without_recompiling(tmp_path, monkeypatch):
    graph, layers, baseline = scene(tmp_path)
    result = single_asset.probe(graph, 'first', layers/'source.png', layers, baseline,
                                tmp_path/'probe', strategy='source-halo')
    monkeypatch.setattr(asset_cache, 'compiler_runtime_fingerprint', lambda: {'version': 'fixture'})
    calls = []
    def compile_pages(irs, output, pages):
        output.mkdir()
        file = output/'result.pptx'
        file.write_bytes(b'synthetic compiler result')
        calls.append(file)
        return {'pptx': str(file)}
    monkeypatch.setattr('product.pipeline.compile_pages', compile_pages)
    monkeypatch.setattr('product.parallel_office.inspect', lambda *a, **kw: {'status': 'PASS_APP_AUTOMATED'})
    monkeypatch.setattr('product.parallel_residual_gate.inspect_residuals', lambda *a: [
        {'id': 'first', 'residual_check_status': 'PASS'}])
    one = single_asset.verify_office(result, graph, layers/'source.png', layers, baseline,
                                     {'fonts': []}, tmp_path/'office')
    two = single_asset.verify_office(result, graph, layers/'source.png', layers, baseline,
                                     {'fonts': []}, tmp_path/'office')
    assert one['status'] == 'PASS' and two['cache_hit'] and len(calls) == 1
