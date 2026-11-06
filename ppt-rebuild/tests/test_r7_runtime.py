"""Actual selected runtime bytes, not package version labels, guard reuse."""
import json
from pathlib import Path

import pytest
from _helpers import SRC
import ocr_engine
import r7_runtime as runtime


def put(path, content=b'fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def package(modules, name, *, dependencies=None, optional=None):
    root = modules / name
    put(root / 'package.json', json.dumps({'name': name, 'version': 'same-version',
        'dependencies': dependencies or {}, 'optionalDependencies': optional or {}}).encode())
    put(root / 'index.mjs')
    return root


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'isolated'; root.mkdir()
    bundle = tmp_path / 'actual-external-bundle'
    modules = bundle / 'node_modules'
    artifact = package(modules, '@oai/artifact-tool', dependencies={'bundled': '1'})
    package(artifact / 'node_modules', 'bundled')
    put(artifact / 'engine.wasm')
    canvas = package(modules, '@napi-rs/canvas', optional={
        '@napi-rs/canvas-win32-x64': '1', '@napi-rs/canvas-linux-arm64': '1'})
    native = package(modules, '@napi-rs/canvas-win32-x64')
    put(native / 'native.node')
    node = put(bundle / 'node.exe')
    main = put(bundle / 'main-python/python.exe')
    helper = put(bundle / 'helper-python/python.exe')
    put(main.parent / 'python312.dll')
    stdlib = bundle / 'stdlib'; put(stdlib / 'json/__init__.py')
    skill = bundle / 'skill'
    for name in ('artifact_tool_utils.mjs', 'runtime_helpers.mjs',
                 'inspect_presentation_package_integrity.py', 'inspect_presentation_layout_geometry.py',
                 'materialize_literal_chart_workbooks.py'):
        put(skill / 'container_tools' / name)
    put(root / 'ppt-rebuild/compiler_bridge.mjs')
    models = bundle / 'ocr-models'
    for name in ('det.onnx', 'cls.onnx', 'rec.onnx'):
        put(models / name)
    python_modules = {}
    for name in runtime.MAIN_REQUIRED:
        folder = bundle / 'python-packages' / name
        python_modules[name] = {'origin': str(put(folder / '__init__.py')), 'roots': [str(folder)]}
    def python_probe(executable):
        return {'executable': str(executable), 'version': 'test-python', 'prefix': str(Path(executable).parent),
                'base_prefix': str(Path(executable).parent), 'stdlib': str(stdlib), 'modules': python_modules}
    monkeypatch.setattr(runtime, '_python_probe', python_probe)
    monkeypatch.setattr(runtime.sys, 'executable', str(main))
    monkeypatch.setattr(ocr_engine, 'MODELS', models)
    class Backend:
        def __init__(self):
            self.root = root; self.node = node
            self.info = {'node_modules': str(modules), 'compiler_python': str(helper),
                         'presentations_skill': str(skill), 'engine': '@oai/artifact-tool',
                         'artifact_version': 'same-version', 'node_version': 'test-node'}
        def invoke(self, *args):
            assert args == ('core_bridge.mjs', 'doctor')
            return self.info.copy()
    return Backend(), {'artifact': artifact, 'native': native, 'skill': skill, 'main': main,
                        'helper': helper, 'models': models, 'canvas': canvas, 'modules': modules}


def test_external_selected_runtime_is_frozen_without_checkout_runtime_folder(tmp_path, monkeypatch):
    backend, paths = fixture(tmp_path, monkeypatch)
    first = runtime.freeze_runtime(backend)
    second = runtime.freeze_runtime(backend)
    assert first == second
    assert not (backend.root / '.runtime').exists()
    assert first['python_compiler']['executable'] == str(paths['helper'].resolve())
    assert str(paths['helper'].resolve()) in first['files']
    assert str((paths['native'] / 'native.node').resolve()) in first['files']
    assert not first['runtime_config']['present']
    assert first['node_packages']['roots']['skia-canvas'] is None
    assert first['node_packages']['missing_optional'] == {
        str(paths['canvas'].resolve()): ['@napi-rs/canvas-linux-arm64']}


@pytest.mark.parametrize('target', ['engine', 'canvas_native', 'helper_script', 'bridge',
                                  'python_dll', 'helper_python', 'ocr', 'python_package'])
def test_same_version_byte_changes_invalidate_freeze(tmp_path, monkeypatch, target):
    backend, paths = fixture(tmp_path, monkeypatch)
    first = runtime.freeze_runtime(backend)
    files = {'engine': paths['artifact'] / 'engine.wasm', 'canvas_native': paths['native'] / 'native.node',
             'helper_script': paths['skill'] / 'container_tools/runtime_helpers.mjs',
             'bridge': backend.root / 'ppt-rebuild/compiler_bridge.mjs',
             'python_dll': paths['main'].parent / 'python312.dll', 'helper_python': paths['helper'],
             'ocr': paths['models'] / 'rec.onnx',
             'python_package': paths['main'].parent.parent / 'python-packages/numpy/__init__.py'}
    put(files[target], b'changed-runtime-bytes')
    second = runtime.freeze_runtime(backend)
    assert first['artifact_version'] == second['artifact_version']
    assert first['sha256'] != second['sha256']


@pytest.mark.parametrize('target', ['node', 'helper', 'ocr', 'helper_script', 'bridge'])
def test_selected_mandatory_file_cannot_be_silently_skipped(tmp_path, monkeypatch, target):
    backend, paths = fixture(tmp_path, monkeypatch)
    files = {'node': backend.node, 'helper': paths['helper'], 'ocr': paths['models'] / 'det.onnx',
             'helper_script': paths['skill'] / 'container_tools/runtime_helpers.mjs',
             'bridge': backend.root / 'ppt-rebuild/compiler_bridge.mjs'}
    files[target].unlink()
    with pytest.raises(ValueError, match='missing'):
        runtime.freeze_runtime(backend)


def test_runtime_configuration_presence_and_content_are_frozen(tmp_path, monkeypatch):
    backend, _ = fixture(tmp_path, monkeypatch)
    before = runtime.freeze_runtime(backend)
    put(backend.root / 'runtime.local.json', b'{"python":"external"}')
    after = runtime.freeze_runtime(backend)
    assert after['runtime_config']['present'] and before['sha256'] != after['sha256']
    put(backend.root / 'runtime.local.json', b'{"python":"other"}')
    assert runtime.freeze_runtime(backend)['sha256'] != after['sha256']


def test_generated_python_cache_does_not_change_runtime_identity(tmp_path, monkeypatch):
    backend, paths = fixture(tmp_path, monkeypatch)
    before = runtime.freeze_runtime(backend)
    put(paths['skill'] / 'container_tools/__pycache__/helper.pyc', b'cache')
    put(paths['artifact'] / '__pycache__/helper.pyc', b'cache')
    assert runtime.freeze_runtime(backend) == before


def test_tree_follows_symlink_once_without_cycles(tmp_path):
    root = tmp_path / 'package'; put(root / 'code.mjs')
    try:
        (root / 'loop').symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip('Host does not permit test directory symlinks')
    files = {}; record = runtime._tree(root, files)
    assert record['file_count'] == 1 and len(files) == 1


def test_artifact_bundle_without_top_level_canvas_is_valid(tmp_path):
    modules = tmp_path / 'node_modules'
    artifact = package(modules, '@oai/artifact-tool')
    engine = put(artifact / 'renderer/engine.wasm')
    files = {}
    record = runtime._node_packages(modules, files)
    assert record['roots'] == {'@oai/artifact-tool': str(artifact.resolve()),
                               'skia-canvas': None, '@napi-rs/canvas': None}
    assert str(engine.resolve()) in files
    assert str(artifact.resolve()) in record['packages']


def test_nested_renderer_native_dependency_remains_frozen(tmp_path):
    modules = tmp_path / 'node_modules'
    artifact = package(modules, '@oai/artifact-tool', dependencies={'renderer': '1'})
    renderer = package(artifact / 'node_modules', 'renderer', dependencies={'renderer-native': '1'})
    native = package(renderer / 'node_modules', 'renderer-native')
    binary = put(native / 'render.node')
    before = {}
    record = runtime._node_packages(modules, before)
    assert record['roots']['skia-canvas'] is None
    assert record['roots']['@napi-rs/canvas'] is None
    assert set(record['packages']) == {str(folder.resolve()) for folder in (artifact, renderer, native)}
    assert str(binary.resolve()) in before
    put(binary, b'changed-native-renderer-bytes')
    after = {}
    runtime._node_packages(modules, after)
    assert before[str(binary.resolve())] != after[str(binary.resolve())]
