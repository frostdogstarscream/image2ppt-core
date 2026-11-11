"""Explicit hash-bound replay of recorded three-branch perception, before fusion."""
import json
from pathlib import Path

from ..materialize import digest
from ..parallel_contracts import SceneGraph


def load(manifest_file,snap):
    manifest_file=Path(manifest_file)
    manifest=json.loads(manifest_file.read_text('utf8'))
    if manifest.get('schema')!='PerceptionReplay/1':raise ValueError('unsupported perception replay')
    row=next(r for r in manifest['pages'] if r['source_sha256']==snap['source_sha256'])
    for name in ('scene','ocr'):
        if digest(row[name]['path'])!=row[name]['sha256']:raise ValueError('perception replay evidence hash mismatch')
    graph=SceneGraph.model_validate_json(Path(row['scene']['path']).read_text('utf8'))
    ocr=json.loads(Path(row['ocr']['path']).read_text('utf8'))
    if graph.source_sha256!=snap['source_sha256'] or graph.image_size!=snap['image_size'] or ocr['source_sha256']!=snap['source_sha256']:
        raise ValueError('perception replay source mismatch')
    return {'text':graph.text,'asset':graph.asset,'surface':graph.surface},ocr
