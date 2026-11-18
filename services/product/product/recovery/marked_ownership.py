"""Synchronize page-wide masks with the individual source-owner receipts."""

import json
import numpy as np

from ..materialize import digest, write


def record_claims(directory, records):
    path = directory / 'ownership.npz'
    meta = directory / 'ownership.json'
    editable = directory / 'editable-mask.npz'
    with np.load(path, allow_pickle=False) as data:
        maps = {k: data[k] for k in data.files}
    with np.load(editable, allow_pickle=False) as data:
        claimed = data['mask'].astype(bool)
    for record in records:
        ref = record['owner_evidence']
        proof = directory / ref['file']
        if digest(proof) != ref['sha256']:
            raise ValueError('invalid source-owner receipt')
        with np.load(proof, allow_pickle=False) as data:
            mask = data['mask'].astype(bool)
        claimed |= mask
        maps['ownership'][mask] = 2 if record['kind'] in ('surface', 'table') else 0
        maps['uncertain_text'][mask] = False
    np.savez_compressed(editable, mask=claimed)
    np.savez_compressed(path, **maps)
    write(meta, {**json.loads(meta.read_text(encoding='utf8')), 'map_sha256': digest(path)})
