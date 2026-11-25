"""Source-bound segmentation receipts and per-object service outcomes."""
import json
import os
from pathlib import Path
import socket
from urllib.error import HTTPError, URLError

from ..materialize import digest, write
from ..object_extract import load_sam
from ..object_registry import SAM_BINDING
from ..object_stage import ROOT, infer_sam
from ..parallel_store import key


def configured():
    return bool(os.environ.get('IMAGE2PPT_SAM_URL') or os.environ.get('IMAGE2PPT_SAM_CONFIG')
                or (ROOT / '.runtime/sam-client.json').is_file())


def prepare_mask(source, graph, asset, cache, *, offline=False):
    plan = {'schema':'LogicalObjects/2', 'source_sha256':graph.source_sha256, 'image_size':graph.image_size,
            'objects':[asset.model_dump()], 'limitations':[]}
    root = Path(cache) / key(plan)[:24]
    plan_file, masks = root / 'sam-plan.json', root / 'sam'
    existing = (masks / (asset.id + '.json')).is_file() and plan_file.is_file()

    def inspect():
        _, reason = load_sam(masks, asset, graph.source_sha256, digest(plan_file),
                             graph.image_size, 'sam3', individual=True, distinguish_empty=True)
        if reason in (None, 'sam_empty_mask', 'sam_no_owned_candidate'):
            meta = json.loads((masks/(asset.id+'.json')).read_text('utf8'))
            if meta.get('adapter') == 'PrivateSamReceipt/1' and any(
                    meta.get(k) != v for k,v in SAM_BINDING.items()):
                return 'sam_model_binding_mismatch'
        return reason

    via = 'cache'
    try:
        if existing:
            try:
                reason = inspect() if json.loads(plan_file.read_text('utf8')) == plan else 'sam_binding_mismatch'
            except (KeyError, ValueError, OSError):
                reason = 'sam_artifact_mismatch'
            if reason in (None, 'sam_empty_mask', 'sam_no_owned_candidate'):
                return plan_file, masks, {'status':reason or 'MASK_READY', 'via':via}
        if offline:
            return plan_file, masks, {'status':'SERVICE_OFFLINE', 'via':'batch_offline'}
        via = 'inference'
        root.mkdir(parents=True, exist_ok=True)
        write(plan_file, plan)
        if infer_sam(source, plan_file, masks, 'sam3', [asset.id]) is None:
            return plan_file, masks, {'status':'REQUEST_FAILED', 'via':via}
        reason = inspect()
        return plan_file, masks, {'status':reason or 'MASK_READY', 'via':via}
    except HTTPError as exc:
        outcome = {'status':'REQUEST_FAILED', 'reason':f'HTTP_{exc.code}'}
    except (TimeoutError, URLError, ConnectionError, socket.gaierror) as exc:
        cause = getattr(exc, 'reason', exc)
        outcome = {'status':'REQUEST_TIMEOUT' if isinstance(cause, TimeoutError) else 'SERVICE_OFFLINE',
                   'reason':type(cause).__name__}
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        status = 'SERVICE_OFFLINE' if str(exc) == 'private SAM worker unavailable' else 'REQUEST_FAILED'
        outcome = {'status':status, 'reason':type(exc).__name__}
    return plan_file, masks, {**outcome, 'via':via}
