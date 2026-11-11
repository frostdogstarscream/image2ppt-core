"""Existing multipart Big LaMa service with durable, non-retrying reservations."""
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from ..materialize import digest, write
from ..parallel_store import key
from ..resource_lease import lease
from .background_adapter import RecoveryBlocked, _sha


class LamaClient:
    def __init__(self, url, model_binding, cache, timeout=300):
        endpoint = urlsplit(url)
        if (endpoint.scheme not in ('http', 'https') or not endpoint.hostname
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError('Big LaMa requires an HTTP(S) URL without credentials or query')
        if not model_binding or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Big LaMa model binding and positive timeout are required')
        self.url = url
        self.model_binding = model_binding
        self.cache = Path(cache)
        self.timeout = timeout
        self.receipt = None

    def __call__(self, source, mask):
        self.receipt = None
        if not isinstance(source, np.ndarray) or source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise RecoveryBlocked('INVALID_SOURCE', 'Big LaMa requires uint8 RGB')
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != source.shape[:2] or not mask.any():
            raise RecoveryBlocked('INVALID_MASK', 'Big LaMa requires nonempty source-sized mask')
        from ..object_extract import ROOT
        from text_mask import remote_inpaint
        identity = {
            'version': 'r12-big-lama/1', 'adapter_sha256': digest(__file__),
            'transport_sha256': digest(ROOT / 'ppt-rebuild/src/text_mask.py'),
            'source_pixels_sha256': _sha(source), 'mask_sha256': _sha(mask),
            'endpoint_sha256': key(self.url), 'model_binding': self.model_binding,
        }
        request_id = key(identity)
        directory = self.cache / request_id
        self.cache.mkdir(parents=True, exist_ok=True)
        with lease('r12-lama-' + request_id, timeout=self.timeout):
            receipt_file = directory / 'request.json'
            if directory.exists():
                receipt = json.loads(receipt_file.read_text('utf8')) if receipt_file.exists() else {}
                if receipt.get('status') != 'COMPLETED':
                    raise RecoveryBlocked('INPAINT_RESPONSE_UNKNOWN', 'Identical unknown request will not be resent')
                if receipt.get('identity') != identity or digest(directory / 'response.png') != receipt['response_sha256']:
                    raise RecoveryBlocked('INPAINT_CACHE_MISMATCH', 'Cached response binding changed')
            else:
                directory.mkdir()
                receipt = {'identity': identity, 'request_id': request_id, 'status': 'RESERVED',
                           'automatic_retry': False, 'quality': 'UNKNOWN'}
                write(receipt_file, receipt)
                try:
                    response = remote_inpaint(source.copy(), mask.copy(), self.url, timeout=self.timeout)
                    if response.dtype != np.uint8 or response.shape != source.shape:
                        raise ValueError('Invalid Big LaMa response dimensions')
                    Image.fromarray(response).save(directory / 'response.png')
                    receipt.update(status='COMPLETED', response_sha256=digest(directory / 'response.png'))
                    write(receipt_file, receipt)
                except Exception as error:
                    receipt.update(status='UNKNOWN', error_type=type(error).__name__)
                    write(receipt_file, receipt)
                    raise RecoveryBlocked('INPAINT_RESPONSE_UNKNOWN', 'Big LaMa response unavailable or invalid') from None
            response = np.array(Image.open(directory / 'response.png').convert('RGB'))
            if response.shape != source.shape:
                raise RecoveryBlocked('INPAINT_CACHE_MISMATCH', 'Cached response dimensions changed')
            self.receipt = {'request_id': request_id, 'receipt_file': str(receipt_file.resolve()),
                            'receipt_sha256': digest(receipt_file), 'model_binding': self.model_binding}
            return response
