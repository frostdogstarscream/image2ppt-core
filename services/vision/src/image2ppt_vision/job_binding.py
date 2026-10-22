"""Integrity checks shared by the region-job sender and the offline applier.

The checks establish that the bytes sent are the bytes prepared, for the
endpoint and model recorded in the binding. They are file-integrity and
version-association checks, not a signature against an actor who can rewrite
both the evidence and its manifest.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from pathlib import Path

from .model_provider import _endpoint

MAX_REQUEST_BYTES = 20_000_000
MAX_CROP_BYTES = 20_000_000
MAX_JSON_BYTES = 1_000_000
HASH_RE = re.compile(r"^[a-f0-9]{64}$")

BINDING_KEYS = {
    "schema", "region_id", "page", "node_id", "bbox", "classification",
    "features", "crop_size", "inventory_sha256", "features_sha256",
    "crop_sha256", "capabilities_sha256", "request_sha256", "prompt_version",
    "prompt_sha256", "endpoint_sha256", "model",
}


class JobBindingError(ValueError):
    """A job failed local integrity checks; the code is safe to log."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def hash_file(path):
    return hash_bytes(Path(path).read_bytes())


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def binding_sha256(binding):
    return hash_bytes(canonical_json_bytes(binding))


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path):
    raw = Path(path).read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("json too large")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_no_duplicate_keys,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("json object required")
    return value


def _hash_field(binding, key):
    value = binding.get(key)
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise JobBindingError("invalid_binding")
    return value


def _string_field(binding, key):
    value = binding.get(key)
    if not isinstance(value, str) or not value:
        raise JobBindingError("invalid_binding")
    return value


def _check_binding(binding):
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
        raise JobBindingError("invalid_binding")
    if binding.get("schema") != "RegionJob/1":
        raise JobBindingError("invalid_binding")
    for key in ("region_id", "page", "node_id", "classification", "prompt_version", "model"):
        _string_field(binding, key)
    for key in ("inventory_sha256", "features_sha256", "crop_sha256", "capabilities_sha256",
                "request_sha256", "prompt_sha256", "endpoint_sha256"):
        _hash_field(binding, key)
    bbox = binding.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or any(
            type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in bbox):
        raise JobBindingError("invalid_binding")
    features = binding.get("features")
    if not isinstance(features, dict) or not features:
        raise JobBindingError("invalid_binding")
    crop_size = binding.get("crop_size")
    if (not isinstance(crop_size, list) or len(crop_size) != 2
            or any(type(value) is not int or value <= 0 for value in crop_size)):
        raise JobBindingError("invalid_binding")


def _verify_request(request_bytes, binding, crop_bytes):
    try:
        request = _read_bytes_json(request_bytes)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise JobBindingError("invalid_request") from exc
    if request.get("model") != binding["model"]:
        raise JobBindingError("request_model_mismatch")
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise JobBindingError("invalid_request")
    system, user = messages
    if not isinstance(system, dict) or system.get("role") != "system" or not isinstance(system.get("content"), str):
        raise JobBindingError("invalid_request")
    if hash_bytes(system["content"].encode("utf-8")) != binding["prompt_sha256"]:
        raise JobBindingError("request_prompt_mismatch")
    if not isinstance(user, dict) or user.get("role") != "user" or not isinstance(user.get("content"), list):
        raise JobBindingError("invalid_request")
    parts = user["content"]
    if len(parts) != 2 or not all(isinstance(part, dict) for part in parts):
        raise JobBindingError("invalid_request")
    text_part, image_part = parts
    if text_part.get("type") != "text" or not isinstance(text_part.get("text"), str):
        raise JobBindingError("invalid_request")
    try:
        evidence = _read_bytes_json(text_part["text"].encode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise JobBindingError("invalid_request") from exc
    if evidence.get("instruction_version") != binding["prompt_version"]:
        raise JobBindingError("request_prompt_mismatch")
    if evidence.get("classification") != binding["classification"]:
        raise JobBindingError("request_classification_mismatch")
    if canonical_json_bytes(evidence.get("features")) != canonical_json_bytes(binding["features"]):
        raise JobBindingError("request_features_mismatch")
    if hash_bytes(canonical_json_bytes(evidence.get("capabilities"))) != binding["capabilities_sha256"]:
        raise JobBindingError("request_capabilities_mismatch")
    image = image_part.get("image_url")
    if image_part.get("type") != "image_url" or not isinstance(image, dict) or not isinstance(image.get("url"), str):
        raise JobBindingError("invalid_request")
    prefix = "data:image/png;base64,"
    if not image["url"].startswith(prefix):
        raise JobBindingError("invalid_request")
    try:
        sent_crop = base64.b64decode(image["url"][len(prefix):], validate=True)
    except (ValueError, TypeError) as exc:
        raise JobBindingError("invalid_request") from exc
    if sent_crop != crop_bytes:
        raise JobBindingError("request_crop_mismatch")


def _read_bytes_json(raw):
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_no_duplicate_keys,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("json object required")
    return value


def verify_job(job_dir, settings_file=None):
    """Verify one prepared job before sending or before accepting a response.

    Returns the binding, exact request bytes, settings and derived fingerprints.
    Raises JobBindingError with a stable code on any mismatch.
    """
    job = Path(job_dir)
    try:
        binding = _read_json(job / "input.json")
        _check_binding(binding)
        request_bytes = (job / "request.json").read_bytes()
        crop_bytes = (job / "crop.png").read_bytes()
        settings = _read_json(settings_file if settings_file is not None else job / "settings.json")
    except JobBindingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise JobBindingError("job_unreadable") from exc
    if len(request_bytes) > MAX_REQUEST_BYTES or len(crop_bytes) > MAX_CROP_BYTES:
        raise JobBindingError("job_too_large")
    if hash_bytes(request_bytes) != binding["request_sha256"]:
        raise JobBindingError("request_hash_mismatch")
    if hash_bytes(crop_bytes) != binding["crop_sha256"]:
        raise JobBindingError("crop_hash_mismatch")
    _verify_request(request_bytes, binding, crop_bytes)

    base_url = settings.get("base_url")
    timeout = settings.get("timeout_seconds")
    if not isinstance(base_url, str) or type(timeout) is bool or not isinstance(timeout, (int, float)) \
            or not math.isfinite(timeout) or not 1 <= timeout <= 1200:
        raise JobBindingError("invalid_settings")
    if set(settings) - {"base_url", "timeout_seconds"}:
        raise JobBindingError("invalid_settings")
    try:
        endpoint = _endpoint(base_url)
    except Exception as exc:  # model_provider raises its own public error
        raise JobBindingError("invalid_endpoint") from exc
    endpoint_sha = hash_bytes(endpoint.encode("utf-8"))
    if endpoint_sha != binding["endpoint_sha256"]:
        raise JobBindingError("endpoint_mismatch")

    return {
        "binding": binding,
        "request_bytes": request_bytes,
        "settings": settings,
        "request_sha256": binding["request_sha256"],
        "endpoint_sha256": endpoint_sha,
        "job_binding_sha256": binding_sha256(binding),
        "model": binding["model"],
        "prompt_version": binding["prompt_version"],
    }
