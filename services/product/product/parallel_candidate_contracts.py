"""Private r11 receipts; a precheck is never a publication approval."""
from typing import Literal
from pydantic import Field, model_validator
from .contracts import Strict


def candidate_path(root,candidate_id):
    """Keep full SHA identities in receipts, bounded directory names on Windows."""
    from pathlib import Path
    root=Path(root)
    old=root/candidate_id
    short=root/candidate_id[:26]
    return old if old.exists() and not short.exists() else short

Gate = Literal['PASS', 'FAIL', 'UNKNOWN']
State = Literal['PROPOSED', 'PRECHECK_BLOCKED', 'PRECHECK_UNKNOWN', 'PRECHECK_PASS',
                'STAGED', 'VERIFIED', 'RETRYABLE', 'REJECTED', 'REVIEW_UNKNOWN', 'SELECTED_FOR_FINAL']


class PreflightResult(Strict):
    schema_version: Literal['PreflightResult/1'] = Field(default='PreflightResult/1', alias='schema')
    candidate_id: str
    status: Gate
    checks: dict[str, Gate]
    unknown_pixels: int | None = Field(default=None, ge=0)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_hashes: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def fail_closed(self):
        required = {'source_binding', 'content_verification', 'owner_coverage',
                    'protected_peer_integrity', 'background_proof', 'region_atomicity'}
        if set(self.checks) != required:
            raise ValueError('incomplete hard checks')
        expected = 'FAIL' if 'FAIL' in self.checks.values() else 'UNKNOWN' if 'UNKNOWN' in self.checks.values() else 'PASS'
        if self.status != expected or (self.status == 'PASS' and self.unknown_pixels != 0):
            raise ValueError('preflight status contradicts evidence')
        return self


class ReconstructionCandidate(Strict):
    schema_version: Literal['ReconstructionCandidate/1'] = Field(default='ReconstructionCandidate/1', alias='schema')
    candidate_id: str = Field(pattern=r'^c-[a-f0-9]{64}$')
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    scene_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    region_id: str
    object_ids: list[str] = Field(min_length=1)
    representation: Literal['NATIVE_TEXT', 'NATIVE_SHAPE', 'TRANSPARENT_PNG', 'HYBRID', 'RASTER_FALLBACK']
    strategy: str
    references: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, str]
    status: State = 'PROPOSED'
    hard_gate: Gate = 'UNKNOWN'
    issues: list[str] = Field(default_factory=list)


def transition(candidate, state):
    allowed = {'PROPOSED': {'PRECHECK_BLOCKED', 'PRECHECK_UNKNOWN', 'PRECHECK_PASS'},
               'PRECHECK_PASS': {'STAGED'}, 'STAGED': {'VERIFIED', 'RETRYABLE', 'REJECTED', 'REVIEW_UNKNOWN'},
               'VERIFIED': {'SELECTED_FOR_FINAL'}}
    if state not in allowed.get(candidate.status, set()):
        raise ValueError('illegal candidate transition')
    return candidate.model_copy(update={'status': state})
