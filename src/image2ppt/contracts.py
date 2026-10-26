from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FileRecord(Record):
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(ge=0)


class AuthoringRecord(Record):
    """Provenance for a hand-authored SlideIR.

    The model step runs outside this project, so the only thing the pipeline can
    check is that the record still describes the IR it claims to: `ir_sha256`
    must equal the digest of the neighbouring slideir.json. Editing the IR
    without re-recording is the failure this exists to catch.

    Three fields are recorded but not yet interpreted by the pipeline; they are
    the author's own record, kept so a reviewer can see the reasoning behind a
    page: `node_decisions` maps a node id to why it is native text or shape, an
    image fallback, or grouped as it is; `font_candidates` lists the typefaces
    considered before the one actually used; `unresolved_ids` names what the
    author could not settle from the source alone.

    `created_at` is when this record was written, not when the page was
    prepared. Refresh it together with `ir_sha256` when you re-record.
    """
    format_version: Literal["1"] = "1"
    tool: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=120)
    created_at: str = Field(min_length=1, max_length=40)
    ir_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    node_decisions: dict[str, str] = Field(default_factory=dict)
    font_candidates: list[str] = Field(default_factory=list)
    # Source-inferred font metadata emitted by the element authoring adapter.
    # Provenance only: this does not establish font identity or acceptance.
    model_typography: dict[str, dict[str, Any]] = Field(default_factory=dict)
    unresolved_ids: list[str] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class ProviderResult(Record):
    kind: Literal["manual", "replay"]
    ir_file: str
    previous_environment: dict[str, Any] | None = None
    model_record: Literal["not_provided"] = "not_provided"
    authoring: AuthoringRecord | None = None


class CheckResult(Record):
    id: str
    status: Literal["PASS", "FAIL", "REVIEW_REQUIRED", "NOT_RUN"]
    method: str
    targets: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class InputPackage(Record):
    directory: str
    files: dict[str, FileRecord]
    key: str = Field(pattern=r"^[a-f0-9]{64}$")


class StageRecord(Record):
    key: str = Field(pattern=r"^[a-f0-9]{64}$")
    elapsed_seconds: float = Field(ge=0)
    data: dict[str, Any]
    files: dict[str, FileRecord]
    cache_hit: bool


class RunConfig(Record):
    timeout_seconds: float = Field(gt=0, le=86400)
    network_calls: Literal[0] = 0
    project_root: str


class RunOrigin(Record):
    kind: Literal['manual', 'replay']
    file: str


class RunState(Record):
    format_version: Literal['1']
    status: Literal['PENDING', 'RUNNING', 'FAILED', 'COMPLETED']
    origin: RunOrigin
    config: RunConfig
    stages: dict[str, StageRecord]
    created_at: str
    input: InputPackage | None = None
    provider: ProviderResult | None = None
    build: str | None = None
    build_sha256: str | None = None
    error: dict[str, str] | None = None
    invalidated: list[dict[str, str]] = Field(default_factory=list)


class BuildManifest(Record):
    format_version: Literal["1"] = "1"
    core_version: str
    run_id: str
    revision: str
    execution_status: Literal["COMPLETED"] = "COMPLETED"
    quality_status: Literal["PASS", "REVIEW_REQUIRED", "FAIL"]
    input_key: str
    pipeline_key: str
    provider: ProviderResult
    environment: dict[str, Any]
    environment_differences: list[str] = Field(default_factory=list)
    config: dict[str, Any]
    stages: list[dict[str, Any]]
    files: dict[str, FileRecord]
    network_calls: Literal[0] = 0
    human_acceptance: Literal[False] = False
    created_at: str


class ProbeCase(Record):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    action: Literal["delete_text", "move_group", "change_fill"]
    targets: list[str] = Field(min_length=1)
    slide: int = Field(default=1, ge=1)
    delta_px: list[float] | None = None
    fill: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{6}$")
    roi: list[int] | None = None
    related_image: str | None = None
    expected_solid: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{6}$")
    synthetic: bool = False

    @model_validator(mode="after")
    def operation(self):
        import math
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate probe targets")
        if self.action == "move_group" and (not self.delta_px or len(self.delta_px) != 2 or not all(math.isfinite(v) and abs(v) <= 8192 for v in self.delta_px)):
            raise ValueError("move_group needs two bounded finite delta_px values")
        if self.action == "change_fill" and not self.fill:
            raise ValueError("change_fill needs fill")
        if self.roi is not None and (len(self.roi) != 4 or min(self.roi[:2]) < 0 or min(self.roi[2:]) <= 0):
            raise ValueError("roi must be positive target-pixel [x,y,width,height]")
        if self.expected_solid and (not self.synthetic or self.roi is None):
            raise ValueError("pixel pass/fail requires a synthetic labelled ROI")
        return self


class ProbeCases(Record):
    format_version: Literal["1"] = "1"
    cases: list[ProbeCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate case IDs")
        return self


class SuitePage(Record):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    ir: str
    baseline_files: dict[str, FileRecord]
    cases: str | None = None


class RegressionSuite(Record):
    format_version: Literal["1"] = "1"
    pages: list[SuitePage] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        if len({p.id for p in self.pages}) != len(self.pages):
            raise ValueError("duplicate page IDs")
        return self
