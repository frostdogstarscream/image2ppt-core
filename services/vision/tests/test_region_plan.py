import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from image2ppt_vision import api_eval_worker, region_plan
from image2ppt_vision.job_binding import binding_sha256


def row(region_id="P12-test", classification="chart"):
    return {
        "region_id": region_id,
        "page": "P12",
        "ir": "runs/test/slideir.json",
        "node_id": "industry-risk-bars",
        "bbox": [10, 20, 80, 40],
        "classification": classification,
        "features": {
            "area_px": 3200,
            "aspect": 2.0,
            "unique_colors": 4,
            "edge_density": 0.05,
            "dark_fraction": 0.1,
            "text_components": 5,
            "median_text_height_px": 12.0,
            "microtext_fraction": 0.0,
            "dominant_color_fraction": 0.8,
        },
    }


def crop(tmp_path):
    path = tmp_path / "crop.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 40), "white").save(path)
    return path


def agent(kind="native"):
    if kind == "native":
        return {
            "decision": "native",
            "confidence": 0.8,
            "can_rebuild": {"feasible": True, "native_parts": ["bars"], "blockers": []},
            "should_rebuild": {"editability_value": "medium", "fidelity_risk": "low", "cost": "low"},
            "native_parts": ["bars"],
            "retained_parts": [],
            "retain_reasons": [],
            "evidence": ["visible bars and labels"],
        }
    if kind == "hybrid":
        return {
            "decision": "hybrid",
            "confidence": 0.7,
            "can_rebuild": {"feasible": True, "native_parts": ["section_title"], "blockers": []},
            "should_rebuild": {"editability_value": "medium", "fidelity_risk": "high", "cost": "medium"},
            "native_parts": ["section_title"],
            "retained_parts": ["dashboard_body"],
            "retain_reasons": ["ui_mockup"],
            "evidence": ["title is native", "body is a dense UI"],
        }
    return {
        "decision": "retain_module",
        "confidence": 0.9,
        "can_rebuild": {"feasible": True, "native_parts": [], "blockers": ["dense microtext"]},
        "should_rebuild": {"editability_value": "low", "fidelity_risk": "high", "cost": "high"},
        "native_parts": [],
        "retained_parts": ["module_body"],
        "retain_reasons": ["ui_mockup", "microtext"],
        "evidence": ["dense UI screenshot"],
    }


def human_scope(kind="hybrid"):
    value = {
        "decision": kind,
        "source": "test",
        "note": "",
        "annotation_level": "explicit",
        "retain_scope": {"native": "none", "hybrid": "partial", "retain_module": "whole_region"}[kind],
        "fidelity_target": "pixel",
        "required_edit_targets": [],
        "allowed_native_parts": [],
        "must_retain_parts": [],
    }
    if kind == "hybrid":
        value.update(required_edit_targets=["section_title"],
                     allowed_native_parts=["section_title", "title_icon"],
                     must_retain_parts=["dashboard_body"])
    elif kind == "native":
        value.update(required_edit_targets=["bars", "labels"], allowed_native_parts=["bars", "labels"])
    return value


def load_shadow():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("region_shadow_fixture", root / "scripts/region_shadow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_rows(tmp_path, plan_rows, human_rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    plans_path = tmp_path / "plans.json"
    human_path = tmp_path / "human.json"
    output = tmp_path / "score.json"
    plans_path.write_text(json.dumps({"schema_version": "region-plan/0.1", "regions": plan_rows}), encoding="utf8")
    human_path.write_text(json.dumps({
        "schema_version": "region-human-decisions/0.2", "decisions": human_rows}), encoding="utf8")
    return load_shadow().score(plans_path, output, human_path)


def response_body(job, body):
    if isinstance(body, str):
        body = body.encode("utf8")
    (job / "response.body").write_bytes(body)
    binding = json.loads((job / "input.json").read_text(encoding="utf8"))
    digest = hashlib.sha256(body).hexdigest()
    transport = {
        "status": "RECEIVED",
        "request_sha256": binding["request_sha256"],
        "endpoint_sha256": binding["endpoint_sha256"],
        "job_binding_sha256": binding_sha256(binding),
        "response_sha256": digest,
        "saved_response_sha256": digest,
        "response_redacted": False,
        "response_bytes": len(body),
        "attempts": 1,
    }
    (job / "transport.json").write_text(json.dumps(transport), encoding="utf8")
    return transport


def response(job, decision):
    return response_body(job, json.dumps({
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision, ensure_ascii=False)}}]
    }))


def prepare(tmp_path, region="P12-test"):
    data = row(region)
    work = tmp_path / region
    result = region_plan.prepare_region_job(
        data, crop(tmp_path), work, model="fixture/model",
        base_url="https://example.invalid/v1", inventory_sha256="a" * 64)
    return data, work, result


def test_prepare_sends_only_measured_fields_and_binds_hashes(tmp_path):
    data, job, result = prepare(tmp_path)
    assert result["status"] == "PREPARED" and result["api_called"] is False
    request = json.loads((job / "request.json").read_text(encoding="utf8"))
    assert set(request) == {"model", "thinking", "messages", "max_tokens", "response_format"}
    user = request["messages"][1]["content"]
    assert len(user) == 2 and user[1]["type"] == "image_url"
    evidence = json.loads(user[0]["text"])
    assert set(evidence) == {"instruction_version", "classification", "features", "capabilities", "decision_contract"}
    serialized = json.dumps(request, ensure_ascii=False)
    for forbidden in ("region_id", "node_id", "bbox", "human", "description", "C:/", "D:/"):
        assert forbidden not in serialized
    binding = json.loads((job / "input.json").read_text(encoding="utf8"))
    assert binding["request_sha256"] == result["request_sha256"]
    assert binding["crop_sha256"] == result["crop_sha256"]
    assert binding["inventory_sha256"] == "a" * 64
    assert binding["model"] == "fixture/model"
    assert len(binding["prompt_sha256"]) == 64


def test_prepare_is_idempotent_and_stale_binding_never_overwrites(tmp_path):
    data, job, _ = prepare(tmp_path)
    again = region_plan.prepare_region_job(
        data, crop(tmp_path), job, model="fixture/model",
        base_url="https://example.invalid/v1", inventory_sha256="a" * 64)
    assert again["status"] == "CACHED"
    changed = region_plan.prepare_region_job(
        data, crop(tmp_path), job, model="other/model",
        base_url="https://example.invalid/v1", inventory_sha256="a" * 64)
    assert changed["status"] == "STALE"
    assert json.loads((job / "request.json").read_text(encoding="utf8"))["model"] == "fixture/model"


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(extra="x"),
    lambda a: a.update(confidence=True),
    lambda a: a.update(confidence=1.5),
    lambda a: a["should_rebuild"].update(editability_value="low"),
    lambda a: a.update(retained_parts=["module_body"]),
    lambda a: a.update(native_parts=["bars", "bars"]),
    lambda a: a.update(native_parts=["unknown_part"]),
    lambda a: a.update(evidence=[]),
    lambda a: a.update(evidence=["1", "2", "3", "4"]),
    lambda a: a["retain_reasons"].append("too_hard"),
])
def test_native_decision_rejects_invalid_combinations(mutate):
    value = agent("native")
    mutate(value)
    with pytest.raises(ValueError):
        region_plan.validate_decision(value)


def test_retain_and_hybrid_combinations_are_checked():
    region_plan.validate_decision(agent("hybrid"))
    region_plan.validate_decision(agent("retain_module"))
    bad = agent("retain_module")
    bad["native_parts"] = ["bars"]
    with pytest.raises(ValueError):
        region_plan.validate_decision(bad)


def test_parse_rejects_markdown_truncation_duplicate_keys_and_nan(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "response.body").write_text(json.dumps({
        "choices": [{"finish_reason": "length", "message": {"content": json.dumps(agent())}}]
    }), encoding="utf8")
    with pytest.raises(ValueError):
        region_plan.parse_response(job)
    (job / "response.body").write_text('{"choices":[],"choices":[]}', encoding="utf8")
    with pytest.raises(ValueError):
        region_plan.parse_response(job)
    content = json.dumps(agent()).replace('"confidence": 0.8', '"confidence": NaN')
    (job / "response.body").write_text(json.dumps({
        "choices": [{"finish_reason": "stop", "message": {"content": content}}]
    }), encoding="utf8")
    with pytest.raises(ValueError):
        region_plan.parse_response(job)
    (job / "response.body").write_text(json.dumps({
        "choices": [{"finish_reason": "stop", "message": {"content": "```json\\n{}\\n```"}}]
    }), encoding="utf8")
    with pytest.raises(ValueError):
        region_plan.parse_response(job)


def test_apply_preserves_human_decision_and_strict_partial(tmp_path):
    data, job, _ = prepare(tmp_path, "P12-test")
    response(job, agent("native"))
    human = {"P12-test": {"decision": "native", "source": "user_review", "note": "confirmed"}}
    result = region_plan.apply_jobs([data], tmp_path, human, inventory_sha256="a" * 64)
    assert result["status"] == "VALIDATED"
    assert result["plans"]["regions"][0]["human"]["decision"] == "native"
    assert result["report"]["shadow_only"] is True

    data2 = row("P8-test", "table")
    # No response for the second job -> strict refuses to write a plan.
    result2 = region_plan.apply_jobs([data, data2], tmp_path, human, inventory_sha256="a" * 64)
    assert result2["status"] == "INCOMPLETE" and result2["plans"] is None
    result3 = region_plan.apply_jobs([data, data2], tmp_path, human, inventory_sha256="a" * 64, strict=False)
    assert result3["status"] == "PARTIAL" and len(result3["plans"]["regions"]) == 1


def test_apply_rejects_tampered_request_and_transport_failure(tmp_path):
    data, job, _ = prepare(tmp_path)
    response(job, agent("native"))
    (job / "request.json").write_text("{}", encoding="utf8")
    tampered = region_plan.apply_jobs([data], tmp_path, {"P12-test": {"decision": "native"}},
                                      inventory_sha256="a" * 64)
    assert tampered["status"] == "INCOMPLETE"
    assert tampered["report"]["rows"][0]["status"] == "INVALID_BINDING"

    data, job, _ = prepare(tmp_path / "two")
    (job / "transport.json").write_text(json.dumps({"status": "FAILED", "error_code": "model_http_500"}), encoding="utf8")
    failed = region_plan.apply_jobs([data], tmp_path / "two", {"P12-test": {"decision": "native"}},
                                    inventory_sha256="a" * 64)
    assert failed["status"] == "INCOMPLETE"
    assert failed["report"]["rows"][0]["status"] == "TRANSPORT_FAILED"


def test_apply_rejects_tampered_or_missing_response_hashes(tmp_path):
    human = {"P12-test": {"decision": "native"}}
    data, job, _ = prepare(tmp_path / "body")
    response(job, agent("native"))
    (job / "response.body").write_bytes((job / "response.body").read_bytes() + b" ")
    changed = region_plan.apply_jobs([data], tmp_path / "body", human, inventory_sha256="a" * 64)
    assert changed["report"]["rows"][0]["status"] == "INVALID_RESPONSE"
    assert changed["report"]["rows"][0]["error_code"] == "response_hash_mismatch"

    data, job, _ = prepare(tmp_path / "missing")
    transport = response(job, agent("native"))
    transport.pop("saved_response_sha256")
    (job / "transport.json").write_text(json.dumps(transport), encoding="utf8")
    missing = region_plan.apply_jobs([data], tmp_path / "missing", human, inventory_sha256="a" * 64)
    assert missing["report"]["rows"][0]["status"] == "INVALID_RESPONSE"
    assert missing["report"]["rows"][0]["error_code"] == "response_hash_missing"


def test_apply_rejects_cross_job_response_and_changed_endpoint(tmp_path):
    data_a, job_a, _ = prepare(tmp_path / "swap", "P12-test")
    response(job_a, agent("native"))
    data_b, job_b, _ = prepare(tmp_path / "swap", "P8-test")
    response(job_b, agent("retain_module"))
    (job_b / "response.body").write_bytes((job_a / "response.body").read_bytes())
    (job_b / "transport.json").write_bytes((job_a / "transport.json").read_bytes())
    human_b = {"P8-test": {"decision": "retain_module"}}
    swapped = region_plan.apply_jobs([data_b], tmp_path / "swap", human_b, inventory_sha256="a" * 64)
    assert swapped["report"]["rows"][0]["status"] == "INVALID_RESPONSE"
    assert swapped["report"]["rows"][0]["error_code"] == "transport_job_binding_sha256_mismatch"

    data, job, _ = prepare(tmp_path / "endpoint")
    response(job, agent("native"))
    settings = json.loads((job / "settings.json").read_text(encoding="utf8"))
    settings["base_url"] = "https://other.invalid/v1"
    (job / "settings.json").write_text(json.dumps(settings), encoding="utf8")
    moved = region_plan.apply_jobs([data], tmp_path / "endpoint", {"P12-test": {"decision": "native"}},
                                   inventory_sha256="a" * 64)
    assert moved["report"]["rows"][0]["status"] == "INVALID_BINDING"
    assert moved["report"]["rows"][0]["error_code"] == "endpoint_mismatch"


@pytest.mark.parametrize("message", [None, [], "text"])
def test_parse_rejects_non_object_message_as_value_error(tmp_path, message):
    job = tmp_path / "job"
    job.mkdir()
    (job / "response.body").write_text(json.dumps({
        "choices": [{"finish_reason": "stop", "message": message}]
    }), encoding="utf8")
    with pytest.raises(ValueError):
        region_plan.parse_response(job)


def test_apply_isolates_bad_response_and_continues_batch(tmp_path):
    data_a, job_a, _ = prepare(tmp_path / "batch", "P12-test")
    response_body(job_a, json.dumps({"choices": [{"finish_reason": "stop", "message": None}]}))
    data_b, job_b, _ = prepare(tmp_path / "batch", "P8-test")
    response(job_b, agent("retain_module"))
    human = {
        "P12-test": {"decision": "native"},
        "P8-test": {"decision": "retain_module"},
    }
    result = region_plan.apply_jobs([data_a, data_b], tmp_path / "batch", human,
                                    inventory_sha256="a" * 64, strict=False)
    assert result["status"] == "PARTIAL"
    assert [row["status"] for row in result["report"]["rows"]] == ["INVALID_RESPONSE", "VALIDATED"]
    assert [row["region_id"] for row in result["plans"]["regions"]] == ["P8-test"]

    strict = region_plan.apply_jobs([data_a, data_b], tmp_path / "batch", human,
                                    inventory_sha256="a" * 64)
    assert strict["status"] == "INCOMPLETE" and strict["plans"] is None
    assert [row["status"] for row in strict["report"]["rows"]] == ["INVALID_RESPONSE", "VALIDATED"]


def test_schema_rejects_unknown_agent_fields():
    from jsonschema import Draft202012Validator
    schema = json.loads(Path(__file__).resolve().parents[3].joinpath("scripts/region_plan.schema.json").read_text(encoding="utf8"))
    value = agent("native")
    value["bbox"] = [0, 0, 1, 1]
    plan = {"schema_version": "region-plan/0.1", "regions": [{
        **row(), "agent": value,
        "human": {"decision": "native", "source": "test", "note": ""},
    }]}
    errors = list(Draft202012Validator(schema).iter_errors(plan))
    assert errors


def test_human_scope_annotation_is_coherent():
    value = human_scope("hybrid")
    checked = region_plan.validate_human_decision(value)
    assert checked["retain_scope"] == "partial"
    assert checked["required_edit_targets"] == ["section_title"]

    outside_allowed = human_scope("hybrid")
    outside_allowed["required_edit_targets"] = ["kpi"]
    with pytest.raises(ValueError):
        region_plan.validate_human_decision(outside_allowed)

    incomplete = human_scope("hybrid")
    incomplete.pop("fidelity_target")
    with pytest.raises(ValueError):
        region_plan.validate_human_decision(incomplete)

    retain_with_native = human_scope("retain_module")
    retain_with_native["allowed_native_parts"] = ["section_title"]
    with pytest.raises(ValueError):
        region_plan.validate_human_decision(retain_with_native)


def test_p5_same_routing_label_but_different_scope(tmp_path):
    plan_agent = agent("hybrid")
    native = ["section_title", "title_icon", "kpi", "table"]
    plan_agent["native_parts"] = native
    plan_agent["can_rebuild"]["native_parts"] = list(native)
    plan_agent["retained_parts"] = ["icons"]
    region_id = "P5-example-058-dashboard"
    plan_row = {**row(region_id), "page": "P5", "agent": plan_agent,
                "human": {key: human_scope("hybrid")[key] for key in ("decision", "source", "note")}}
    result = score_rows(tmp_path, [plan_row], {region_id: human_scope("hybrid")})
    assert result["routing"]["agreed"] == 1
    assert result["scope"]["agreed"] == 0
    item = result["rows"][0]["scope"]
    assert "kpi" in item["over_decomposition"]
    assert "dashboard_body" in item["retain_violations"]


def test_scope_detects_under_decomposition_and_legacy_annotation(tmp_path):
    region_id = "P12-test"
    plan_agent = agent("native")
    plan_agent["native_parts"] = ["bars"]
    plan_agent["can_rebuild"]["native_parts"] = ["bars"]
    plan_row = {**row(region_id), "agent": plan_agent, "human": {"decision": "native", "source": "test", "note": ""}}
    result = score_rows(tmp_path / "scope", [plan_row], {region_id: human_scope("native")})
    assert result["scope"]["agreed"] == 0
    assert result["rows"][0]["scope"]["under_decomposition"] == ["labels"]

    legacy = score_rows(tmp_path / "legacy", [plan_row], {
        region_id: {"decision": "native", "source": "test", "note": ""}})
    assert legacy["scope"]["decided"] == 0
    assert legacy["rows"][0]["scope"] == {"annotated": False, "agreement": None}


def test_worker_refuses_tampered_job_before_network(tmp_path, monkeypatch):
    _, job, _ = prepare(tmp_path)
    (job / "request.json").write_text("{}", encoding="utf8")
    monkeypatch.setenv("VISION_LLM_API_KEY", "fixture-key")
    api_eval_worker.exchange(job / "request.json", job / "settings.json", job)
    transport = json.loads((job / "transport.json").read_text(encoding="utf8"))
    assert transport["status"] == "FAILED"
    assert transport["error_code"] == "job_binding_mismatch"
    assert transport["binding_error"] == "request_hash_mismatch"


def test_schema_accepts_human_scope_and_rejects_unknown_fields():
    from jsonschema import Draft202012Validator
    schema = json.loads(Path(__file__).resolve().parents[3].joinpath("scripts/region_plan.schema.json").read_text(encoding="utf8"))
    plan = {"schema_version": "region-plan/0.1", "regions": [{
        **row(), "agent": agent("hybrid"), "human": human_scope("hybrid")}]}
    assert not list(Draft202012Validator(schema).iter_errors(plan))
    plan["regions"][0]["human"]["unexpected"] = "x"
    assert list(Draft202012Validator(schema).iter_errors(plan))


def test_schema_rejects_unknown_planner_fields():
    from jsonschema import Draft202012Validator
    schema = json.loads(Path(__file__).resolve().parents[3].joinpath("scripts/region_plan.schema.json").read_text(encoding="utf8"))
    plan = {
        "schema_version": "region-plan/0.1",
        "planner": {"kind": "llm", "prompt_version": "region-decision/1", "raw_artifacts": "local only"},
        "regions": [],
    }
    errors = list(Draft202012Validator(schema).iter_errors(plan))
    assert errors


def test_archived_plans_match_schema_and_validation_hash():
    from jsonschema import Draft202012Validator
    root = Path(__file__).resolve().parents[3]
    schema = json.loads((root / "scripts/region_plan.schema.json").read_text(encoding="utf8"))
    plans_path = root / "docs/region-shadow-llm/plans.llm.json"
    plans = json.loads(plans_path.read_text(encoding="utf8"))
    assert not list(Draft202012Validator(schema).iter_errors(plans))
    validation = json.loads((root / "docs/region-shadow-llm/validation.llm.json").read_text(encoding="utf8"))
    assert validation["valid"] is True
    assert validation["checked"]["plans_sha256"] == hashlib.sha256(plans_path.read_bytes()).hexdigest()


@pytest.mark.parametrize('scoped', [False, True])
def test_apply_validate_score_complete_flow(tmp_path, scoped):
    data, job, _ = prepare(tmp_path)
    response(job, agent('native'))
    human = human_scope('native') if scoped else {'decision': 'native', 'source': 'test', 'note': ''}
    applied = region_plan.apply_jobs([data], tmp_path, {data['region_id']: human}, inventory_sha256='a' * 64)
    assert applied['status'] == 'VALIDATED'
    shadow = load_shadow()
    # The semantic validator needs an inventory image node, not a compiler.
    ir = tmp_path / 'source-ir.json'
    shadow.write_json(ir, {'nodes': [{'id': data['node_id'], 'type': 'image', 'bbox': data['bbox']}]})
    applied['plans']['regions'][0]['ir'] = str(ir)
    plans, humans = tmp_path / 'plans.json', tmp_path / 'humans.json'
    shadow.write_json(plans, applied['plans'])
    shadow.write_json(humans, {'decisions': {data['region_id']: human}})
    assert shadow.validate(plans, tmp_path / 'validation.json', humans)['valid']
    scored = shadow.score(plans, tmp_path / 'score.json', humans)
    assert scored['routing']['agreed'] == 1
    assert scored['scope']['decided'] == int(scoped)
    if scoped:
        # Omitted scope must not be silently compared as a legacy decision.
        applied['plans']['regions'][0]['human'] = {k: human[k] for k in ('decision', 'source', 'note')}
        shadow.write_json(plans, applied['plans'])
        result = shadow.validate(plans, tmp_path / 'missing.json', humans)
        assert not result['valid'] and any('reapply' in error for error in result['errors'])
        changed = dict(human, fidelity_target='balanced')
        applied['plans']['regions'][0]['human'] = changed
        shadow.write_json(plans, applied['plans'])
        result = shadow.validate(plans, tmp_path / 'changed.json', humans)
        assert not result['valid'] and any('differs' in error for error in result['errors'])
