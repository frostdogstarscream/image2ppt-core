import json
from pathlib import Path

import pytest
from PIL import Image

from image2ppt_vision import region_plan


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


def response(job, decision):
    (job / "response.body").write_text(json.dumps({
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision, ensure_ascii=False)}}]
    }), encoding="utf8")
    (job / "transport.json").write_text(json.dumps({"status": "RECEIVED"}), encoding="utf8")


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
