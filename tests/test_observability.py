"""AI traces: a trace per job, a generation per model call, and what each run cost."""
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import engines, main, observability, profiles

LOCAL = TestClient(main.app, base_url="http://127.0.0.1:8765")


def test_a_trace_keeps_its_model_calls_tokens_and_cost():
    with observability.trace("tailor", session_id="job1", user_id="me", metadata={"market": "india"}) as t:
        with observability.generation("tailor", model="sonnet", engine="claude:sonnet", input="the brief") as g:
            g.set_output('{"cv": "written"}')
            g.add_usage(input=1_200, output=340, cache_read=8_000)
            g.set_cost(0.0421)
    trace = observability.get(t.id)
    assert trace["status"] == "ok"
    assert trace["session_id"] == "job1" and trace["user_id"] == "me"
    assert trace["metadata"] == {"market": "india"}
    assert trace["usage"] == {"input": 1_200, "output": 340, "total": 9_540, "cache_read": 8_000, "cache_creation": 0}
    assert trace["cost_usd"] == 0.0421
    assert len(trace["generations"]) == 1
    gen = trace["generations"][0]
    assert gen["model"] == "sonnet" and gen["engine"] == "claude:sonnet"
    assert gen["input"] == "the brief" and gen["output"] == '{"cv": "written"}'
    assert gen["latency_ms"] >= 0 and gen["level"] == "DEFAULT"


def test_a_failed_generation_is_recorded_as_an_error_and_still_raises():
    with pytest.raises(engines.EngineError, match="no model"):
        with observability.trace("review") as t:
            with observability.generation("review"):
                raise engines.EngineError("no model")
    trace = observability.get(t.id)
    assert trace["status"] == "error" and "no model" in trace["status_message"]
    assert trace["generations"][0]["level"] == "ERROR"
    assert "EngineError: no model" in trace["generations"][0]["status_message"]


def test_a_model_call_with_no_trace_opens_one_named_after_the_caller():
    with observability.generation("discover", model="sonnet", input="search") as g:
        g.set_output("[]")
    trace = observability.recent(1)[0]
    assert trace["name"] == "discover"
    assert trace["metadata"] == {"implicit": True}
    assert trace["status"] == "ok" and len(trace["generations"]) == 1


def test_long_briefs_and_replies_are_cut_to_head_and_tail():
    brief = "x" * 50_000 + "TAIL"
    with observability.generation("tailor", input=brief) as g:
        g.set_output("y" * 50_000)
    gen = observability.recent(1)[0]["generations"][0]
    assert len(gen["input"]) <= observability.MAX_FIELD + 100
    assert "50004 chars" in gen["input"]
    assert len(gen["output"]) <= observability.MAX_FIELD + 100


def test_summary_adds_up_runs_tokens_cost_and_latency():
    with observability.trace("scan"):
        for n in range(3):
            with observability.generation("scan", model="sonnet") as g:
                g.add_usage(input=100, output=10)
                g.set_cost(0.01)
    with observability.trace("facts"):
        with observability.generation("facts", model="sonnet") as g:
            g.add_usage(input=50, output=5)
            g.set_cost(0.002)
    s = observability.summary(days=1)
    assert s["traces"] == 2 and s["generations"] == 4
    assert s["tokens"]["input"] == 350 and s["tokens"]["output"] == 35
    assert s["cost_usd"] == pytest.approx(0.032)
    assert s["by_model"]["sonnet"]["runs"] == 4
    assert s["by_name"]["scan"]["runs"] == 3 and s["by_name"]["facts"]["runs"] == 1
    assert s["latency_ms"]["p50"] is not None


def test_cost_is_estimated_from_list_price_only_when_the_cli_reports_none():
    with observability.generation("tailor", model="sonnet") as g:
        g.add_usage(input=1_000_000, output=1_000_000)
    gen = observability.recent(1)[0]["generations"][0]
    assert gen["cost_usd"] == pytest.approx(18.0)
    assert gen["cost_source"] == "estimate"
    with observability.generation("tailor", model="sonnet") as g:
        g.add_usage(input=1_000_000, output=1_000_000)
        g.set_cost(0.25)
    assert observability.recent(1)[0]["generations"][0]["cost_usd"] == 0.25


def test_a_score_lands_on_its_trace():
    with observability.trace("tailor") as t:
        observability.score("fit", 78, comment="strong on Python")
    scores = observability.get(t.id)["scores"]
    assert [(s["name"], s["value"]) for s in scores] == [("fit", 78)]


def test_rotated_files_are_still_read(monkeypatch):
    monkeypatch.setattr(observability, "MAX_BYTES", 2_000)
    ids = []
    for n in range(10):
        with observability.trace(f"job{n}") as t:
            with observability.generation(f"job{n}", model="sonnet") as g:
                g.set_output("z" * 200)
            ids.append(t.id)
    rotated = observability.trace_path().with_name(observability.trace_path().name + ".1")
    assert rotated.exists()
    found = {t["id"]: t for t in observability.read()}
    assert ids[-1] in found, "the newest trace survived the rotation"
    assert found[ids[-1]]["generations"][0]["trace_id"] == ids[-1]
    assert set(found) <= set(ids)
    assert len(found) < len(ids), "old traces rotate away on purpose"


def test_recording_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("JOBMAGNET_OBSERVABILITY", "0")
    with observability.trace("tailor"):
        with observability.generation("tailor"):
            pass
    assert not observability.trace_path().exists()


def test_each_profile_keeps_its_own_traces():
    row = profiles.create("Temp")
    token = profiles.activate(row["id"])
    try:
        with observability.trace("tailor"):
            pass
        assert observability.trace_path().parent == profiles.data_dir(row["id"])
    finally:
        profiles.reset(token)
    other = profiles.activate("me")
    try:
        with observability.trace("discover"):
            pass
        assert [t["name"] for t in observability.recent(5)] == ["discover"]
    finally:
        profiles.reset(other)


def test_a_claude_run_records_the_envelope_usage_and_cost(monkeypatch):
    envelope = {"result": '{"ok": true}', "usage": {"input_tokens": 1_500, "output_tokens": 300,
                                                   "cache_read_input_tokens": 20_000, "cache_creation_input_tokens": 100},
                "total_cost_usd": 0.0731, "duration_ms": 1_200, "num_turns": 1, "session_id": "sess-1"}
    monkeypatch.setattr(engines.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(engines, "_run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(envelope), stderr="", returncode=0))
    assert engines.ClaudeCLI("sonnet").run("brief") == '{"ok": true}'
    gen = observability.recent(1)[0]["generations"][0]
    assert gen["model"] == "sonnet" and gen["engine"] == "claude:sonnet"
    assert gen["usage"] == {"input": 1_500, "output": 300, "total": 21_900, "cache_read": 20_000, "cache_creation": 100}
    assert gen["cost_usd"] == 0.0731 and gen["cost_source"] == "cli"
    assert gen["metadata"]["turns"] == 1 and gen["metadata"]["cli_session"] == "sess-1"
    assert gen["input"] == "brief" and gen["output"] == '{"ok": true}'


def test_a_failed_claude_run_says_so_in_its_trace(monkeypatch):
    monkeypatch.setattr(engines.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(engines, "_run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({"is_error": True, "result": "Not logged in"}), stderr="", returncode=1))
    with pytest.raises(engines.EngineError):
        engines.ClaudeCLI("sonnet").run("brief")
    trace = observability.recent(1)[0]
    assert trace["status"] == "error"
    assert trace["generations"][0]["level"] == "ERROR"


def test_a_job_is_one_trace_with_the_job_id_as_its_session():
    def work(progress):
        with observability.generation("facts") as g:
            g.set_output("ok")
        return "done"

    jid = main.start_job("facts", work)
    for _ in range(200):
        if main.JOBS[jid]["status"] != "running":
            break
        time.sleep(0.01)
    assert main.JOBS[jid]["status"] == "done", main.JOBS[jid]
    trace = observability.get(main.JOBS[jid]["ai_trace"])
    assert trace["session_id"] == jid and trace["name"] == "facts"
    assert len(trace["generations"]) == 1


def test_a_failed_job_marks_its_trace_failed():
    def boom(progress):
        with observability.generation("facts"):
            raise RuntimeError("no model")
    jid = main.start_job("facts", boom)
    for _ in range(200):
        if main.JOBS[jid]["status"] != "running":
            break
        time.sleep(0.01)
    trace = observability.get(main.JOBS[jid]["ai_trace"])
    assert trace["status"] == "error" and "no model" in trace["status_message"]


def test_the_observability_endpoints_read_this_profiles_traces():
    with observability.trace("tailor", session_id="s1") as t:
        with observability.generation("tailor", model="sonnet") as g:
            g.add_usage(input=10, output=5)
    body = LOCAL.get("/api/observability?days=1").json()
    assert body["path"].endswith("ai_traces.jsonl")
    assert body["summary"]["generations"] >= 1
    assert any(x["id"] == t.id for x in body["traces"])
    one = LOCAL.get(f"/api/observability/{t.id}").json()
    assert one["name"] == "tailor" and len(one["generations"]) == 1
    assert LOCAL.get("/api/observability/nope").status_code == 404
