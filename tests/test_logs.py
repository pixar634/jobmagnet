"""The app's log: one file per profile, tagged with the profile and job, and the endpoint that reads its tail."""
import logging
import time

from fastapi.testclient import TestClient

from app import logs, main, profiles

LOCAL = TestClient(main.app, base_url="http://127.0.0.1:8765")


def _flush():
    for handler in logging.getLogger(logs.NAME).handlers:
        handler.flush()


def test_setup_attaches_handlers_only_once():
    logs.setup()
    logs.setup()
    names = [h.name for h in logging.getLogger(logs.NAME).handlers]
    assert names.count("jobmagnet-console") == 1
    assert names.count("jobmagnet-file") == 1


def test_records_land_in_the_profiles_file_with_the_profile_and_job():
    logs.setup()
    profile = profiles.activate("me")
    job = logs.bind_job("abc123")
    try:
        logs.get_logger("test").info("tailored the CV")
    finally:
        logs.reset_job(job)
        profiles.reset(profile)
    _flush()
    text = logs.log_path().read_text("utf-8")
    assert "tailored the CV" in text
    assert "[me/abc123]" in text


def test_tail_returns_the_latest_lines():
    logs.setup()
    log = logs.get_logger("test")
    for n in range(5):
        log.info("line %s", n)
    _flush()
    recent = logs.tail(2)
    assert len(recent) == 2
    assert recent[-1].endswith("line 4")


def test_log_endpoint_reads_this_profiles_file():
    logs.setup()
    logs.get_logger("test").warning("something worth reporting")
    _flush()
    body = LOCAL.get("/api/logs?lines=20").json()
    assert body["path"].endswith("jobmagnet.log")
    assert any("something worth reporting" in line for line in body["lines"])


def test_deleting_a_profile_releases_its_log_file():
    logs.setup()
    row = profiles.create("Temp")
    profile = profiles.activate(row["id"])
    try:
        logs.get_logger("test").info("written before the profile goes")
    finally:
        profiles.reset(profile)
    profiles.delete(row["id"])
    assert not (profiles.PROFILES / row["id"]).exists()


def test_a_failed_job_is_logged_with_its_traceback_and_job_id():
    logs.setup()
    jid = main.start_job("boom", lambda progress: (_ for _ in ()).throw(RuntimeError("no model")))
    for _ in range(200):
        _flush()
        text = logs.log_path().read_text("utf-8") if logs.log_path().exists() else ""
        if f"job {jid} (boom) failed" in text and "RuntimeError: no model" in text:
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"job {jid} was not logged")
    assert f"/{jid}]" in text
    assert main.JOBS[jid]["status"] == "error"
