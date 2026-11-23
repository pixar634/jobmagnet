"""The adaptive coding test: written a level at a time, level 2 first, so the page can start while the rest is
written. A long reply is never cut off before its last brace, and each level is saved as it lands."""
import json
import re

import pytest
from fastapi.testclient import TestClient

from app import interview, main
from app.config import output_dir
from app.main import app

LOCAL = TestClient(app, base_url="http://127.0.0.1:8765")


def _problem(level: int, title: str) -> dict:
    return {
        "title": title, "level": level, "topic": "arrays", "statement": "Return the total.",
        "examples": [{"input": "nums = [1, 2]", "output": "3", "explain": ""}],
        "constraints": ["1 <= len(nums) <= 10"],
        "function": {"python": "total", "javascript": "total"},
        "params": ["nums"],
        "starter": {"python": "def total(nums):\n    pass\n", "javascript": "function total(nums) {\n}\n"},
        "tests": [{"args": [[1, 2]], "expected": 3}, {"args": [[]], "expected": 0}, {"args": [[-1, 1]], "expected": 0}],
        "compare": "exact",
        "solution": {"python": "def total(nums):\n    return sum(nums)\n",
                     "javascript": "function total(nums) { return nums.reduce((a, b) => a + b, 0); }\n"},
        "hints": ["add them"], "complexity": "O(n)",
    }


@pytest.fixture
def app_id(sandbox):
    folder = output_dir() / "2026-10-30_Acme_Lead"
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps({"company": "Acme", "role": "Lead Engineer"}), "utf-8")
    (folder / "plan.json").write_text("{}", "utf-8")
    (folder / "jd.txt").write_text("Build APIs in Python.", "utf-8")
    return folder.name


def _level(brief: str) -> int:
    return int(re.search(r"level (\d) only", brief).group(1))


def test_the_coding_test_is_written_one_level_at_a_time(app_id, monkeypatch):
    runs = []

    class Writer:
        def run(self, brief, timeout=0, tools=None, name=""):
            level = _level(brief)
            runs.append(level)
            return json.dumps({"problems": [_problem(level, f"L{level} A"), _problem(level, f"L{level} B")]})

    monkeypatch.setattr(interview, "get_engine", lambda e: Writer())
    out = interview.generate_codetest(app_id, "claude:opus")
    assert runs == [2, 3, 1, 4, 5]      # five small replies, level 2 first: the level the test opens on
    assert [p["level"] for p in out["problems"]] == [2, 2, 3, 3, 1, 1, 4, 4, 5, 5]
    assert [p["id"] for p in out["problems"]] == [f"p{i}" for i in range(1, 11)]
    assert interview.load_codetest(app_id)["problems"] == out["problems"]


def test_level_2_is_saved_before_the_rest_is_written(app_id, monkeypatch):
    """The page starts the test as soon as level 2 is on disk, and the other levels fill in behind it."""
    seen = []

    class Writer:
        def run(self, brief, timeout=0, tools=None, name=""):
            level = _level(brief)
            saved = interview.load_codetest(app_id) or {"problems": []}
            seen.append((level, sorted({p["level"] for p in saved["problems"]})))
            return json.dumps({"problems": [_problem(level, f"L{level} A"), _problem(level, f"L{level} B")]})

    monkeypatch.setattr(interview, "get_engine", lambda e: Writer())
    out = interview.generate_codetest(app_id, "claude:opus")
    assert seen == [(2, []), (3, [2]), (1, [2, 3]), (4, [1, 2, 3]), (5, [1, 2, 3, 4])]
    assert out["missing_levels"] == []


def test_a_level_that_never_arrives_keeps_the_levels_already_written(app_id, monkeypatch):
    """A partial test starts on the nearest level written instead of throwing the whole run away."""
    class Writer:
        def run(self, brief, timeout=0, tools=None, name=""):
            level = _level(brief)
            if level == 3:
                return json.dumps({"problems": []})
            return json.dumps({"problems": [_problem(level, f"L{level} A"), _problem(level, f"L{level} B")]})

    monkeypatch.setattr(interview, "get_engine", lambda e: Writer())
    out = interview.generate_codetest(app_id, "claude:opus")
    assert sorted({p["level"] for p in out["problems"]}) == [1, 2, 4, 5]
    assert out["missing_levels"] == [3]
    assert interview.load_codetest(app_id)["missing_levels"] == [3]


def test_asking_again_for_a_missing_level_clears_it(app_id, monkeypatch):
    """more_problems fills the gap and the page stops offering to retry that level."""
    state = {"fail": True}

    class Writer:
        def run(self, brief, timeout=0, tools=None, name=""):
            level = _level(brief)
            if level == 3 and state["fail"]:
                return json.dumps({"problems": []})
            return json.dumps({"problems": [_problem(level, f"L{level} A"), _problem(level, f"L{level} B")]})

    monkeypatch.setattr(interview, "get_engine", lambda e: Writer())
    out = interview.generate_codetest(app_id, "claude:opus")
    assert out["missing_levels"] == [3]
    state["fail"] = False
    data = interview.more_problems(app_id, 3, "claude:opus")
    assert data["missing_levels"] == []
    assert [p["level"] for p in data["problems"]].count(3) == 2     # the two the writer just added
    assert len({p["id"] for p in data["problems"]}) == len(data["problems"])


def test_a_partial_test_is_served_while_the_writer_still_runs(app_id, monkeypatch):
    """The page starts on level 2 with the writer going, and asking for another level joins that job, not a second."""
    url = f"/api/apps/{app_id}/codetest"
    monkeypatch.setattr(main, "JOBS", {})
    monkeypatch.setattr(main, "CODE_JOBS", {})
    interview._save_codetest(app_id, {"generated_at": "2026-11-24T00:00:00", "engine": "claude:opus",
                                      "language": "python", "problems": [interview._problem(_problem(2, "L2 A"))],
                                      "missing_levels": []})
    main.JOBS["live"] = {"id": "live", "kind": "codetest", "status": "running", "log": [], "result": None,
                         "error": None}
    main.CODE_JOBS[main._key(app_id)] = "live"

    got = LOCAL.get(url).json()
    assert [p["level"] for p in got["test"]["problems"]] == [2]
    assert got["job"] == "live"
    assert LOCAL.post(url, json={"level": 4}).json() == {"job": "live"}   # joins the writer
    assert list(main.JOBS) == ["live"]                                     # no second job started


def test_the_whole_test_is_saved_and_read_back_when_every_level_arrives(app_id, monkeypatch):
    class Writer:
        def run(self, brief, timeout=0, tools=None, name=""):
            level = _level(brief)
            return json.dumps({"problems": [_problem(level, f"L{level} A"), _problem(level, f"L{level} B")]})

    monkeypatch.setattr(interview, "get_engine", lambda e: Writer())
    interview.generate_codetest(app_id, "claude:opus")
    got = LOCAL.get(f"/api/apps/{app_id}/codetest").json()
    assert got["job"] is None
    assert len(got["test"]["problems"]) == 10
    assert got["test"]["missing_levels"] == []


def test_a_cut_off_reply_is_asked_for_again(app_id, monkeypatch):
    calls = {"n": 0}

    class Writer:
        def run(self, brief, timeout=0, tools=None, name=""):
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"problems": [{"title": "this reply stops before the last brace"'
            level = _level(brief)
            return json.dumps({"problems": [_problem(level, f"L{level} A"), _problem(level, f"L{level} B")]})

    monkeypatch.setattr(interview, "get_engine", lambda e: Writer())
    out = interview.generate_codetest(app_id, "claude:opus")
    assert len(out["problems"]) == 10
    assert calls["n"] == 6             # the cut-off batch was asked for once more


def test_a_level_that_never_arrives_leaves_an_error_not_an_empty_file(app_id, monkeypatch):
    class Writer:
        def run(self, brief, timeout=0, tools=None, name=""):
            return json.dumps({"problems": []})

    monkeypatch.setattr(interview, "get_engine", lambda e: Writer())
    with pytest.raises(RuntimeError, match="no usable level 3 problems"):
        interview.more_problems(app_id, 3, "claude:opus")
    assert interview.load_codetest(app_id) is None
