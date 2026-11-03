"""Projects stay off every CV by default; one CV can tick a project in, reword it, or leave a bullet out,
without touching the profile."""
import json

from app import render, tailor
from app.config import output_dir
from app.profile import load_profile, save_profile

PROJECTS = [{"id": "tool", "name": "Tool", "role": "Open-source app", "url": "github.com/x/tool", "start": "2026-10",
             "bullets": [{"id": "t1", "text": "Built the app."}, {"id": "t2", "text": "Wrote the tests."}]},
            {"id": "site", "name": "Site", "role": "Side project", "bullets": [{"id": "s1", "text": "Made a site."}]}]
OPTIONS = {"market": "india", "variant": "flowcv", "max_pages": 2, "track": "ic_lead"}


def _setup():
    p = load_profile()
    p["projects"] = PROJECTS
    save_profile(p)
    folder = output_dir() / "a1"
    folder.mkdir(parents=True)
    plan = {"headline": "Lead Engineer", "summary": "Builds things.", "skills": [{"label": "Languages", "items": ["Python"]}],
            "experience": {"acme": ["Did work."]}}
    (folder / "plan.json").write_text(json.dumps(plan), "utf-8")
    (folder / "report.json").write_text(json.dumps({"options": OPTIONS, "cover_letter": ""}), "utf-8")
    return p, plan


def test_projects_stay_off_by_default_but_are_editable_sections():
    profile, plan = _setup()
    assert render.build_doc(profile, plan, OPTIONS)["projects"] == []
    got = tailor.get_plan("a1")["projects"]
    assert [(p["id"], p["role"], p["bullets"], p["hidden"]) for p in got] == [
        ("tool", "Open-source app", ["Built the app.", "Wrote the tests."], True), ("site", "Side project", ["Made a site."], True)]


def test_a_ticked_project_goes_on_that_cv_only(monkeypatch):
    profile, _ = _setup()
    monkeypatch.setattr(tailor, "rerender", lambda app_id, changes: {"id": app_id})
    tailor.save_plan("a1", {"projects": [{"id": "tool", "role": "Open-source app", "bullets": ["Built the app."], "hidden": False}]})
    plan = json.loads((output_dir() / "a1" / "plan.json").read_text("utf-8"))
    assert [p["name"] for p in render.build_doc(load_profile(), plan, OPTIONS)["projects"]] == ["Tool"]
    assert [p["hidden"] for p in tailor.get_plan("a1")["projects"]] == [False, True]


def test_a_cv_can_leave_out_or_reword_without_touching_the_profile(monkeypatch):
    profile, _ = _setup()
    monkeypatch.setattr(tailor, "rerender", lambda app_id, changes: {"id": app_id})
    tailor.save_plan("a1", {"projects": [{"id": "tool", "role": "Open-source app", "bullets": ["Built the app.", ""], "hidden": False},
                                         {"id": "site", "role": "Side project", "bullets": ["Made a site."], "hidden": True},
                                         {"id": "unknown", "bullets": ["Invented."]}]})
    plan = json.loads((output_dir() / "a1" / "plan.json").read_text("utf-8"))
    doc = render.build_doc(load_profile(), plan, OPTIONS)
    assert [(p["name"], p["bullets"]) for p in doc["projects"]] == [("Tool", ["Built the app."])]
    assert load_profile()["projects"] == PROJECTS                      # the profile keeps the original
    tailor.save_plan("a1", {"projects": [{"id": "tool", "hidden": True}, {"id": "site", "hidden": True}]})
    plan = json.loads((output_dir() / "a1" / "plan.json").read_text("utf-8"))
    assert render.build_doc(load_profile(), plan, OPTIONS)["projects"] == []   # the whole section left out
