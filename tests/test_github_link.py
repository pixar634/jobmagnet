"""The profile's GitHub link prints in every CV's header, next to LinkedIn, and is cleaned when edited."""
from fastapi.testclient import TestClient

from app import render
from app.main import app
from app.profile import load_profile

PLAN = {"headline": "Lead Engineer", "summary": "Builds things.", "skills": [{"label": "Languages", "items": ["Python"]}],
        "experience": {}}
OPTIONS = {"market": "india", "variant": "flowcv", "max_pages": 2, "track": "ic_lead"}


def test_github_link_is_cleaned_and_printed_in_the_header():
    TestClient(app, base_url="http://127.0.0.1:8765").post("/api/profile/basics", json={"github": " https://www.github.com/maya-rao/ "})
    profile = load_profile()
    assert profile["basics"]["github"] == "github.com/maya-rao"
    doc = render.build_doc(profile, PLAN, OPTIONS)
    assert doc["github"] == "github.com/maya-rao"
    assert '<a href="https://github.com/maya-rao">github.com/maya-rao</a>' in render.render_html(doc)


def test_no_github_means_no_header_entry():
    doc = render.build_doc(load_profile(), PLAN, OPTIONS)
    assert doc["github"] == ""
