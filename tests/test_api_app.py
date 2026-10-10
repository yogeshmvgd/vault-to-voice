import pytest
from fastapi.testclient import TestClient

from tests.fakes import make_service
from vault_to_voice.api.app import create_app
from vault_to_voice.api.events import parse_sse

ANSWERS = ["The tells.", "Twenty deaths.", "Patience."]
APPROVE = {"action": "approve"}


@pytest.fixture
def api(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    return TestClient(create_app(service)), idea.idea_id


def events(client, method, url, **kwargs):
    with client.stream(method, url, **kwargs) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return list(parse_sse(response.iter_lines()))


def resume(client, tid, payload):
    return events(client, "POST", f"/threads/{tid}/resume", json={"payload": payload})


def test_health_and_ideas(api):
    client, tid = api
    assert client.get("/health").json() == {"status": "ok", "tracing": False}
    ideas = client.get("/ideas").json()
    assert [i["idea_id"] for i in ideas] == [tid] and ideas[0]["status"] == "new"
    assert client.get("/threads").json() == []
    assert client.get(f"/threads/{tid}").json()["status"] == "new"


def test_full_flow_over_http(api):
    client, tid = api
    first = events(client, "POST", "/threads", json={"idea_id": tid})
    assert first[0] == ("node", {"node": "load_context"})
    assert first[-1][0] == "interrupt" and first[-1][1]["type"] == "interview_question"
    for answer in ANSWERS:
        last = resume(client, tid, answer)[-1]
    assert last[1]["type"] == "outline_review"
    view = client.get(f"/threads/{tid}").json()
    assert view["status"] == "waiting" and view["pending"]["type"] == "outline_review"
    assert len(view["qa"]) == 3
    last = resume(client, tid, APPROVE)[-1]
    assert last[1]["type"] == "draft_review" and last[1]["coverage"] == 1.0
    done = resume(client, tid, APPROVE)
    assert ("node", {"node": "publish"}) in done and done[-1][0] == "done"
    post = client.get(f"/threads/{tid}/post").json()
    assert post["final_path"] == done[-1][1]["final_path"]
    assert "# Fair Bosses" in post["markdown"] and post["sources"]["paragraphs"]
    assert [t["status"] for t in client.get("/threads").json()] == ["published"]
    assert client.get("/ideas").json()[0]["status"] == "done"


def test_errors_map_to_http_status_codes(api):
    client, tid = api
    r = client.post("/threads", json={"idea_id": "idea_nope"})
    assert r.status_code == 404 and "unknown idea" in r.json()["detail"]
    assert client.get("/threads/idea_nope").status_code == 404
    assert client.post(f"/threads/{tid}/resume", json={"payload": "x"}).status_code == 409
    assert client.get(f"/threads/{tid}/stream").status_code == 409
    assert client.get(f"/threads/{tid}/post").status_code == 404
    events(client, "POST", "/threads", json={"idea_id": tid})
    assert client.post("/threads", json={"idea_id": tid}).status_code == 409


@pytest.mark.parametrize("body", [{}, {"payload": None}, {"payload": 5}, {"payload": [1]}])
def test_invalid_resume_body_is_rejected(api, body):
    client, tid = api
    events(client, "POST", "/threads", json={"idea_id": tid})
    assert client.post(f"/threads/{tid}/resume", json=body).status_code == 422
    assert client.get(f"/threads/{tid}").json()["pending"]["number"] == 1  # nothing consumed
