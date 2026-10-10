import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fakes import make_service
from vault_to_voice.api.app import create_app
from vault_to_voice.ui.client import ApiClient, ApiError

APPROVE = {"action": "approve"}


@pytest.fixture
def client(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    return ApiClient(TestClient(create_app(service))), idea.idea_id


def test_client_drives_a_post_to_publication(client):
    api, tid = client
    assert api.health()["status"] == "ok"
    assert [i["idea_id"] for i in api.ideas()] == [tid]
    events = list(api.start(tid))
    assert events[0] == ("node", {"node": "load_context"}) and events[-1][0] == "interrupt"
    for answer in ["a", "b", "c"]:
        last = list(api.resume(tid, answer))[-1]
    assert last[1]["type"] == "outline_review"
    assert list(api.resume(tid, APPROVE))[-1][1]["type"] == "draft_review"
    assert list(api.resume(tid, APPROVE))[-1][0] == "done"
    assert api.thread(tid)["status"] == "published"
    assert api.threads()[0]["thread_id"] == tid
    assert api.post(tid)["sources"]["paragraphs"]


def test_http_errors_raise_api_error_with_detail(client):
    api, tid = client
    with pytest.raises(ApiError) as exc:
        list(api.start("idea_nope"))
    assert exc.value.status_code == 404 and "unknown idea" in exc.value.detail
    with pytest.raises(ApiError) as exc:
        api.post(tid)
    assert exc.value.status_code == 404
    with pytest.raises(ApiError) as exc:
        list(api.retry(tid))
    assert exc.value.status_code == 409


def test_from_url_builds_an_httpx_client():
    api = ApiClient.from_url("http://127.0.0.1:9", timeout=5)
    assert isinstance(api.http, httpx.Client) and str(api.http.base_url) == "http://127.0.0.1:9"
    with pytest.raises(httpx.ConnectError):
        api.health()  # nothing listens on port 9: the UI must show this, not crash
