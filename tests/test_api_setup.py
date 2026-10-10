import pytest
from fastapi.testclient import TestClient

from tests.fakes import make_service
from vault_to_voice import setup_actions
from vault_to_voice.api.app import create_app


@pytest.fixture
def api(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    return TestClient(create_app(service)), service, idea.idea_id


def test_config_shows_settings_without_secrets(api, monkeypatch):
    client, service, _ = api
    service.deps.settings.openai_api_key = "sk-secret-123"
    service.deps.settings.langsmith_api_key = "ls-secret-456"
    body = client.get("/config")
    assert body.status_code == 200
    data = body.json()
    assert data["vault_format"] == "flat" and data["embedding_provider"] == "hash"
    assert {"vault_path", "llm_model", "data_dir"} <= data.keys()
    assert "secret" not in body.text and not any("key" in k for k in data)


def test_ingest_style_and_detect_over_http(api):
    client, service, _ = api
    r = client.post("/ingest", json={})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["unchanged"] == 5 and data["new"] == 0 and data["stats"] is None
    assert client.post("/ingest", json={"no_score": True}).json()["scored"] == 0
    style = client.post("/style").json()
    assert style["version"] == 1 and style["tone"] == ["wry"] and style["voice_notes"]
    assert service.deps.settings.style_profile_path.exists()
    ideas = client.post("/ideas/detect")
    assert ideas.status_code == 200 and isinstance(ideas.json(), list)
    assert [i["idea_id"] for i in client.get("/ideas").json()] == [
        i["idea_id"] for i in ideas.json()
    ]


def test_ingest_without_body_defaults_to_scoring(api):
    client, *_ = api
    assert client.post("/ingest").status_code == 200


def test_detect_out_of_sync_is_409(api):
    client, service, _ = api
    service.deps.manifest.set_meta("dirty", "1")
    r = client.post("/ideas/detect")
    assert r.status_code == 409 and "ingest" in r.json()["detail"]


def test_setup_refused_while_a_post_runs(api):
    client, service, tid = api
    stream = service.start(tid)  # holds the thread's run lock until closed
    try:
        for url in ("/ingest", "/style", "/ideas/detect"):
            r = client.post(url)
            assert r.status_code == 409 and "post" in r.json()["detail"]
    finally:
        stream.close()
    assert client.post("/style").status_code == 200


def test_setup_actions_not_concurrent_and_block_posts(api):
    client, service, tid = api
    service._maintenance.acquire()
    try:
        r = client.post("/ingest")
        assert r.status_code == 409 and "setup" in r.json()["detail"]
        assert client.post("/threads", json={"idea_id": tid}).status_code == 409
    finally:
        service._maintenance.release()
    assert client.post("/ingest").status_code == 200


def test_unexpected_failure_is_clean_500(api, monkeypatch):
    client, *_ = api

    def boom(*_a, **_k):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(setup_actions, "build_style", boom)
    r = client.post("/style")
    assert r.status_code == 500
    assert r.json() == {"detail": "build style profile failed: model unreachable"}


def test_style_with_empty_vault_is_400(api):
    client, service, _ = api
    for f in service.deps.settings.vault_path.glob("*.md"):
        f.unlink()
    r = client.post("/style")
    assert r.status_code == 400 and r.json()["detail"] == "no notes found"
