from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.fakes import FakeLLMClient, graph_responses, seed_idea, sse_events
from vault_to_voice import cli
from vault_to_voice.api.factory import build_service
from vault_to_voice.config import Settings


def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        outbox_dir=tmp_path / "outbox",
        embedding_provider="hash",
    )


def test_restarted_server_resumes_the_same_thread(tmp_path, vault_dir):
    s = settings(tmp_path)
    first = build_service(s, FakeLLMClient(graph_responses()))
    idea = seed_idea(first.deps, vault_dir)
    sse_events(first.start(idea.idea_id))
    sse_events(first.resume(idea.idea_id, "The tells."))

    fresh_llm = FakeLLMClient(graph_responses())
    second = build_service(s, fresh_llm)  # a new process: same SQLite files, nothing in memory
    view = second.thread(idea.idea_id)
    assert view.status == "waiting" and view.pending["number"] == 2
    assert [q["answer"] for q in view.qa] == ["The tells."]
    assert fresh_llm.calls == []  # reading state never re-asks the model
    assert sse_events(second.resume(idea.idea_id, "Twenty deaths."))[-1][1]["number"] == 3


def test_serve_builds_the_app_and_runs_uvicorn(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setattr(cli, "make_llm", lambda settings: FakeLLMClient(graph_responses()))
    seen = {}

    def fake_run(app, host, port):
        seen.update(app=app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = CliRunner().invoke(cli.app, ["serve", "--port", "9001"])
    assert result.exit_code == 0, result.output
    assert seen["host"] == "127.0.0.1" and seen["port"] == 9001
    assert TestClient(seen["app"]).get("/health").json() == {"status": "ok", "tracing": False}
