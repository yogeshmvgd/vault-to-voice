import os

import pytest

from vault_to_voice.config import Settings
from vault_to_voice.graph.runner import thread_config
from vault_to_voice.observability import configure_tracing


@pytest.fixture
def clean_env(monkeypatch):
    # monkeypatch records these, so whatever configure_tracing writes is undone after the test
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(name, raising=False)


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_tracing_is_off_by_default(clean_env):
    assert make().langsmith_tracing is False
    assert configure_tracing(make()) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_needs_an_api_key(clean_env):
    assert configure_tracing(make(langsmith_tracing=True)) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_on_sets_langsmith_env(clean_env):
    s = make(langsmith_tracing=True, langsmith_api_key="lsv2-test", langsmith_project="p1")
    assert configure_tracing(s) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2-test"
    assert os.environ["LANGSMITH_PROJECT"] == "p1"


def test_tracing_reads_env_vars(clean_env, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    s = make()
    assert s.langsmith_tracing is True and s.langsmith_api_key == "k"
    assert s.langsmith_project == "vault-to-voice"


def test_tests_never_trace():
    # the autouse conftest fixture forces tracing off for every test
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_thread_config_labels_traces():
    cfg = thread_config("idea_x")
    assert cfg["configurable"] == {"thread_id": "idea_x"}
    assert cfg["run_name"] == "post_graph" and "post-graph" in cfg["tags"]
    assert cfg["metadata"] == {"idea_id": "idea_x"}
