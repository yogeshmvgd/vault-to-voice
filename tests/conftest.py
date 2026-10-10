from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_langsmith(monkeypatch):
    """Tracing is opt-in: tests never send traces, even from a shell that enables them."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_real_provider_keys(monkeypatch):
    """A real key in .env must never reach tests; the fake value also wins over .env."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")


@pytest.fixture
def vault_dir(tmp_path) -> Path:
    d = tmp_path / "vault"
    d.mkdir()
    return d


@pytest.fixture
def write_note(vault_dir):
    def _write(rel_path: str, text: str) -> Path:
        p = vault_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    return _write


def note_text(id_: str, body: str, type_: str = "game", title: str = "T", date: str = "2026-01-01"):
    return f"---\nid: {id_}\ndate: {date}\ntype: {type_}\ntitle: {title}\n---\n{body}\n"
