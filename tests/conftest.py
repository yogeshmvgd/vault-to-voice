from pathlib import Path

import pytest


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
