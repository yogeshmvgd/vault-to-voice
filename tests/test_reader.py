import datetime as dt

import pytest

from tests.conftest import note_text
from vault_to_voice.vault.reader import FilesystemVaultReader, slugify


def read(vault_dir):
    return {n.id: n for n in FilesystemVaultReader(vault_dir).iter_notes()}


def test_parses_frontmatter(vault_dir, write_note):
    write_note("games/a.md", note_text("a1", "Loved it.", title="Alpha"))
    n = read(vault_dir)["a1"]
    assert (n.title, n.type, n.date, n.body, n.path) == (
        "Alpha",
        "game",
        dt.date(2026, 1, 1),
        "Loved it.",
        "games/a.md",
    )


def test_id_falls_back_to_path_slug(vault_dir, write_note):
    write_note("games/My Note.md", "---\ntitle: X\n---\nBody text.\n")
    assert "games-my-note" in read(vault_dir)


def test_slugify():
    assert slugify("Games/My Note!") == "games-my-note"


def test_skips_hidden_dirs(vault_dir, write_note):
    write_note(".obsidian/workspace.md", note_text("h", "hidden"))
    write_note("ok.md", note_text("ok", "visible"))
    assert set(read(vault_dir)) == {"ok"}


def test_bad_yaml_skipped_others_read(vault_dir, write_note):
    write_note("bad.md", "---\ntitle: [unclosed\n---\nBody\n")
    write_note("good.md", note_text("good", "fine"))
    assert set(read(vault_dir)) == {"good"}


def test_defaults_for_missing_or_invalid_fields(vault_dir, write_note):
    write_note("x.md", "---\ntype: podcast\ndate: not-a-date\n---\nSome body.\n")
    n = read(vault_dir)["x"]
    assert (n.type, n.date, n.title) == ("other", None, "x")


def test_empty_body_skipped(vault_dir, write_note):
    write_note("empty.md", note_text("e", "   "))
    assert read(vault_dir) == {}


def test_duplicate_id_disambiguated(vault_dir, write_note):
    write_note("a/one.md", note_text("dup", "first"))
    write_note("b/two.md", note_text("dup", "second"))
    notes = read(vault_dir)
    assert set(notes) == {"dup", "b-two"}


def test_non_utf8_does_not_crash(vault_dir):
    (vault_dir / "latin.md").write_bytes(b"---\nid: l\n---\ncaf\xe9 notes\n")
    assert "l" in read(vault_dir)


def test_hash_changes_with_content(vault_dir, write_note):
    p = write_note("a.md", note_text("a", "v1"))
    h1 = read(vault_dir)["a"].hash
    p.write_text(note_text("a", "v2"), encoding="utf-8")
    assert read(vault_dir)["a"].hash != h1


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(FilesystemVaultReader(tmp_path / "nope").iter_notes())
