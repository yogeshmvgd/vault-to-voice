import datetime as dt

import pytest

from vault_to_voice.vault.obsidian import ObsidianVaultReader

DAILY = """---
date: 10-10-2026
---
## Games
- Played [[Witcher 3]] — the Bloody Baron quest was brutal.
  - The choices felt heavy.
Combat still floaty.
- Tried [[Hades|Hades (2020)]] and it beats [[Dead Cells]] for me.
- Too tired to play much.
- [[Witcher 3]]
- Back to [[Games/Witcher 3#Combat|the Witcher]] later, the swords feel better now.

## Mood
- Happy about [[Witcher 3]] still.

## Books
* Read 40 pages of [[Dune]]. Paul's visions are getting creepy.
1. Also `[[Not A Link]]` and ![[cover.png]] with [[Dune]] again.

Free paragraph about [[Dune]] outside any bullet.
"""


def read(vault_dir, **kw):
    reader = ObsidianVaultReader(vault_dir, **kw)
    notes = {n.id: n for n in reader.iter_notes()}
    return notes, reader.stats


def test_daily_entries_split_by_bullet_and_category(vault_dir, write_note):
    write_note("Daily/10-10-2026.md", DAILY)
    notes, stats = read(vault_dir)
    assert set(notes) == {
        "witcher-3@2026-10-10",
        "witcher-3@2026-10-10-2",
        "hades@2026-10-10",
        "dune@2026-10-10",
        "dune@2026-10-10-2",
    }
    w = notes["witcher-3@2026-10-10"]
    assert w.body == (
        "Played Witcher 3 — the Bloody Baron quest was brutal.\n"
        "The choices felt heavy.\nCombat still floaty."
    )
    assert (w.type, w.date, w.work_id, w.title, w.path) == (
        "game",
        dt.date(2026, 10, 10),
        "witcher-3",
        "Witcher 3",
        "Daily/10-10-2026.md",
    )
    assert notes["witcher-3@2026-10-10-2"].body.startswith("Back to the Witcher later")
    h = notes["hades@2026-10-10"]
    assert h.links == ("hades", "dead-cells") and "[[" not in h.body
    assert notes["dune@2026-10-10"].type == "book"
    d2 = notes["dune@2026-10-10-2"]
    assert d2.links == ("dune",) and "cover.png" not in d2.body
    assert (stats.daily_files, stats.entries, stats.unlinked_skipped, stats.empty_skipped) == (
        1,
        5,
        1,
        1,
    )


def test_work_pages_resolve_titles_and_types(vault_dir, write_note):
    write_note("Daily/10-10-2026.md", "## Games\n- Loved [[witcher 3]] tonight.\n")
    write_note("Games/Witcher 3.md", "Overall the best side quests. See [[Hades]].\n")
    write_note("Misc/Thing.md", "---\ntype: book\n---\nA book I keep thinking about.\n")
    write_note("Notes/Random.md", "Just a thought.\n")
    notes, stats = read(vault_dir)
    assert notes["witcher-3@2026-10-10"].title == "Witcher 3"
    page = notes["witcher-3"]
    assert (page.type, page.work_id, page.title, page.links, page.date) == (
        "game",
        "witcher-3",
        "Witcher 3",
        ("hades",),
        None,
    )
    assert notes["thing"].type == "book" and notes["random"].type == "other"
    assert stats.pages == 3


def test_missing_page_uses_link_text(vault_dir, write_note):
    write_note("Daily/10-10-2026.md", "## Movies\n- Watched [[Neon Ferry]], silly fun.\n")
    notes, _ = read(vault_dir)
    n = notes["neon-ferry@2026-10-10"]
    assert (n.title, n.type) == ("Neon Ferry", "movie")


def test_empty_hub_page_skipped(vault_dir, write_note):
    write_note("Books/Dune.md", "---\ntype: book\n---\n# Dune\n\n## Notes\n")
    notes, stats = read(vault_dir)
    assert notes == {} and stats.empty_skipped == 1


def test_wrongly_named_daily_file_treated_as_page(vault_dir, write_note):
    write_note("Daily/Oct 10.md", "## Games\n- Played [[Hades]].\n")
    notes, _ = read(vault_dir)
    assert list(notes) == ["oct-10"] and notes["oct-10"].work_id == "oct-10"


def test_custom_date_format_and_categories(vault_dir, write_note):
    write_note("Journal/2026-10-10.md", "## Gaming\n- Played [[Hades]] all night.\n")
    notes, _ = read(
        vault_dir, daily_folder="Journal", date_format="%Y-%m-%d", categories={"gaming": "game"}
    )
    assert notes["hades@2026-10-10"].type == "game"


def test_tabs_crlf_and_blank_lines_inside_list(vault_dir):
    text = (
        "## Games\r\n- Played [[Hades]] again.\r\n\r\n\t- Still great.\r\n- Then [[Dune]] time.\r\n"
    )
    (vault_dir / "Daily").mkdir()
    (vault_dir / "Daily" / "10-10-2026.md").write_bytes(text.encode())
    notes, _ = read(vault_dir)
    assert notes["hades@2026-10-10"].body == "Played Hades again.\nStill great."
    assert "dune@2026-10-10" in notes


def test_editing_one_bullet_keeps_other_hashes(vault_dir, write_note):
    p = write_note(
        "Daily/10-10-2026.md", "## Games\n- Played [[Hades]].\n## Books\n- Read [[Dune]].\n"
    )
    before, _ = read(vault_dir)
    p.write_text(
        "## Games\n- Played [[Hades]].\n## Books\n- Read [[Dune]] twice.\n", encoding="utf-8"
    )
    after, _ = read(vault_dir)
    assert before["hades@2026-10-10"].hash == after["hades@2026-10-10"].hash
    assert before["dune@2026-10-10"].hash != after["dune@2026-10-10"].hash


def test_bad_yaml_and_hidden_dirs_skipped(vault_dir, write_note):
    write_note("Daily/10-10-2026.md", "---\ndate: [oops\n---\n## Games\n- [[Hades]] rocks.\n")
    write_note(".obsidian/x.md", "## Games\n- [[Hades]] hidden.\n")
    write_note("Daily/11-10-2026.md", "## Games\n- [[Hades]] rocks.\n")
    notes, _ = read(vault_dir)
    assert list(notes) == ["hades@2026-10-11"]


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(ObsidianVaultReader(tmp_path / "nope").iter_notes())
