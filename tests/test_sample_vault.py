import datetime as dt
from collections import Counter

import frontmatter

from vault_to_voice.config import SAMPLE_VAULT
from vault_to_voice.vault.obsidian import ObsidianVaultReader


def test_every_file_is_fictional():
    files = sorted(SAMPLE_VAULT.rglob("*.md"))
    assert files
    for path in files:
        assert frontmatter.load(path).get("fictional") is True, path


def test_layout():
    daily = sorted((SAMPLE_VAULT / "Daily").glob("*.md"))
    assert len(daily) >= 20
    for p in daily:
        dt.datetime.strptime(p.stem, "%d-%m-%Y")
    pages = [
        p for folder in ("Games", "Books", "Movies") for p in (SAMPLE_VAULT / folder).glob("*.md")
    ]
    assert len(pages) >= 12


def test_reader_shape():
    reader = ObsidianVaultReader(SAMPLE_VAULT)
    notes = list(reader.iter_notes())
    entries = [n for n in notes if "@" in n.id]
    assert len(entries) >= 40
    assert reader.stats.unlinked_skipped >= 1 and reader.stats.empty_skipped >= 1
    assert any(len(n.links) > 1 for n in entries)
    per_work = Counter(n.work_id for n in notes)
    assert sum(1 for c in per_work.values() if c >= 4) >= 4  # one rich work per theme
    assert {n.type for n in entries} == {"game", "book", "movie"}
