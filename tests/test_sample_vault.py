import frontmatter

from vault_to_voice.config import SAMPLE_VAULT


def _notes():
    return sorted(SAMPLE_VAULT.rglob("*.md"))


def test_sample_vault_size():
    assert len(_notes()) >= 28


def test_every_note_is_fictional_and_complete():
    for path in _notes():
        post = frontmatter.load(path)
        assert post.get("fictional") is True, path
        assert post.get("type") in {"game", "book", "movie"}, path
        for key in ("id", "date", "title"):
            assert key in post.metadata, (path, key)
        assert len(post.content.split()) >= 60, path


def test_ids_unique():
    ids = [frontmatter.load(p)["id"] for p in _notes()]
    assert len(ids) == len(set(ids))
