import threading

from vault_to_voice.index.manifest import Manifest


def test_one_manifest_is_safe_across_threads():
    """The API shares one Manifest between worker threads; without a lock sqlite3 raises
    InterfaceError / 'cannot start a transaction within a transaction' or loses writes."""
    manifest = Manifest(":memory:")
    errors: list[str] = []
    start = threading.Barrier(8)

    def work(i: int) -> None:
        try:
            start.wait()
            for j in range(60):
                manifest.set_meta(f"k{i}", str(j))
                ids = manifest.add_chunks(f"n{i}", ["a", "b", "c"])
                assert set(ids) <= set(manifest.chunk_ids(f"n{i}"))
                assert [c.chunk_id for c in manifest.get_chunks(ids)] == ids
                assert manifest.get_meta(f"k{i}") == str(j)
                manifest.list_ideas()
        except Exception as exc:  # collected, so the main thread can report it
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert manifest.chunk_count() == 8 * 60 * 3
