import numpy as np

from tests.fakes import FakeLLMClient
from vault_to_voice.ideas.detector import Idea, detect_ideas, make_idea_id
from vault_to_voice.ideas.labeler import ClusterLabel
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.vault.models import Note

KW = dict(min_notes=4, min_opinionated=3, opinion_threshold=0.5, seed=42)


def build(tmp_path, groups):
    """groups: list of (n_notes, opinion_scores) — each group gets its own axis direction."""
    m = Manifest(":memory:")
    s = FaissStore(tmp_path / "f.index", 8)
    rng = np.random.default_rng(0)
    for g, scores in enumerate(groups):
        for i, score in enumerate(scores):
            nid = f"g{g}n{i}"
            m.upsert_note(Note(id=nid, title=nid, body="b", path=f"{nid}.md", hash="h"))
            if score is not None:
                m.set_opinion(nid, score, [f"opinion {nid}"])
            v = np.eye(8)[g] + rng.normal(0, 0.01, 8)
            s.add(m.add_chunks(nid, ["p"]), (v / np.linalg.norm(v)).reshape(1, -1))
    return m, s


def label():
    return ClusterLabel(
        title="Fair difficulty", angle="Hard is fine if honest.", theme="difficulty"
    )


def test_only_qualifying_clusters_become_ideas(tmp_path):
    # group0 qualifies; group1 too few opinionated; group2 too small (but n//3 allows k=3)
    m, s = build(tmp_path, [[0.9, 0.8, 0.7, 0.1], [0.9, 0.2, 0.1, None], [0.9, 0.9, 0.9]])
    fake = FakeLLMClient({ClusterLabel: lambda _: label()})
    ideas = detect_ideas(m, s, fake, **KW)
    assert len(ideas) == 1
    idea = ideas[0]
    assert idea.note_ids == ["g0n0", "g0n1", "g0n2", "g0n3"]
    assert idea.idea_id == make_idea_id(["g0n3", "g0n0", "g0n1", "g0n2"])
    assert idea.opinion_score == 0.625
    assert [i.idea_id for i in m.list_ideas()] == [idea.idea_id]


def test_rerun_reuses_existing_idea_without_llm_call(tmp_path):
    m, s = build(tmp_path, [[0.9, 0.8, 0.7, 0.6], [0.1, 0.1, 0.1, 0.1]])
    fake = FakeLLMClient({ClusterLabel: lambda _: label()})
    detect_ideas(m, s, fake, **KW)
    calls = len(fake.calls)
    detect_ideas(m, s, fake, **KW)
    assert len(fake.calls) == calls


def test_done_ideas_preserved(tmp_path):
    m, s = build(tmp_path, [[0.9, 0.8, 0.7, 0.6], [0.1, 0.1, 0.1, 0.1]])
    done = Idea(
        idea_id="idea_old",
        title="t",
        angle="a",
        theme="x",
        note_ids=["z"],
        opinion_score=1.0,
        status="done",
    )
    m.save_ideas([done])
    detect_ideas(m, s, FakeLLMClient({ClusterLabel: lambda _: label()}), **KW)
    statuses = {i.idea_id: i.status for i in m.list_ideas()}
    assert statuses["idea_old"] == "done" and len(statuses) == 2


def test_empty_and_tiny_vaults(tmp_path):
    fake = FakeLLMClient({})
    assert detect_ideas(Manifest(":memory:"), FaissStore(tmp_path / "e.index", 8), fake, **KW) == []
    m, s = build(tmp_path, [[0.9, 0.9]])
    assert detect_ideas(m, s, fake, **KW) == []


def test_label_prompt_wraps_notes(tmp_path):
    m, s = build(tmp_path, [[0.9, 0.8, 0.7, 0.6], [0.1, 0.1, 0.1, 0.1]])
    fake = FakeLLMClient({ClusterLabel: lambda _: label()})
    detect_ideas(m, s, fake, **KW)
    prompt = fake.calls[0][1]
    assert prompt.count("<note id=") == 4 and "opinion g0n0" in prompt


def test_out_of_sync_index_raises_clear_error(tmp_path):
    import pytest

    from vault_to_voice.ideas.detector import IndexOutOfSyncError

    m, _ = build(tmp_path, [[0.9, 0.8, 0.7, 0.6]])
    wrong_dim = FaissStore(tmp_path / "other.index", 4)  # e.g. provider switched
    with pytest.raises(IndexOutOfSyncError, match="v2v ingest"):
        detect_ideas(m, wrong_dim, FakeLLMClient({}), **KW)


def build_works(tmp_path, works):
    """works: list of (work_id, axis, scores) — every note of a work points along `axis`."""
    m = Manifest(":memory:")
    s = FaissStore(tmp_path / "w.index", 8)
    rng = np.random.default_rng(1)
    for work_id, axis, scores in works:
        for i, score in enumerate(scores):
            nid = f"{work_id}@2026-01-{i + 1:02d}"
            m.upsert_note(
                Note(
                    id=nid, title=work_id, body="b", path=f"Daily/{i}.md", hash="h", work_id=work_id
                )
            )
            m.set_opinion(nid, score, [f"opinion {nid}"])
            v = np.eye(8)[axis] + rng.normal(0, 0.01, 8)
            s.add(m.add_chunks(nid, ["p"]), (v / np.linalg.norm(v)).reshape(1, -1))
    return m, s


def test_work_idea_for_well_covered_work(tmp_path):
    m, s = build_works(tmp_path, [("witcher-3", 0, [0.9, 0.8, 0.7, 0.2]), ("dune", 1, [0.9, 0.9])])
    ideas = detect_ideas(m, s, FakeLLMClient({ClusterLabel: lambda _: label()}), **KW)
    work = [i for i in ideas if i.kind == "work"]
    assert [i.idea_id for i in work] == ["idea_work_witcher-3"]
    assert work[0].work_ids == ["witcher-3"] and len(work[0].note_ids) == 4


def test_work_idea_id_stable_as_entries_grow(tmp_path):
    m, s = build_works(tmp_path, [("witcher-3", 0, [0.9, 0.8, 0.7, 0.6])])
    fake = FakeLLMClient({ClusterLabel: lambda _: label()})
    first = detect_ideas(m, s, fake, **KW)[0]
    m.set_idea_status(first.idea_id, "done")
    m.upsert_note(
        Note(
            id="witcher-3@2026-02-01",
            title="w",
            body="b",
            path="d.md",
            hash="h",
            work_id="witcher-3",
        )
    )
    v = np.eye(8)[0]
    s.add(m.add_chunks("witcher-3@2026-02-01", ["p"]), v.reshape(1, -1).astype(np.float32))
    calls = len(fake.calls)
    again = detect_ideas(m, s, fake, **KW)[0]
    assert again.idea_id == first.idea_id and len(again.note_ids) == 5
    assert again.status == "done" and again.title == first.title and len(fake.calls) == calls


def test_theme_idea_spans_works(tmp_path):
    hot = [("a0", 0, [0.9, 0.9]), ("a1", 0, [0.9, 0.8]), ("a2", 0, [0.8, 0.9])]
    cold = [("b0", 1, [0.1, 0.1]), ("b1", 1, [0.1, 0.2]), ("b2", 1, [0.2, 0.1])]
    m, s = build_works(tmp_path, hot + cold)
    ideas = detect_ideas(m, s, FakeLLMClient({ClusterLabel: lambda _: label()}), **KW)
    assert [(i.kind, i.work_ids) for i in ideas] == [("theme", ["a0", "a1", "a2"])]
    assert ideas[0].idea_id == make_idea_id(["a0", "a1", "a2"])


def test_single_work_cluster_is_not_a_theme(tmp_path):
    m, s = build_works(tmp_path, [("w", 0, [0.9, 0.9, 0.9, 0.9])])
    ideas = detect_ideas(m, s, FakeLLMClient({ClusterLabel: lambda _: label()}), **KW)
    assert [i.kind for i in ideas] == ["work"]


def test_work_label_prompt_mentions_single_work(tmp_path):
    m, s = build_works(tmp_path, [("witcher-3", 0, [0.9, 0.8, 0.7, 0.6])])
    fake = FakeLLMClient({ClusterLabel: lambda _: label()})
    detect_ideas(m, s, fake, **KW)
    assert "one work" in fake.calls[0][1]


def test_no_theme_ideas_without_real_clustering(tmp_path):
    # 4 works (< 6) cannot be split into >= 2 clusters, so no catch-all theme idea
    works = [(f"w{i}", 0, [0.9, 0.9]) for i in range(4)]
    m, s = build_works(tmp_path, works)
    assert detect_ideas(m, s, FakeLLMClient({}), **KW) == []
