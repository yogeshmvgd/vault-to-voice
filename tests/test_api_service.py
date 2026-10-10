import json
from pathlib import Path

import pytest

from tests.fakes import graph_responses, make_service, sse_events
from vault_to_voice.api.service import Conflict, NotFound
from vault_to_voice.graph.state import Outline

ANSWERS = ["The tells.", "Twenty deaths.", "Patience."]
APPROVE = {"action": "approve"}


def last(frames):
    return sse_events(frames)[-1]


def test_new_thread_view(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    view = service.thread(idea.idea_id)
    assert view.status == "new" and view.pending is None and view.title == "Fair difficulty"
    assert service.list_threads() == []
    assert [i.idea_id for i in service.list_ideas()] == [idea.idea_id]


def test_full_run_through_the_service(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    tid = idea.idea_id
    assert last(service.start(tid))[1]["type"] == "interview_question"
    service_view = service.thread(tid)
    assert service_view.status == "waiting" and service_view.pending["number"] == 1
    for answer in ANSWERS[:2]:
        last(service.resume(tid, answer))
    view = service.thread(tid)
    assert [q["answer"] for q in view.qa] == ANSWERS[:2]  # read from the paused subgraph
    assert last(service.resume(tid, ANSWERS[2]))[1]["type"] == "outline_review"
    assert service.thread(tid).outline["title"] == "Fair Bosses"
    review = last(service.resume(tid, APPROVE))[1]
    assert review["type"] == "draft_review" and review["coverage"] == 1.0
    name, data = last(service.resume(tid, APPROVE))
    assert name == "done" and Path(data["final_path"]).exists()
    view = service.thread(tid)
    assert view.status == "published" and view.final_path == data["final_path"]
    assert [t.thread_id for t in service.list_threads()] == [tid]
    post = service.post(tid)
    assert post["markdown"].startswith("---\n")
    assert post["sources"] == json.loads(
        Path(data["final_path"]).with_suffix(".sources.json").read_text(encoding="utf-8")
    )


def test_unknown_idea_is_not_found(tmp_path, vault_dir):
    service, _, _ = make_service(tmp_path, vault_dir)
    with pytest.raises(NotFound, match="unknown idea"):
        service.start("idea_nope")
    with pytest.raises(NotFound):
        service.thread("idea_nope")


def test_wrong_state_operations_conflict(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    tid = idea.idea_id
    with pytest.raises(Conflict, match="nothing is waiting"):
        service.resume(tid, "answer")
    with pytest.raises(Conflict, match="not stalled"):
        service.retry(tid)
    last(service.start(tid))
    with pytest.raises(Conflict, match="already started"):
        service.start(tid)
    with pytest.raises(NotFound, match="not published"):
        service.post(tid)


def test_one_run_at_a_time_per_thread(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    other = idea.model_copy(update={"idea_id": "idea_other", "title": "Other"})
    service.deps.manifest.save_ideas([idea, other])
    running = service.start(idea.idea_id)
    next(running)  # first frame produced; the run holds the thread's lock
    with pytest.raises(Conflict, match="already running"):
        service.start(idea.idea_id)
    second = sse_events(service.start("idea_other"))  # other threads are not blocked
    assert second[-1][0] == "interrupt"
    list(running)  # exhausting the stream releases the lock
    assert last(service.resume(idea.idea_id, "ok"))[0] == "interrupt"


def test_dropped_stream_releases_the_lock(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    stream = service.start(idea.idea_id)
    stream.close()  # e.g. the client disconnected before the first frame
    assert last(service.start(idea.idea_id))[0] == "interrupt"


def test_failed_step_stalls_then_retry_continues(tmp_path, vault_dir):
    calls = {"n": 0}
    good = graph_responses()[Outline]

    def flaky_outline(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model is down")  # RuntimeError is not retried by LangGraph
        return good(prompt)

    service, idea, fake = make_service(tmp_path, vault_dir)
    fake.responses[Outline] = flaky_outline
    tid = idea.idea_id
    last(service.start(tid))
    last(service.resume(tid, ANSWERS[0]))
    last(service.resume(tid, ANSWERS[1]))
    assert last(service.resume(tid, ANSWERS[2])) == (
        "error",
        {"message": "RuntimeError: model is down"},
    )
    assert service.thread(tid).status == "stalled"
    assert last(service.retry(tid))[1]["type"] == "outline_review"


def test_malformed_review_decision_shows_the_review_again(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    tid = idea.idea_id
    last(service.start(tid))
    for answer in ANSWERS:
        last(service.resume(tid, answer))
    name, again = last(service.resume(tid, {"action": "publish-now"}))
    assert name == "interrupt" and again["type"] == "outline_review" and again["error"]
