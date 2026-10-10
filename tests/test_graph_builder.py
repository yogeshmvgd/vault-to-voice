from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.checkpoint import make_checkpointer
from vault_to_voice.graph.runner import (
    pending_interrupt,
    run_until_interrupt,
    start_input,
    thread_config,
)
from vault_to_voice.graph.state import CoverageCheck, Outline, Question, Section


def fake_llm():
    count = {"n": 0}

    def ask(_prompt):
        count["n"] += 1
        return Question(question=f"Q{count['n']}?")

    return FakeLLMClient(
        {
            Question: ask,
            CoverageCheck: lambda _: CoverageCheck(enough=True),
            Outline: lambda _: Outline(
                title="Fair Bosses", sections=[Section(id="x", heading="H", point="P")]
            ),
        }
    )


def test_interview_then_outline_review_and_revise_loop(tmp_path, vault_dir):
    deps = make_deps(tmp_path, fake_llm())
    idea = seed_idea(deps, vault_dir)
    graph = build_post_graph(deps, checkpointer=InMemorySaver())
    cfg = thread_config(idea.idea_id)
    value = run_until_interrupt(graph, cfg, start_input(idea))
    questions = 0
    while value["type"] == "interview_question":
        questions += 1
        value = run_until_interrupt(graph, cfg, Command(resume=f"answer {questions}"))
    assert questions == 3 and value["type"] == "outline_review"
    assert pending_interrupt(graph, cfg)["type"] == "outline_review"
    value = run_until_interrupt(graph, cfg, Command(resume={"action": "revise", "feedback": "x"}))
    assert value["type"] == "outline_review" and value["rounds_left"] == 2
    assert sum(1 for s, _ in deps.llm.calls if s is Outline) == 2


def test_resume_after_restart_with_sqlite(tmp_path, vault_dir):
    deps = make_deps(tmp_path, fake_llm())
    idea = seed_idea(deps, vault_dir)
    path = tmp_path / "cp.db"
    cfg = thread_config(idea.idea_id)
    first = build_post_graph(deps, checkpointer=make_checkpointer(path))
    run_until_interrupt(first, cfg, start_input(idea))
    run_until_interrupt(first, cfg, Command(resume="one"))
    restarted = build_post_graph(deps, checkpointer=make_checkpointer(path))  # new process
    assert pending_interrupt(restarted, cfg)["number"] == 2
    assert run_until_interrupt(restarted, cfg, Command(resume="two"))["number"] == 3
    assert sum(1 for s, _ in deps.llm.calls if s is Question) == 3  # nothing re-asked
