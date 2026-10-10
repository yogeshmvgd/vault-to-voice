from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.interview import build_interview_graph
from vault_to_voice.graph.state import CoverageCheck, Question

CFG = {"configurable": {"thread_id": "t"}}


def run_interview(tmp_path, vault_dir, coverage, answers):
    count = {"n": 0}

    def ask(_prompt):
        count["n"] += 1
        return Question(question=f"Question {count['n']}?", grounding_note_ids=["n0", "bogus"])

    fake = FakeLLMClient({Question: ask, CoverageCheck: lambda _: CoverageCheck(enough=coverage)})
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    state = {"idea": idea, **load_context({"idea": idea}, deps)}
    app = build_interview_graph(deps).compile(checkpointer=InMemorySaver())
    result = app.invoke(state, CFG)
    asked = []
    for answer in answers:
        if "__interrupt__" not in result:
            break
        asked.append(result["__interrupt__"][0].value)
        result = app.invoke(Command(resume=answer), CFG)
    return result, asked, fake


def calls(fake, schema):
    return [prompt for s, prompt in fake.calls if s is schema]


def test_min_three_questions_even_if_covered(tmp_path, vault_dir):
    result, _, fake = run_interview(tmp_path, vault_dir, True, ["a"] * 6)
    assert len(result["qa"]) == 3 and "__interrupt__" not in result
    assert len(calls(fake, CoverageCheck)) == 1  # only consulted once the minimum is reached


def test_max_five_questions(tmp_path, vault_dir):
    result, _, _ = run_interview(tmp_path, vault_dir, False, ["a"] * 9)
    assert len(result["qa"]) == 5


def test_stops_between_min_and_max_when_covered(tmp_path, vault_dir):
    count = {"n": 0}

    def coverage(_prompt):
        count["n"] += 1
        return CoverageCheck(enough=count["n"] >= 2)  # not enough at 3, enough at 4

    fake = FakeLLMClient({Question: lambda _: Question(question="Q?"), CoverageCheck: coverage})
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    app = build_interview_graph(deps).compile(checkpointer=InMemorySaver())
    result = app.invoke({"idea": idea, **load_context({"idea": idea}, deps)}, CFG)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume="a"), CFG)
    assert len(result["qa"]) == 4


def test_question_payload_qa_ids_and_no_reask_on_resume(tmp_path, vault_dir):
    result, asked, fake = run_interview(tmp_path, vault_dir, True, ["x", "y", "z"])
    assert asked[0] == {
        "type": "interview_question",
        "question": "Question 1?",
        "grounding_note_ids": ["n0"],
        "number": 1,
        "max": 5,
    }
    assert [q.qa_id for q in result["qa"]] == ["qa:1", "qa:2", "qa:3"]
    assert len(calls(fake, Question)) == 3  # resuming re-runs wait_for_answer, not ask_question


def test_blank_answer_recorded_as_skipped(tmp_path, vault_dir):
    result, _, _ = run_interview(tmp_path, vault_dir, True, ["   ", "b", "c"])
    assert result["qa"][0].answer == "(skipped)"


def test_prompt_includes_notes_and_previous_answers(tmp_path, vault_dir):
    _, _, fake = run_interview(tmp_path, vault_dir, True, ["I loved the tells", "b", "c"])
    second = calls(fake, Question)[1]
    assert '<note id="n0">' in second and "I loved the tells" in second
