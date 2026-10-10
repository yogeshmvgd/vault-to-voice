from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from vault_to_voice.graph.deps import LLM_RETRY, GraphDeps
from vault_to_voice.graph.prompts import notes_block, qa_block
from vault_to_voice.graph.state import QA, CoverageCheck, PostState, Question
from vault_to_voice.guardrails import DATA_INSTRUCTION

QUESTION_PROMPT = (
    "You are interviewing the author to gather material for a blog post.\n"
    "Post idea: {title} — {angle}\n"
    "Ask ONE short, specific question that draws out a personal opinion, memory or example "
    "that the notes and previous answers do not already cover. Never repeat a question.\n"
    "grounding_note_ids: ids of the notes your question builds on.\n"
)
COVERAGE_PROMPT = (
    "Decide whether this interview has gathered enough personal material (opinions, memories, "
    "concrete examples) to write the post. enough: true or false. missing: what is still "
    "missing, if anything.\n"
)


def ask_question(state: PostState, deps: GraphDeps) -> dict:
    idea = state["idea"]
    parts = [
        QUESTION_PROMPT.format(title=idea.title, angle=idea.angle),
        "Notes:",
        notes_block(state["notes"]),
    ]
    if state.get("qa"):
        parts += ["Answers so far:", qa_block(state["qa"])]
    if state.get("past_answers"):
        parts += [
            "Already answered in earlier interviews (do not ask again):",
            qa_block(state["past_answers"]),
        ]
    question = deps.llm.structured("\n\n".join(parts), Question, system=DATA_INSTRUCTION)
    known = {n.note_id for n in state["notes"]}
    grounding = [i for i in question.grounding_note_ids if i in known]
    return {"pending_question": question.model_copy(update={"grounding_note_ids": grounding})}


def wait_for_answer(state: PostState, deps: GraphDeps) -> dict:
    # No LLM call here: on resume LangGraph re-runs this whole node, and interrupt() returns
    # the resume value instead of pausing.
    question = state["pending_question"]
    qa = state.get("qa", [])
    number = len(qa) + 1
    answer = interrupt(
        {
            "type": "interview_question",
            "question": question.question,
            "grounding_note_ids": question.grounding_note_ids,
            "number": number,
            "max": deps.settings.interview_max_questions,
        }
    )
    text = str(answer).strip() or "(skipped)"
    new = QA(
        qa_id=f"qa:{number}",
        question=question.question,
        answer=text,
        grounding_note_ids=question.grounding_note_ids,
    )
    return {"qa": [*qa, new], "pending_question": None}


def check_coverage(state: PostState, deps: GraphDeps) -> dict:
    n = len(state.get("qa", []))
    if n < deps.settings.interview_min_questions:
        return {"interview_done": False}
    if n >= deps.settings.interview_max_questions:
        return {"interview_done": True}
    idea = state["idea"]
    prompt = "\n\n".join(
        [COVERAGE_PROMPT, f"Post idea: {idea.title} — {idea.angle}", qa_block(state["qa"])]
    )
    check = deps.llm.structured(prompt, CoverageCheck, system=DATA_INSTRUCTION)
    return {"interview_done": check.enough}


def build_interview_graph(deps: GraphDeps) -> StateGraph:
    g = StateGraph(PostState)
    g.add_node("ask_question", partial(ask_question, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("wait_for_answer", partial(wait_for_answer, deps=deps))
    g.add_node("check_coverage", partial(check_coverage, deps=deps), retry_policy=LLM_RETRY)
    g.add_edge(START, "ask_question")
    g.add_edge("ask_question", "wait_for_answer")
    g.add_edge("wait_for_answer", "check_coverage")
    g.add_conditional_edges(
        "check_coverage",
        lambda s: END if s["interview_done"] else "ask_question",
        ["ask_question", END],
    )
    return g
