from functools import partial

from langgraph.graph import END, START, StateGraph

from vault_to_voice.graph.deps import LLM_RETRY, GraphDeps
from vault_to_voice.graph.nodes.citations import check_citations
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.draft import draft_section, fan_out_sections, merge_draft
from vault_to_voice.graph.nodes.interview import build_interview_graph
from vault_to_voice.graph.nodes.outline import (
    make_outline,
    outline_review,
    route_after_outline_review,
)
from vault_to_voice.graph.nodes.publish import publish
from vault_to_voice.graph.nodes.review import (
    draft_review,
    prepare_auto_revise,
    route_after_checks,
    route_after_draft_review,
)
from vault_to_voice.graph.state import PostState


def _dispatch(state: PostState) -> dict:
    """Join point before the Send fan-out (outline approval, auto-revise and user revise)."""
    return {"review_error": None}


def build_post_graph(deps: GraphDeps, *, checkpointer=None):
    g = StateGraph(PostState)
    g.add_node("load_context", partial(load_context, deps=deps))
    g.add_node("interview", build_interview_graph(deps).compile())  # subgraph as a node
    g.add_node("make_outline", partial(make_outline, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("outline_review", partial(outline_review, deps=deps))
    g.add_node("dispatch", _dispatch)
    g.add_node("draft_section", partial(draft_section, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("merge_draft", merge_draft)
    g.add_node("check_citations", partial(check_citations, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("prepare_auto_revise", prepare_auto_revise)
    g.add_node("draft_review", partial(draft_review, deps=deps))
    g.add_node("publish", partial(publish, deps=deps))

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "interview")
    g.add_edge("interview", "make_outline")
    g.add_edge("make_outline", "outline_review")
    g.add_conditional_edges(
        "outline_review", route_after_outline_review, ["make_outline", "outline_review", "dispatch"]
    )
    g.add_conditional_edges("dispatch", fan_out_sections, ["draft_section"])
    g.add_edge("draft_section", "merge_draft")
    g.add_edge("merge_draft", "check_citations")
    g.add_conditional_edges(
        "check_citations", route_after_checks, ["prepare_auto_revise", "draft_review"]
    )
    g.add_edge("prepare_auto_revise", "dispatch")
    g.add_conditional_edges(
        "draft_review",
        route_after_draft_review,
        ["dispatch", "check_citations", "draft_review", "publish"],
    )
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer, store=deps.store)
