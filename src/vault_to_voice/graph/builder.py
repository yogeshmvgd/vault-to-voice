from functools import partial

from langgraph.graph import END, START, StateGraph

from vault_to_voice.graph.deps import LLM_RETRY, GraphDeps
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.interview import build_interview_graph
from vault_to_voice.graph.nodes.outline import (
    make_outline,
    outline_review,
    route_after_outline_review,
)
from vault_to_voice.graph.state import PostState


def build_post_graph(deps: GraphDeps, *, checkpointer=None):
    g = StateGraph(PostState)
    g.add_node("load_context", partial(load_context, deps=deps))
    g.add_node("interview", build_interview_graph(deps).compile())  # subgraph as a node
    g.add_node("make_outline", partial(make_outline, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("outline_review", partial(outline_review, deps=deps))

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "interview")
    g.add_edge("interview", "make_outline")
    g.add_edge("make_outline", "outline_review")
    g.add_conditional_edges(
        "outline_review",
        route_after_outline_review,
        {"make_outline": "make_outline", "outline_review": "outline_review", "dispatch": END},
    )
    return g.compile(checkpointer=checkpointer, store=deps.store)
