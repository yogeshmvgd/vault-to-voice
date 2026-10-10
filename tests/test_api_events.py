from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import FakeLLMClient, graph_responses, make_deps, seed_idea, sse_events
from vault_to_voice.api.events import parse_sse, sse, stream_run
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.runner import start_input, thread_config


def test_sse_frame_round_trips():
    frame = sse("interrupt", {"type": "outline_review", "title": "Café\nmenu"})
    assert frame.startswith("event: interrupt\ndata: ") and frame.endswith("\n\n")
    assert frame.count("\n") == 3  # JSON escapes the newline, so one data line
    assert list(parse_sse(frame.split("\n"))) == [
        ("interrupt", {"type": "outline_review", "title": "Café\nmenu"})
    ]


def test_parse_sse_ignores_comments_and_flushes_the_last_event():
    lines = [": keep-alive", "event: node", 'data: {"node": "a"}', "", "event: done", "data: {}"]
    assert list(parse_sse(lines)) == [("node", {"node": "a"}), ("done", {})]


def test_stream_run_yields_nodes_then_the_interrupt(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient(graph_responses()))
    idea = seed_idea(deps, vault_dir)
    graph = build_post_graph(deps, checkpointer=InMemorySaver())
    events = sse_events(stream_run(graph, thread_config(idea.idea_id), start_input(idea)))
    assert events[0] == ("node", {"node": "load_context"})
    name, value = events[-1]
    assert name == "interrupt" and value["type"] == "interview_question"
    assert value["number"] == 1


class _Boom:
    def stream(self, *args, **kwargs):
        raise RuntimeError("model is down")


def test_stream_run_turns_exceptions_into_an_error_event():
    events = sse_events(stream_run(_Boom(), thread_config("t1"), None))
    assert events == [("error", {"message": "RuntimeError: model is down"})]
