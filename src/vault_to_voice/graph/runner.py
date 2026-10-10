from vault_to_voice.ideas.detector import Idea


def thread_config(thread_id: str) -> dict:
    """Run config for one post thread; run_name/tags/metadata label its LangSmith traces."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "post_graph",
        "tags": ["vault-to-voice", "post-graph"],
        "metadata": {"idea_id": thread_id},
    }


def start_input(idea: Idea) -> dict:
    return {"idea": idea}


def run_until_interrupt(graph, config: dict, value) -> dict | None:
    """Invoke (start, resume via Command, or retry via None); return the next interrupt or None."""
    result = graph.invoke(value, config)
    pending = result.get("__interrupt__") if isinstance(result, dict) else None
    return pending[0].value if pending else None


def pending_interrupt(graph, config: dict) -> dict | None:
    snapshot = graph.get_state(config)
    return snapshot.interrupts[0].value if snapshot.interrupts else None
