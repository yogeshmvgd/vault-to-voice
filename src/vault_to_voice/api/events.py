import json
from collections.abc import Iterable, Iterator
from typing import Any

import structlog

log = structlog.get_logger()


def sse(event: str, data: Any) -> str:
    """One Server-Sent Events frame. JSON never contains raw newlines, so one data line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, Any]]:
    """Inverse of `sse` over a stream of lines (as from httpx `iter_lines`)."""
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if event is not None or data:
                yield event or "message", json.loads("\n".join(data)) if data else None
            event, data = None, []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
    if event is not None or data:
        yield event or "message", json.loads("\n".join(data)) if data else None


def stream_run(graph, config: dict, value) -> Iterator[str]:
    """Run the post graph until its next interrupt (or the end), yielding SSE frames.

    Events: `node` {node} per finished node · `interrupt` {...payload} · `done` {final_path}
    · `error` {message}. On error the thread stays at its last checkpoint and can be retried.
    """
    thread_id = config["configurable"]["thread_id"]
    try:
        for chunk in graph.stream(value, config, stream_mode="updates"):
            for name, update in chunk.items():
                if name == "__interrupt__":
                    yield sse("interrupt", update[0].value)
                else:
                    yield sse("node", {"node": name})
        snapshot = graph.get_state(config)
    except Exception as exc:
        log.warning("run_failed", thread_id=thread_id, error=repr(exc))
        yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        return
    if not snapshot.next:
        log.info("run_done", thread_id=thread_id)
        yield sse("done", {"final_path": snapshot.values.get("final_path")})
