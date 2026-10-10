from langgraph.store.base import BaseStore

from vault_to_voice.graph.state import QA
from vault_to_voice.style.profile import StyleProfile

STYLE_NS = ("style",)
STYLE_KEY = "profile"
ANSWERS_NS = ("answers",)


def put_style(store: BaseStore, profile: StyleProfile) -> None:
    store.put(STYLE_NS, STYLE_KEY, profile.model_dump(mode="json"))


def get_style(store: BaseStore) -> dict | None:
    item = store.get(STYLE_NS, STYLE_KEY)
    return item.value if item else None


def save_answers(store: BaseStore, idea_id: str, qa: list[QA]) -> None:
    store.put(ANSWERS_NS, idea_id, {"qa": [q.model_dump() for q in qa]})


def past_answers(store: BaseStore, *, exclude_idea: str | None = None, limit: int = 10) -> list[QA]:
    items = sorted(store.search(ANSWERS_NS, limit=100), key=lambda i: i.updated_at, reverse=True)
    answers = [
        QA.model_validate(q) for item in items if item.key != exclude_idea for q in item.value["qa"]
    ]
    return answers[:limit]
