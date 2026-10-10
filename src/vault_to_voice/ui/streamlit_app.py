"""Streamlit UI over the vault-to-voice API.

Run: `uv run streamlit run src/vault_to_voice/ui/streamlit_app.py` (with `v2v serve` running).
"""

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import streamlit as st

from vault_to_voice.ui import views
from vault_to_voice.ui.client import ApiClient, ApiError, Event


def get_client() -> ApiClient:
    # Tests put a client (over FastAPI's TestClient) into session_state before the first run.
    if "client" not in st.session_state:
        url = os.environ.get("V2V_API_URL", "http://127.0.0.1:8000")
        st.session_state["client"] = ApiClient.from_url(url)
    return st.session_state["client"]


def drive(events: Iterator[Event]) -> None:
    """Run one graph step, showing live node progress from the SSE stream, then redraw."""
    with st.status("Working...", expanded=True) as box:
        try:
            name, data = views.last_event(events, on_node=lambda n: box.write(f"finished `{n}`"))
        except (ApiError, httpx.HTTPError) as exc:
            name, data = "error", {"message": str(exc)}
        box.update(label="Step finished", state="error" if name == "error" else "complete")
    if name == "error":
        st.session_state["flash"] = data["message"]
    st.rerun()


def _ingest_summary(r: dict) -> str:
    text = (
        f"Indexed: new={r['new']} changed={r['changed']} deleted={r['deleted']} "
        f"unchanged={r['unchanged']} scored={r['scored']}"
    )
    if r.get("stats"):
        st_ = r["stats"]
        text += f" (daily files={st_['daily_files']}, entries={st_['entries']})"
    return text


def _style_summary(r: dict) -> str:
    return f"Style profile v{r['version']} saved. Tone: {', '.join(r['tone'])}. {r['voice_notes']}"


def _ideas_summary(ideas: list) -> str:
    return f"Found {len(ideas)} idea(s)." if ideas else "No ideas yet (need more notes)."


def run_setup(label: str, call, summarize) -> None:
    """Run one setup action (can take minutes on a local model), then redraw."""
    with st.status(f"{label}...", expanded=False) as box:
        try:
            st.session_state["setup_msg"] = summarize(call())
            box.update(label=f"{label}: done", state="complete")
        except (ApiError, httpx.HTTPError) as exc:
            st.session_state["flash"] = f"{label} failed: {getattr(exc, 'detail', exc)}"
            box.update(label=f"{label}: failed", state="error")
    st.rerun()


def setup_panel(client: ApiClient) -> None:
    with st.expander("Setup", expanded=False):
        try:
            cfg = client.config()
            st.caption(
                f"Vault: `{cfg['vault_path']}` ({cfg['vault_format']})  \n"
                f"LLM: `{cfg['llm_model']}` · embeddings: `{cfg['embedding_provider']}`  \n"
                f"Data: `{cfg['data_dir']}`"
            )
        except (ApiError, httpx.HTTPError) as exc:
            st.caption(f"config unavailable: {exc}")
        score = st.checkbox("Score opinions while indexing", value=True, key="setup_score")
        if st.button("Index vault", key="setup_ingest", width="stretch"):
            run_setup("Index vault", lambda: client.ingest(no_score=not score), _ingest_summary)
        if st.button("Build style profile", key="setup_style", width="stretch"):
            run_setup("Build style profile", client.build_style, _style_summary)
        if st.button("Find ideas", key="setup_ideas", width="stretch"):
            run_setup("Find ideas", client.detect_ideas, _ideas_summary)
        msg = st.session_state.pop("setup_msg", None)
        if msg:
            st.success(msg)


def sidebar(client: ApiClient) -> None:
    with st.sidebar:
        st.header("vault-to-voice")
        setup_panel(client)
        try:
            ideas = client.ideas()
            threads = {t["thread_id"]: t["status"] for t in client.threads()}
        except (ApiError, httpx.HTTPError) as exc:
            st.error(f"API unavailable: {exc}")
            return
        st.subheader("Ideas")
        if not ideas:
            st.caption(
                "No ideas yet. Open Setup above and click Index vault, "
                "Build style profile, then Find ideas."
            )
        for idea in ideas:
            status = threads.get(idea["idea_id"], "new")
            label = f"{idea['title']} · {status}"
            if st.button(label, key=f"idea_{idea['idea_id']}", width="stretch"):
                st.session_state["thread_id"] = idea["idea_id"]


def interview(client: ApiClient, view: dict, pending: dict) -> None:
    for qa in view["qa"]:
        st.chat_message("assistant").write(qa["question"])
        st.chat_message("user").write(qa["answer"])
    st.chat_message("assistant").write(
        f"**Q{pending['number']}/{pending['max']}:** {pending['question']}"
    )
    answer = st.text_area("Your answer", key=f"answer_{pending['number']}")
    if st.button("Send answer", key="send_answer"):
        drive(client.resume(view["thread_id"], answer))


def outline_review(client: ApiClient, view: dict, pending: dict) -> None:
    outline, rounds = pending["outline"], pending["rounds_left"]
    st.subheader("Outline review")
    st.caption(f"{rounds} revisions left")
    title = st.text_input("Title", outline["title"], key=f"ol_title_{rounds}")
    headings, points = [], []
    for i, section in enumerate(outline["sections"]):
        key = f"{rounds}_{i}"  # new keys each round, so a revised outline is not masked
        headings.append(st.text_input(f"Section {i + 1}", section["heading"], key=f"ol_h_{key}"))
        points.append(st.text_area("Point", section["point"], key=f"ol_p_{key}"))
    feedback = st.text_input("Feedback for a revision", key=f"ol_feedback_{rounds}")
    approve, revise, edit = st.columns(3)
    tid = view["thread_id"]
    if approve.button("Approve", key="ol_approve"):
        drive(client.resume(tid, {"action": "approve"}))
    if revise.button("Revise", key="ol_revise"):
        drive(client.resume(tid, {"action": "revise", "feedback": feedback}))
    if edit.button("Use my edits", key="ol_edit"):
        edited = views.edited_outline(outline, title, headings, points)
        drive(client.resume(tid, {"action": "edit", "edited": edited}))


def draft_review(client: ApiClient, view: dict, pending: dict) -> None:
    rounds, flags = pending["rounds_left"], pending["flags"]
    sections = (view.get("outline") or {}).get("sections", [])
    headings = {s["id"]: s["heading"] for s in sections}
    st.subheader("Draft review")
    coverage, flagged, style = st.columns(3)
    coverage.metric("Citation coverage", f"{pending['coverage']:.0%}")
    flagged.metric("Flagged sentences", len(flags))
    style.metric("Style match", views.style_label(pending.get("style_score")))
    for warning in pending.get("warnings", []):
        st.warning(warning)
    notes = pending.get("critic_notes", [])
    if notes:
        st.markdown("**Editor notes**\n\n" + views.critic_markdown(notes, headings))
    st.markdown(views.flagged_markdown(pending["draft"], flags, headings))
    tid = view["thread_id"]
    with st.expander("Edit sections"):
        original = views.section_texts(pending["draft"])
        edited = {
            sid: st.text_area(headings.get(sid, sid), text, key=f"dr_{rounds}_{sid}", height=200)
            for sid, text in original.items()
        }
        if st.button("Save edits", key="dr_edit"):
            changed = views.changed_sections(original, edited)
            if changed:
                drive(client.resume(tid, {"action": "edit", "edited": changed}))
            st.warning("No section was changed.")
    st.caption(f"{rounds} revisions left")
    feedback = st.text_input("Feedback for a revision", key=f"dr_feedback_{rounds}")
    approve, revise = st.columns(2)
    if approve.button("Approve and publish", key="dr_approve"):
        drive(client.resume(tid, {"action": "approve"}))
    if revise.button("Revise", key="dr_revise"):
        drive(client.resume(tid, {"action": "revise", "feedback": feedback}))


def published(client: ApiClient, view: dict) -> None:
    post = client.post(view["thread_id"])
    st.success(f"Published: {post['final_path']}")
    st.download_button(
        "Download Markdown",
        post["markdown"],
        file_name=Path(post["final_path"]).name,
        mime="text/markdown",
        key="download",
    )
    st.markdown(views.strip_frontmatter(post["markdown"]))
    sources = post["sources"]
    with st.expander("Sources"):
        st.metric("Citation coverage", f"{sources['citation_coverage']:.0%}")
        for paragraph in sources["paragraphs"]:
            st.markdown(f"> {paragraph['text']}")
            st.caption("sources: " + (", ".join(paragraph["sources"]) or "none"))


PENDING = {
    "interview_question": interview,
    "outline_review": outline_review,
    "draft_review": draft_review,
}


def main() -> None:
    st.set_page_config(page_title="vault-to-voice", layout="wide")
    client = get_client()
    sidebar(client)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.error(flash)
    thread_id = st.session_state.get("thread_id")
    if thread_id is None:
        st.info("Pick an idea in the sidebar to start a post.")
        return
    try:
        view = client.thread(thread_id)
    except (ApiError, httpx.HTTPError) as exc:
        st.error(str(exc))
        return
    st.title(view["title"])
    tid = view["thread_id"]
    if view["status"] == "new":
        if st.button("Start post", key="start"):
            drive(client.start(tid))
    elif view["status"] == "stalled":
        st.warning("The last step did not finish (an error, or the server stopped).")
        if st.button("Retry", key="retry"):
            drive(client.retry(tid))
    elif view["status"] == "published":
        published(client, view)
    else:
        pending = view["pending"]
        if pending.get("error"):
            st.error(pending["error"])
        PENDING[pending["type"]](client, view, pending)


main()
