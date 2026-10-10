from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vault_to_voice.api.service import RunService, ServiceError, ThreadView
from vault_to_voice.ideas.detector import Idea


class StartRequest(BaseModel):
    idea_id: str


class ResumeRequest(BaseModel):
    # Interview: the answer text. Reviews: {"action": "approve"|"revise"|"edit", ...}.
    payload: str | dict


class PostOut(BaseModel):
    final_path: str
    markdown: str
    sources: dict


def _sse(frames: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(
        frames,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(service: RunService) -> FastAPI:
    app = FastAPI(title="vault-to-voice", version="0.1.0")
    app.state.service = service

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "tracing": service.tracing}

    @app.get("/ideas")
    def ideas() -> list[Idea]:
        return service.list_ideas()

    @app.get("/threads")
    def threads() -> list[ThreadView]:
        return service.list_threads()

    @app.post("/threads")
    def start(body: StartRequest) -> StreamingResponse:
        return _sse(service.start(body.idea_id))

    @app.get("/threads/{thread_id}")
    def thread(thread_id: str) -> ThreadView:
        return service.thread(thread_id)

    @app.post("/threads/{thread_id}/resume")
    def resume(thread_id: str, body: ResumeRequest) -> StreamingResponse:
        return _sse(service.resume(thread_id, body.payload))

    @app.get("/threads/{thread_id}/stream")
    def stream(thread_id: str) -> StreamingResponse:
        """Continue a stalled thread (a step failed or the server died mid-step)."""
        return _sse(service.retry(thread_id))

    @app.get("/threads/{thread_id}/post")
    def post(thread_id: str) -> PostOut:
        return PostOut(**service.post(thread_id))

    return app
