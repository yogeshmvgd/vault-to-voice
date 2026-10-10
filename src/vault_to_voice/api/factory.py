from vault_to_voice.api.service import RunService
from vault_to_voice.config import Settings
from vault_to_voice.graph.checkpoint import make_checkpointer, make_store
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.index.embedder import make_embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient
from vault_to_voice.publish import OutboxPublisher


def build_service(settings: Settings, llm: LLMClient, *, tracing: bool = False) -> RunService:
    """Production wiring: SQLite manifest/checkpoints/Store under DATA_DIR, outbox publisher."""
    embedder = make_embedder(settings)
    deps = GraphDeps(
        llm=llm,
        manifest=Manifest(settings.manifest_path),
        index=FaissStore(settings.faiss_path, embedder.dim),
        embedder=embedder,
        settings=settings,
        store=make_store(settings.store_path),
        publisher=OutboxPublisher(settings.outbox_dir),
    )
    return RunService(deps, make_checkpointer(settings.checkpoint_path), tracing=tracing)
