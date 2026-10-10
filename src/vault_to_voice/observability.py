import os

import structlog

from vault_to_voice.config import Settings

log = structlog.get_logger()


def configure_tracing(settings: Settings) -> bool:
    """Opt-in LangSmith tracing for every LangChain/LangGraph call in this process.

    Off unless LANGSMITH_TRACING=true *and* LANGSMITH_API_KEY is set. LangChain reads these
    environment variables at call time, so setting them here is all the wiring needed.
    """
    if settings.langsmith_tracing and not settings.langsmith_api_key:
        log.warning("tracing_disabled", reason="LANGSMITH_TRACING=true but no LANGSMITH_API_KEY")
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    log.info("tracing_enabled", project=settings.langsmith_project)
    return True
