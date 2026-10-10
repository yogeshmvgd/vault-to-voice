from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def structured(self, prompt: str, schema: type[T], *, system: str | None = None) -> T: ...


class LangChainLLMClient:
    """Provider-agnostic client: model is 'provider:model', e.g. 'openai:gpt-4o-mini'."""

    def __init__(self, model: str, *, timeout: float = 60.0, max_retries: int = 3):
        from langchain.chat_models import init_chat_model

        self._model = init_chat_model(model, timeout=timeout, max_retries=max_retries)

    def structured(self, prompt: str, schema: type[T], *, system: str | None = None) -> T:
        messages = [("system", system)] if system else []
        messages.append(("human", prompt))
        result = self._model.with_structured_output(schema).invoke(messages)
        return schema.model_validate(result)
