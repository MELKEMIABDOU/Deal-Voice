"""Keep generation unit tests offline, including provider construction."""

from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda


@pytest.fixture
def offline_ollama(monkeypatch):
    def forbidden(_):
        pytest.fail("A unit test attempted an unmocked Ollama call.")

    client = SimpleNamespace(
        with_structured_output=lambda schema, *, method: RunnableLambda(forbidden)
    )
    monkeypatch.setattr(
        "deal_voice.generation.generator.ChatOllama", lambda **kwargs: client
    )
