from contextvars import copy_context

from langchain_core.runnables import RunnableLambda

from deal_voice.config import settings
from deal_voice.generation import generator as generator_module
from deal_voice.generation.schemas import GroundingCheck


def test_ollama_generation_uses_configured_memory_limits(monkeypatch):
    captured = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema, *, method):
            captured.setdefault("output_schemas", []).append(schema)
            captured["output_schema"] = schema
            captured["output_method"] = method
            if schema is GroundingCheck:

                def check(messages, **kwargs):
                    captured["verification_options"] = kwargs["options"]
                    return {"verdict": "supported"}

                return RunnableLambda(check)
            return RunnableLambda(lambda _: {"status": "insufficient"})

    monkeypatch.setattr(generator_module, "ChatOllama", FakeChatOllama)

    generator = generator_module.RAGGenerator(model_name="qwen3:8b")
    generator.verifier.invoke([])

    assert captured["base_url"] == settings.ollama_base_url
    assert captured["num_ctx"] == settings.ollama_num_ctx
    assert captured["num_predict"] == settings.ollama_num_predict
    assert captured["num_thread"] == settings.ollama_num_thread
    assert captured["reasoning"] is settings.ollama_reasoning
    assert captured["keep_alive"] == settings.ollama_keep_alive
    assert (
        captured["client_kwargs"]["timeout"] == settings.ollama_request_timeout_seconds
    )
    assert captured["output_method"] == "json_schema"
    assert captured["verification_options"] == {
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": generator_module.VERIFICATION_OUTPUT_TOKENS,
        "num_thread": settings.ollama_num_thread,
        "temperature": 0.0,
    }
    branches = next(
        schema for schema in captured["output_schemas"] if isinstance(schema, dict)
    )["anyOf"]
    refusal = next(
        item
        for item in branches
        if item["properties"]["status"]["const"] == "insufficient"
    )
    assert set(refusal["properties"]) == {"status"}
    assert refusal["additionalProperties"] is False


def test_generation_trace_is_isolated_between_execution_contexts(offline_ollama):
    generator = generator_module.RAGGenerator()
    generator._record({"request": "first"})
    other = copy_context()
    other.run(generator._record, {"request": "second"})

    assert generator.last_trace == {"request": "first"}
    assert other.run(lambda: generator.last_trace) == {"request": "second"}
