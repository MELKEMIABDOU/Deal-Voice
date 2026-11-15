import json
from contextvars import copy_context

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from deal_voice.config import settings
from deal_voice.generation import generator as generator_module
from deal_voice.generation.schemas import (
    GenerationPlan,
    GroundingCheck,
    decoding_schema,
)


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
                    return {"coverage": "complete", "verdicts": ["supported"]}

                return RunnableLambda(check)
            return RunnableLambda(
                lambda _: {"parts": [{"request": "Test", "status": "insufficient"}]}
            )

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
    assert captured["output_schemas"] == [decoding_schema(), GroundingCheck]
    branches = captured["output_schemas"][0]["$defs"]["AnswerPart"]["anyOf"]
    refusal = next(
        part
        for part in branches
        if part["properties"]["status"]["const"] == "insufficient"
    )
    assert set(refusal["properties"]) == {"request", "status"}
    assert refusal["additionalProperties"] is False


def test_ollama_schema_keeps_structure_without_expanding_bounded_strings(monkeypatch):
    """Exercise the real LangChain/Ollama adapter with an offline HTTP transport."""
    question = "Que faut-il acheter ?"
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        output = (
            {
                "parts": [
                    {
                        "request": question,
                        "status": "clarify",
                        "clarification": "criterion",
                    }
                ]
            }
            if len(requests) == 1
            else {"coverage": "complete", "verdicts": ["supported"]}
        )
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "message": {"role": "assistant", "content": json.dumps(output)},
                    "done": True,
                }
            )
            + "\n",
        )

    real_client = generator_module.ChatOllama

    def offline_client(**kwargs):
        kwargs["client_kwargs"]["transport"] = httpx.MockTransport(respond)
        return real_client(**kwargs)

    monkeypatch.setattr(generator_module, "ChatOllama", offline_client)
    generator = generator_module.RAGGenerator()
    answer = generator.generate(
        question,
        [
            Document(
                page_content="Catalogue des articles.", metadata={"source": "a.pdf"}
            )
        ],
    )
    assert "Quel critère" in answer
    assert generator.last_trace["status"] == "clarify"
    assert len(requests) == 2
    schema = requests[0]["format"]
    assert schema == decoding_schema()
    assert '"maxLength"' not in json.dumps(schema)
    assert schema["$defs"]["AnswerClaim"]["properties"]["text"]["minLength"] == 1
    assert schema["properties"]["parts"]["maxItems"] == 6
    assert requests[0]["options"]["num_predict"] == settings.ollama_num_predict
    assert requests[1]["format"] == GroundingCheck.model_json_schema()


@pytest.mark.parametrize(
    ("path", "limit"),
    [
        (("request",), 2000),
        (("claims", 0, "text"), 1200),
        (("recommendation", "action"), 500),
        (("recommendation", "criterion"), 300),
        (("table_queries", 0, "calculations", 0, "name"), 100),
    ],
)
def test_decoder_relaxation_preserves_python_length_validation(path, limit):
    decoding_schema()
    part = {
        "request": "Que faut-il acheter ?",
        "status": "answer",
        "claims": [{"text": "Stock faible.", "references": [{"evidence_id": "e1"}]}],
        "table_queries": [
            {
                "evidence_id": "e2",
                "operation": "select",
                "calculations": [{"name": "Ecart", "left": "Seuil", "right": "Stock"}],
            }
        ],
        "recommendation": {
            "action": "Vérifier les besoins.",
            "criterion": "Stock",
            "claim_indices": [0],
            "limits": ["Délai inconnu."],
        },
    }
    GenerationPlan.model_validate({"parts": [part]})
    field = part
    for key in path[:-1]:
        field = field[key]
    field[path[-1]] = "x" * (limit + 1)
    with pytest.raises(ValidationError) as error:
        GenerationPlan.model_validate({"parts": [part]})
    assert error.value.errors()[0]["type"] == "string_too_long"


def test_generation_trace_is_isolated_between_execution_contexts(offline_ollama):
    generator = generator_module.RAGGenerator()
    generator._record({"request": "first"})
    other = copy_context()
    other.run(generator._record, {"request": "second"})

    assert generator.last_trace == {"request": "first"}
    assert other.run(lambda: generator.last_trace) == {"request": "second"}
