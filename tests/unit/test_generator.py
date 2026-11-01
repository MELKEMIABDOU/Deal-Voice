"""Generation contracts, without a running LLM or model downloads."""

import json

import pytest
from langchain_core.documents import Document

from deal_voice.config import settings
from deal_voice.evidence.citations import evidence_from_documents
from deal_voice.generation.generator import VERIFICATION_OUTPUT_TOKENS, RAGGenerator
from deal_voice.generation.prompts import (
    SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    prompt_values,
)
from deal_voice.generation.schemas import (
    GenerationPlan,
    GroundingCheck,
    NarrativePlan,
    TablePlan,
)

pytestmark = pytest.mark.usefixtures("offline_ollama")


class FakeChain:
    def __init__(self, response):
        self.payload = None
        self.response = response

    def invoke(self, payload):
        self.payload = payload
        entries = [json.loads(line) for line in payload["context"].splitlines()[1:]]
        return self.response(entries)


class FakeVerifier:
    def __init__(self, verdict="supported"):
        self.verdict = verdict
        self.payload = None

    def invoke(self, messages):
        self.payload = json.loads(messages[-1][1])
        return {"verdict": self.verdict}


def generator_for(response, *, verdict="supported"):
    generator = RAGGenerator()
    generator.chains = dict.fromkeys(generator.chains, FakeChain(response))
    generator.verifier = FakeVerifier(verdict)
    return generator


def narrative(text="Le contrat commence lundi.", **metadata):
    return Document(page_content=text, metadata={"source": "contrat.txt", **metadata})


def table_document():
    return Document(
        page_content=(
            "## Inventaire\n"
            "| Identifiant | Article | Charge | Famille |\n"
            "| --- | --- | --- | --- |\n"
            "| Z3 | Alpha | 1,50 € | F1 |\n"
            "| A1 | Beta | 9,75 € | F2 |\n"
            "| B2 | Gamma | 9,75 € | F2 |\n"
        ),
        metadata={
            "source": "registre.pdf",
            "content_kind": "table",
            "table_title": "Inventaire",
            "table_complete": True,
        },
    )


def test_planner_schemas_do_not_allow_numeric_locale_decisions():
    flat = TablePlan.model_json_schema()["properties"]["decimal_separator"]
    mixed = GenerationPlan.model_json_schema()["$defs"]["PlannedTableQuery"]
    assert flat["type"] == "null"
    assert mixed["properties"]["decimal_separator"]["type"] == "null"
    for separator in (",", "."):
        with pytest.raises(ValueError):
            TablePlan(evidence_id="E", operation="max", decimal_separator=separator)
        with pytest.raises(ValueError):
            GenerationPlan.model_validate(
                {
                    "status": "answer",
                    "table_queries": [
                        {
                            "evidence_id": "E",
                            "operation": "max",
                            "decimal_separator": separator,
                        }
                    ],
                }
            )


@pytest.mark.parametrize(
    "separator,formatted", [(None, "1,234"), (",", "1,234"), (".", "1.234")]
)
def test_only_source_metadata_can_disambiguate_numeric_cells(separator, formatted):
    document = Document(
        page_content=f"| Code | Valeur |\n| --- | --- |\n| A | {formatted} |",
        metadata={
            "source": "values.pdf",
            "content_kind": "table",
            "table_complete": True,
            "decimal_separator": separator,
        },
    )
    generator = generator_for(
        lambda entries: query_plan(entries, "sum", column="Valeur")
    )
    answer = generator.generate("Quelle est la somme ?", [document])
    if separator is None:
        assert generator.last_trace["status"] == "invalid_plan"
    else:
        assert generator.last_trace["status"] == "answer"
        assert "1.234" in answer
        assert (
            generator.last_trace["computations"][0]["query"]["decimal_separator"]
            == separator
        )


def claim_plan(entries, text="Le contrat commence lundi."):
    return {
        "status": "answer",
        "claims": [
            {"text": text, "references": [{"evidence_id": entries[0]["evidence_id"]}]}
        ],
    }


def query_plan(entries, operation="select", **kwargs):
    return {"evidence_id": entries[0]["evidence_id"], "operation": operation, **kwargs}


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Combien de collaborateurs compte l’agence ?", "French"),
        ("Comment fonctionne le produit ?", "French"),
        ("Pourquoi ce choix ?", "French"),
        ("Quand commence le contrat ?", "French"),
        ("Quels produits sont disponibles ?", "French"),
        ("How many employees does the agency have?", "English"),
    ],
)
def test_language_constraint_is_transmitted_to_planner(question, language):
    generator = generator_for(claim_plan)
    answer = generator.generate(question, [narrative()])
    assert (
        generator.chains[NarrativePlan]
        .payload["answer_language_instruction"]
        .startswith(f"OUTPUT LANGUAGE: {language}.")
    )
    assert "Source : contrat.txt" in answer


def test_tables_expose_schema_to_model_and_execute_maximum_with_all_ties():
    def response(entries):
        assert "Beta" not in json.dumps(entries)
        assert entries[0]["columns"] == ["Identifiant", "Article", "Charge", "Famille"]
        return query_plan(entries, "max", column="Charge")

    generator = generator_for(response)
    answer = generator.generate("Quel est le maximum ?", [table_document()])
    assert "Beta" in answer and "Gamma" in answer
    assert "| Z3 |" not in answer
    assert "9,75 €" in answer
    assert "Source : registre.pdf — table « Inventaire »" in answer
    assert {
        "entity": "A1",
        "field": "Charge",
        "value": "9,75 €",
    } in generator.last_trace["actual_values"]


def test_lookup_and_filter_use_unmodified_schema_and_source_values():
    def response(entries):
        return query_plan(
            entries,
            "select",
            filters=[{"column": "Famille", "operator": "eq", "value": "F2"}],
            select_columns=["Article", "Charge"],
        )

    generator = generator_for(response)
    answer = generator.generate("Quels articles de la famille F2 ?", [table_document()])
    assert "| A1 | Beta | 9,75 € |" in answer
    assert "| B2 | Gamma | 9,75 € |" in answer
    assert "Alpha" not in answer


def test_table_cell_claims_cannot_bypass_executor():
    generator = generator_for(
        lambda entries: claim_plan(entries, "La charge vaut 999 €.")
    )
    answer = generator.generate("Quel est le montant ?", [table_document()])
    assert "999" not in answer
    assert generator.last_trace["status"] == "invalid_plan"


@pytest.mark.parametrize("kind", ["reference", "column", "locator", "schema", "locale"])
def test_invalid_plan_fails_closed(kind):
    def response(entries):
        evidence_id = entries[0]["evidence_id"]
        if kind in {"reference", "locator"}:
            return {
                "status": "answer",
                "claims": [
                    {
                        "text": "Une affirmation.",
                        "references": [
                            {
                                "evidence_id": "invented"
                                if kind == "reference"
                                else evidence_id,
                                "locator": "Invented section"
                                if kind == "locator"
                                else None,
                            }
                        ],
                    }
                ],
            }
        if kind == "schema":
            return {"status": "answer", "generated_python": "print(1)"}
        return query_plan(
            entries,
            "max",
            column="Unknown" if kind == "column" else "Charge",
            decimal_separator="," if kind == "locale" else None,
        )

    document = narrative() if kind in {"reference", "locator"} else table_document()
    generator = generator_for(response)
    generator.generate("Quelle est la réponse ?", [document])
    assert generator.last_trace["status"] == "invalid_plan"


def test_multiple_passages_from_same_file_retain_their_own_locators():
    documents = [
        narrative("Le délai est de dix jours.", section="Conditions", page=2),
        narrative("Le paiement intervient lundi.", section="Paiement", page=5),
    ]

    def response(entries):
        return {
            "status": "answer",
            "claims": [
                {
                    "text": "Le délai est de dix jours.",
                    "references": [{"evidence_id": entries[0]["evidence_id"]}],
                },
                {
                    "text": "Le paiement intervient lundi.",
                    "references": [{"evidence_id": entries[1]["evidence_id"]}],
                },
            ],
        }

    answer = generator_for(response).generate(
        "Quels sont les délais et le paiement ?", documents
    )
    assert "section « Conditions » — page 2" in answer
    assert "section « Paiement » — page 5" in answer
    assert "Lien entre" not in answer


def test_explicit_insufficiency_and_clarification_are_distinct():
    generator = generator_for(lambda _: {"status": "insufficient"})
    assert "Je ne dispose pas" in generator.generate(
        "Quel est le revenu ?", [narrative()]
    )
    generator.chains = dict.fromkeys(
        generator.chains,
        FakeChain(lambda _: {"status": "clarify", "clarification": "column"}),
    )
    assert (
        generator.generate("Quel est le revenu ?", [narrative()])
        == "Quelle colonne souhaitez-vous utiliser ?"
    )
    assert generator.last_trace["status"] == "clarify"


def test_context_limit_prevents_llm_call_without_splitting_evidence(monkeypatch):
    monkeypatch.setattr(settings, "ollama_num_ctx", 512)
    generator = generator_for(
        lambda _: pytest.fail("Oversized context must not reach the model.")
    )
    answer = generator.generate("Quel est le contenu ?", [narrative("x" * 10000)])
    assert generator.last_trace["status"] == "context_limit"
    assert "capacité" in answer


def test_normal_audio_parent_fits_without_truncation_or_larger_context(monkeypatch):
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)
    monkeypatch.setattr(settings, "ollama_num_predict", 512)
    content = "**SPEAKER_00**: Nous fabriquons des instruments de mesure.\n\n"
    content += (
        "**SPEAKER_01**: " + "Nous précisons les modalités du projet. " * 15 + "\n\n"
    ) * 4
    document = narrative(
        content,
        source="appel.mp3",
        file_type="audio",
        start_time=8.5,
        end_time=111.5,
    )

    def response(entries):
        assert entries[0]["content"] == content
        return claim_plan(entries, "L’entreprise fabrique des instruments de mesure.")

    generator = generator_for(response)
    answer = generator.generate(
        "Quelle est l’activité de l’entreprise contactée ?", [document]
    )
    assert generator.last_trace["status"] == "answer"
    assert "instruments de mesure" in answer
    assert "Source : appel.mp3 — 8.5–111.5 s" in answer
    assert generator.last_trace["omitted_evidence_count"] == 0
    payload = generator.chains[NarrativePlan].payload
    rendered = SYSTEM_PROMPT.format(**payload) + payload["question"]
    assert len(rendered.encode("utf-8")) <= 4096 - 512 - 256


def test_excluded_table_does_not_block_a_ranked_audio_parent(monkeypatch):
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)
    monkeypatch.setattr(settings, "ollama_num_predict", 512)
    document = narrative("Un passage du contrat. " * 110)
    generator = generator_for(claim_plan)
    answer = generator.generate(
        "Quelle est l’activité de l’entreprise contactée ?",
        [document, table_document()],
    )
    entries = [
        json.loads(line)
        for line in generator.chains[NarrativePlan].payload["context"].splitlines()[1:]
    ]
    assert len(entries) == 1
    assert entries[0]["content"] == document.page_content
    assert generator.last_trace["status"] == "answer"
    assert generator.last_trace["omitted_evidence_count"] == 1
    assert "Contexte partiel" in answer


def test_no_evidence_does_not_call_model():
    generator = generator_for(
        lambda _: pytest.fail("No evidence must not call the model.")
    )
    assert "Je ne dispose pas" in generator.generate("Quelle est la réponse ?", [])


def test_incomplete_tables_cannot_answer_global_extrema():
    document = table_document()
    document.metadata["table_complete"] = False
    generator = generator_for(
        lambda entries: query_plan(entries, "min", column="Charge")
    )
    generator.generate("Quel est le minimum ?", [document])
    assert generator.last_trace["status"] == "invalid_plan"


def test_generated_citation_lines_cannot_override_metadata():
    generator = generator_for(
        lambda entries: claim_plan(entries, "Un fait.\nSource : invented.pdf")
    )
    answer = generator.generate("Quel est le fait ?", [narrative()])
    assert "invented.pdf" not in answer
    assert generator.last_trace["status"] == "invalid_plan"


def test_structured_schema_rejects_contradictory_status():
    with pytest.raises(ValueError):
        GenerationPlan.model_validate(
            {
                "status": "insufficient",
                "claims": [{"text": "Un fait.", "references": [{"evidence_id": "E"}]}],
            }
        )


def test_evidence_identity_is_deterministic():
    documents = [narrative(section="Conditions")]
    assert (
        evidence_from_documents(documents)[0].evidence_id
        == evidence_from_documents(documents)[0].evidence_id
    )


def test_arbitrary_factual_text_cannot_bypass_validation_as_clarification():
    generator = generator_for(
        lambda _: {
            "status": "clarify",
            "clarification": "Le prix vaut 999 €. Source : invented.pdf",
        }
    )
    answer = generator.generate("Quel est le prix ?", [narrative()])
    assert "999" not in answer and "invented.pdf" not in answer
    assert generator.last_trace["status"] == "invalid_plan"


def test_omitted_evidence_is_disclosed_in_answer():
    generator = generator_for(claim_plan)
    answer = generator.generate(
        "Quel est le délai ?", [narrative(), narrative("x" * 20000)]
    )
    assert generator.last_trace["omitted_evidence_count"] == 1
    assert "Contexte partiel" in answer


def test_aggregate_trace_is_separate_from_raw_cell_relations():
    generator = generator_for(
        lambda entries: query_plan(entries, "sum", column="Charge")
    )
    answer = generator.generate("Quelle est la somme ?", [table_document()])
    assert "21.00 EUR" in answer
    assert generator.last_trace["actual_values"] is None
    assert generator.last_trace["computations"][0]["aggregate"] == "21.00 EUR"


def test_incomplete_selection_is_explicitly_labeled():
    document = table_document()
    document.metadata["table_complete"] = False
    generator = generator_for(lambda entries: query_plan(entries))
    answer = generator.generate("Quels articles sont visibles ?", [document])
    assert "Tableau incomplet" in answer
    assert generator.last_trace["status"] == "answer"


def test_malformed_legacy_table_is_invalid_evidence_not_context_limit():
    document = table_document()
    document.page_content = "| A | B |\n| --- | --- |\n| one |"
    generator = generator_for(
        lambda _: pytest.fail("Invalid tables cannot reach the planner.")
    )
    generator.generate("Quel est le maximum ?", [document])
    assert generator.last_trace["status"] == "invalid_evidence"


def test_planner_receives_actual_output_schema_not_just_decoder_constraints():
    generator = generator_for(
        lambda entries: query_plan(entries, "min", column="Charge")
    )
    generator.generate("Quel est le minimum ?", [table_document()])
    schema = json.loads(generator.chains[TablePlan].payload["output_schema"])
    assert "column" in schema["properties"]
    assert "min" in schema["properties"]["operation"]["enum"]
    assert (
        generator.last_trace["proposed_plan"]["table_queries"][0]["operation"] == "min"
    )


def test_audio_schema_does_not_consume_budget_with_unavailable_table_tools():
    generator = generator_for(claim_plan)
    generator.generate(
        "Quelle règle de parole ?", [narrative("Le contrat commence lundi. " * 40)]
    )
    example = json.loads(generator.chains[NarrativePlan].payload["output_schema"])
    assert NarrativePlan.model_validate(example).status == "answer"
    assert "table_queries" not in example
    assert generator.last_trace["omitted_evidence_count"] == 0


def test_narrative_plan_cannot_request_a_hidden_table_tool():
    with pytest.raises(ValueError):
        NarrativePlan.model_validate(
            {
                "status": "answer",
                "table_queries": [
                    {"evidence_id": "invented", "operation": "max", "column": "Charge"}
                ],
            }
        )


def test_text_and_table_plans_share_the_same_status_consistency_rules():
    for schema in (NarrativePlan, GenerationPlan):
        with pytest.raises(ValueError):
            schema.model_validate(
                {"status": "insufficient", "clarification": "operation"}
            )


def test_output_language_never_translates_machine_protocol_values():
    instruction = prompt_values("Quel est le maximum ?", "", TablePlan)[
        "answer_language_instruction"
    ]
    assert "for claim text only" in instruction
    assert "status/operation enums" in instruction
    assert "unchanged" in instruction


def test_table_only_evidence_exposes_only_executable_table_answers():
    generator = generator_for(
        lambda entries: query_plan(entries, "max", column="Charge")
    )
    generator.generate("Quel est le maximum ?", [table_document()])
    schema = json.loads(generator.chains[TablePlan].payload["output_schema"])
    assert "operation" in schema["properties"]
    assert "claims" not in schema["properties"]
    assert "AnswerClaim" not in schema["$defs"]


def test_mixed_evidence_keeps_both_narrative_and_table_capabilities():
    generator = generator_for(claim_plan)
    generator.generate("Quand commence le contrat ?", [narrative(), table_document()])
    schema = json.loads(generator.chains[GenerationPlan].payload["output_schema"])
    assert "claims" in schema["properties"]
    assert "table_queries" in schema["properties"]
    assert generator.last_trace["status"] == "answer"


def test_table_planner_can_refuse_without_executing_an_operation():
    generator = generator_for(lambda entries: query_plan(entries, "insufficient"))
    answer = generator.generate("Quel est le numéro de téléphone ?", [table_document()])
    assert "Je ne dispose pas" in answer
    assert generator.last_trace["status"] == "insufficient"
    assert not generator.last_trace.get("computations")


def test_table_plan_decline_cannot_reach_arithmetic_executor():
    plan = TablePlan(evidence_id="E", operation="clarify").as_generation_plan()
    assert plan.status == "clarify"
    assert plan.table_queries == []


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient"])
def test_wrong_entity_claim_is_not_displayed_when_verification_rejects_it(verdict):
    document = narrative(
        "Agence Delta, bonjour. Ici la société Orion, spécialiste des capteurs."
    )
    question = "Quelle est l’activité de l’entreprise contactée ?"
    generator = generator_for(
        lambda entries: claim_plan(
            entries, "L’entreprise contactée fabrique des capteurs."
        ),
        verdict=verdict,
    )
    answer = generator.generate(question, [document])
    assert "capteurs" not in answer
    assert "passages consultés" in answer
    assert generator.last_trace["status"] == "unverified"
    assert generator.last_trace["verification"] == verdict
    assert generator.verifier.payload["question"] == question


def test_verified_claim_uses_only_its_original_cited_evidence():
    cited = narrative("Le vendeur Orion propose des capteurs.", page=3)
    unrelated = narrative("Le client Delta achète des pompes.", source="autre.txt")
    generator = generator_for(
        lambda entries: claim_plan(entries, "Orion propose des capteurs.")
    )
    answer = generator.generate("Que propose Orion ?", [cited, unrelated])
    payload = generator.verifier.payload
    assert generator.last_trace["verification"] == "supported"
    assert "Orion propose des capteurs" in answer
    assert len(payload["evidence"]) == 1
    assert next(iter(payload["evidence"].values()))["content"] == cited.page_content
    assert "page 3" in next(iter(payload["evidence"].values()))["citation"]
    assert set(payload["evidence"]) == {
        payload["claims"][0]["references"][0]["evidence_id"]
    }


def test_table_calculations_do_not_trigger_narrative_verification():
    generator = generator_for(
        lambda entries: query_plan(entries, "max", column="Charge")
    )
    generator.generate("Quel est le maximum ?", [table_document()])
    assert generator.last_trace["status"] == "answer"
    assert generator.verifier.payload is None


def test_invalid_verifier_output_cannot_release_the_draft_answer():
    generator = generator_for(claim_plan, verdict="maybe")
    answer = generator.generate("Quand commence le contrat ?", [narrative()])
    assert "commence lundi" not in answer
    assert generator.last_trace["status"] == "invalid_plan"
    with pytest.raises(ValueError):
        GroundingCheck(verdict="supported", explanation="unvalidated field")


def test_verification_budget_refuses_without_truncating_or_calling_the_model(
    monkeypatch,
):
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)
    monkeypatch.setattr(settings, "ollama_num_predict", 512)
    generator = generator_for(lambda entries: claim_plan(entries, "Un fait. " * 200))
    answer = generator.generate("Quel est le fait ?", [narrative("Un fait. " * 220)])
    assert generator.last_trace["verification"] == "context_limit"
    assert generator.verifier.payload is None
    assert "capacité" in answer
    assert generator.last_trace["status"] == "context_limit"


def test_verifier_budget_uses_its_own_short_output_limit(monkeypatch):
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)
    monkeypatch.setattr(settings, "ollama_num_predict", 512)
    document = narrative("Contexte de la conversation. " * 94, source="appel.mp3")
    evidence = evidence_from_documents([document])
    references = [{"evidence_id": evidence[0].evidence_id}]
    plan = GenerationPlan(
        status="answer",
        claims=[
            {
                "text": "La société Orion appelle l'agence Delta.",
                "references": references,
            },
            {
                "text": "L'agence Delta reçoit cet appel commercial.",
                "references": references,
            },
        ],
    )
    generator = generator_for(claim_plan)
    assert generator._verify_claims("Qui appelle qui ?", plan, evidence) == "supported"
    payload = json.dumps(
        generator.verifier.payload, ensure_ascii=False, separators=(",", ":")
    )
    size = len((VERIFICATION_PROMPT + payload).encode("utf-8"))
    assert 4096 - 512 - 256 < size <= 4096 - VERIFICATION_OUTPUT_TOKENS - 256
