"""Generation contracts, without a running LLM or model downloads."""

import json

import pytest
from langchain_core.documents import Document

from deal_voice.config import settings
from deal_voice.evidence.citations import evidence_from_documents
from deal_voice.generation.generator import RAGGenerator
from deal_voice.generation.prompts import (
    SYSTEM_PROMPT,
    prompt_values,
)
from deal_voice.generation.schemas import (
    AnswerPart,
    GenerationPlan,
    GroundingCheck,
    PlannedTableQuery,
)

pytestmark = pytest.mark.usefixtures("offline_ollama")


class FakeChain:
    def __init__(self, response):
        self.payload = None
        self.response = response

    def invoke(self, payload):
        self.payload = payload
        entries = [json.loads(line) for line in payload["context"].splitlines()[1:]]
        result = self.response(entries)
        if "parts" not in result:
            result = {"parts": [{"request": payload["question"], **result}]}
        return result


class FakeVerifier:
    def __init__(self, verdict="supported", coverage="complete"):
        self.verdict = verdict
        self.coverage = coverage
        self.payload = None

    def invoke(self, messages):
        self.payload = json.loads(messages[-1][1])
        verdicts = (
            self.verdict
            if isinstance(self.verdict, list)
            else [self.verdict] * len(self.payload["parts"])
        )
        return {"coverage": self.coverage, "verdicts": verdicts}


def generator_for(response, *, verdict="supported", coverage="complete"):
    generator = RAGGenerator()
    generator.chain = FakeChain(response)
    generator.verifier = FakeVerifier(verdict, coverage)
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
    return {
        "status": "answer",
        "table_queries": [
            {"evidence_id": entries[0]["evidence_id"], "operation": operation, **kwargs}
        ],
    }


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
    assert generator.chain.payload["answer_language_instruction"].startswith(
        f"OUTPUT LANGUAGE: {language}."
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
    generator.chain = FakeChain(
        lambda _: {"status": "clarify", "clarification": "column"}
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
    payload = generator.chain.payload
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
        json.loads(line) for line in generator.chain.payload["context"].splitlines()[1:]
    ]
    assert len(entries) == 1
    assert entries[0]["content"] == document.page_content
    assert generator.last_trace["status"] == "answer"
    assert generator.last_trace["omitted_evidence_count"] == 1
    assert "Contexte partiel" not in answer


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
    generator = generator_for(claim_plan, verdict="insufficient")
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


def test_output_language_never_translates_machine_protocol_values():
    instruction = prompt_values("Quel est le maximum ?", "", True)[
        "answer_language_instruction"
    ]
    assert "for human-readable text only" in instruction
    assert "status/operation enums" in instruction
    assert "unchanged" in instruction


def test_table_planner_can_refuse_without_executing_an_operation():
    generator = generator_for(lambda entries: {"status": "insufficient"})
    answer = generator.generate("Quel est le numéro de téléphone ?", [table_document()])
    assert "Je ne dispose pas" in answer
    assert generator.last_trace["status"] == "insufficient"
    assert not generator.last_trace.get("computations")


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
    assert generator.last_trace["verification"]["verdicts"] == [verdict]
    assert generator.verifier.payload["question"] == question


def test_verified_claim_uses_only_its_original_cited_evidence():
    cited = narrative("Le vendeur Orion propose des capteurs.", page=3)
    unrelated = narrative("Le client Delta achète des pompes.", source="autre.txt")
    generator = generator_for(
        lambda entries: claim_plan(entries, "Orion propose des capteurs.")
    )
    answer = generator.generate("Que propose Orion ?", [cited, unrelated])
    payload = generator.verifier.payload
    assert generator.last_trace["verification"]["verdicts"] == ["supported"]
    assert "Orion propose des capteurs" in answer
    assert len(payload["evidence"]) == 1
    assert next(iter(payload["evidence"].values()))["content"] == cited.page_content
    assert "page 3" in next(iter(payload["evidence"].values()))["citation"]
    assert set(payload["evidence"]) == {
        payload["parts"][0]["claims"][0]["references"][0]["evidence_id"]
    }


def test_table_query_meaning_is_checked_before_display():
    generator = generator_for(
        lambda entries: query_plan(entries, "max", column="Charge")
    )
    generator.generate("Quel est le maximum ?", [table_document()])
    assert generator.last_trace["status"] == "answer"
    assert (
        generator.verifier.payload["parts"][0]["table_queries"][0]["column"] == "Charge"
    )


def test_invalid_verifier_output_cannot_release_the_draft_answer():
    generator = generator_for(claim_plan, verdict="maybe")
    answer = generator.generate("Quand commence le contrat ?", [narrative()])
    assert "commence lundi" not in answer
    assert generator.last_trace["status"] == "invalid_plan"
    with pytest.raises(ValueError):
        GroundingCheck(
            coverage="complete", verdicts=["supported"], explanation="unvalidated field"
        )


def test_single_part_claim_verdicts_are_combined_conservatively():
    generator = generator_for(
        claim_plan,
        verdict=["supported", "supported"],
    )
    answer = generator.generate("Quand commence le contrat ?", [narrative()])
    assert "commence lundi" in answer
    assert generator.last_trace["verification"]["verdicts"] == ["supported"]

    generator = generator_for(
        claim_plan,
        verdict=["supported", "insufficient"],
    )
    answer = generator.generate("Quand commence le contrat ?", [narrative()])
    assert "commence lundi" not in answer
    assert generator.last_trace["verification"]["verdicts"] == ["insufficient"]


def test_verification_budget_refuses_without_truncating_or_calling_the_model(
    monkeypatch,
):
    monkeypatch.setattr(settings, "ollama_num_ctx", 4096)
    monkeypatch.setattr(settings, "ollama_num_predict", 512)
    generator = generator_for(lambda entries: claim_plan(entries, "Un fait. " * 120))
    answer = generator.generate("Quel est le fait ?", [narrative("Un fait. " * 220)])
    assert generator.last_trace["verification"] == "context_limit"
    assert generator.verifier.payload is None
    assert "capacité" in answer
    assert generator.last_trace["status"] == "context_limit"


def test_decoder_schema_supports_multiple_parts_and_tools_without_locale_override():
    schema = GenerationPlan.model_json_schema()
    assert schema["properties"]["parts"]["maxItems"] == 6
    assert (
        PlannedTableQuery.model_json_schema()["properties"]["decimal_separator"]["type"]
        == "null"
    )
    for separator in (",", "."):
        with pytest.raises(ValueError):
            PlannedTableQuery(
                evidence_id="E", operation="max", decimal_separator=separator
            )


@pytest.mark.parametrize("status", ["insufficient", "clarify"])
def test_unresolved_parts_cannot_contain_facts(status):
    with pytest.raises(ValueError):
        AnswerPart(
            request="Quand commence le contrat ?",
            status=status,
            clarification="scope" if status == "clarify" else None,
            claims=[{"text": "lundi", "references": [{"evidence_id": "E"}]}],
        )


def test_prompt_shows_a_valid_parts_example_and_only_available_table_tools():
    for has_tables in (False, True):
        values = prompt_values("Quelle valeur ?", "", has_tables)
        example = json.loads(values["output_schema"])
        assert GenerationPlan.model_validate(example).parts[0].status == "answer"
        assert ("calculations=" in values["planner_instruction"]) is has_tables


def test_explicit_table_decision_criterion_is_not_treated_as_ambiguous_advice():
    instruction = prompt_values(
        "Selon le stock restant le plus faible, quel produit faut-il réapprovisionner ?",
        "",
        True,
    )["planner_instruction"]

    assert "exactly one part when the question has one request" in instruction
    assert "never duplicate a request" in instruction
    assert "explicitly supplies a decision criterion" in instruction
    assert (
        "needs neither a clarification nor a separate recommendation part"
        in instruction
    )
