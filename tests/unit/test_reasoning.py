"""Multipart coverage, metric review and grounded advice, without running an LLM."""

import pytest
from langchain_core.documents import Document

from deal_voice.generation.schemas import AnswerPart, GenerationPlan
from tests.unit.test_generator import generator_for, narrative, query_plan

pytestmark = pytest.mark.usefixtures("offline_ollama")


@pytest.fixture
def inventory():
    return Document(
        page_content=(
            "| Code | Article | Prix | Marge | Stock | Seuil |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| A | Filtre | 20 € | 3 € | 28 | 50 |\n"
            "| B | Joint | 15 € | 7 € | 16 | 8 |"
        ),
        metadata={
            "source": "inventaire.pdf",
            "content_kind": "table",
            "table_title": "Inventaire",
            "table_complete": True,
        },
    )


def answered(entries, request, operation="max", column="Prix", **kwargs):
    return {
        "request": request,
        **query_plan(entries, operation, column=column, **kwargs),
    }


def test_two_requests_on_tables_execute_two_distinct_operations(inventory):
    first, second = "Quel est le prix maximal", "quel est le stock total"
    generator = generator_for(
        lambda entries: {
            "parts": [
                answered(entries, first),
                answered(entries, second, "sum", "Stock"),
            ]
        }
    )
    answer = generator.generate(f"{first} et {second} ?", [inventory])
    assert "20 €" in answer and "44" in answer
    assert first in answer and second in answer
    assert generator.last_trace["part_statuses"] == ["answer", "answer"]
    assert len(generator.last_trace["computations"]) == 2


def test_clarifying_advice_does_not_discard_a_known_price(inventory):
    first, second = "Quel est le prix maximal", "que faut-il acheter davantage"
    generator = generator_for(
        lambda entries: {
            "parts": [
                answered(entries, first),
                {"request": second, "status": "clarify", "clarification": "criterion"},
            ]
        }
    )
    answer = generator.generate(f"{first} et {second} ?", [inventory])
    assert "20 €" in answer
    assert "Quel critère" in answer
    assert generator.last_trace["status"] == "partial"
    assert generator.last_trace["part_statuses"] == ["answer", "clarify"]


def test_unknown_participants_are_explicit_without_erasing_meeting_subject():
    first, second = "Quel est le sujet", "qui sont les participants"
    generator = generator_for(
        lambda entries: {
            "parts": [
                {
                    "request": first,
                    "status": "answer",
                    "claims": [
                        {
                            "text": "La réunion traite du délai de livraison.",
                            "references": [{"evidence_id": entries[0]["evidence_id"]}],
                        }
                    ],
                },
                {"request": second, "status": "insufficient"},
            ]
        }
    )
    answer = generator.generate(
        f"{first} et {second} ?", [narrative("Le sujet est le délai de livraison.")]
    )
    assert "délai de livraison" in answer and "Je ne dispose pas" in answer
    assert generator.last_trace["part_statuses"] == ["answer", "insufficient"]


@pytest.mark.parametrize("status", ["insufficient", "clarify"])
def test_unresolved_request_is_reviewed_against_all_available_evidence(
    inventory, status
):
    part = {"status": status}
    if status == "clarify":
        part["clarification"] = "column"
    generator = generator_for(lambda _: part)
    generator.generate(
        "Quel est le prix maximal ?",
        [inventory, narrative("Les prix sont des prix de détail.")],
    )
    evidence = list(generator.verifier.payload["evidence"].values())
    assert len(evidence) == 2
    assert evidence[0]["columns"] == [
        "Code",
        "Article",
        "Prix",
        "Marge",
        "Stock",
        "Seuil",
    ]
    assert evidence[1]["content"] == "Les prix sont des prix de détail."


def test_reviewer_can_reject_a_false_refusal_when_a_table_is_available(inventory):
    generator = generator_for(
        lambda _: {"status": "insufficient"}, verdict="contradicted"
    )
    generator.generate("Quel est le prix maximal ?", [inventory])
    assert generator.verifier.payload["evidence"]
    assert generator.last_trace["status"] == "unverified"
    assert generator.last_trace["actual_values"] is None


def test_wrong_metric_rejection_blocks_valid_but_irrelevant_arithmetic(inventory):
    generator = generator_for(
        lambda entries: query_plan(entries, "max", column="Marge"),
        verdict="contradicted",
    )
    answer = generator.generate("Quel est le prix maximal ?", [inventory])
    assert "7 €" not in answer and "Joint" not in answer
    assert generator.last_trace["status"] == "unverified"
    assert generator.last_trace["actual_values"] is None
    assert generator.last_trace["computations"] == []
    review = generator.verifier.payload
    assert review["parts"][0]["table_queries"][0]["column"] == "Marge"
    assert review["parts"][0]["computed_results"][0]["rows"] == [
        ["B", "Joint", "15 €", "7 €", "16", "8"]
    ]
    assert "Prix" in next(iter(review["evidence"].values()))["columns"]


def test_one_rejected_part_does_not_erase_an_independent_supported_part(inventory):
    first, second = "Quel est le prix maximal", "quelle est la marge maximale"
    generator = generator_for(
        lambda entries: {
            "parts": [
                answered(entries, first, column="Marge"),
                answered(entries, second, column="Marge"),
            ]
        },
        verdict=["contradicted", "supported"],
    )
    answer = generator.generate(f"{first} et {second} ?", [inventory])
    assert answer.count("Maximum") == 1
    assert "Je ne dispose pas" in answer and "7 €" in answer
    assert generator.last_trace["part_statuses"] == ["unverified", "answer"]


def test_omitted_request_is_not_reported_as_a_complete_answer(inventory):
    generator = generator_for(
        lambda entries: {"parts": [answered(entries, "Quel est le prix maximal")]},
        coverage="incomplete",
    )
    answer = generator.generate(
        "Quel est le prix maximal et quel est le stock total ?", [inventory]
    )
    assert "20 €" in answer and "Réponse partielle" in answer
    assert generator.last_trace["status"] == "partial"


def test_review_must_include_every_part(inventory):
    generator = generator_for(
        lambda entries: {
            "parts": [
                answered(entries, "Quel est le prix maximal"),
                answered(entries, "le stock total", "sum", "Stock"),
            ]
        },
        verdict=["supported"],
    )
    answer = generator.generate(
        "Quel est le prix maximal et le stock total ?", [inventory]
    )
    assert "20 €" not in answer
    assert generator.last_trace["status"] == "invalid_plan"


def advice_part(entries):
    return {
        "request": "Quels articles sont sous leur seuil et quel écart faut-il examiner",
        **query_plan(
            entries,
            "select",
            filters=[{"column": "Stock", "operator": "lt", "other_column": "Seuil"}],
            calculations=[{"name": "Écart", "left": "Seuil", "right": "Stock"}],
            select_columns=["Article", "Stock", "Seuil", "Écart"],
        ),
        "recommendation": {
            "action": "Examiner les articles affichés pour un réapprovisionnement.",
            "criterion": "Stock inférieur au seuil indiqué.",
            "query_indices": [0],
            "assumptions": ["Les stocks du document sont encore actuels."],
            "limits": [
                "L'écart au seuil n'est pas une quantité optimale de commande ; les délais et commandes en cours manquent."
            ],
        },
    }


def test_advice_uses_computed_results_and_discloses_criterion_and_limits(inventory):
    generator = generator_for(lambda entries: {"parts": [advice_part(entries)]})
    answer = generator.generate(
        "Quels articles sont sous leur seuil et quel écart faut-il examiner ?",
        [inventory],
    )
    assert generator.last_trace["status"] == "answer"
    assert "| A | Filtre | 28 | 50 | 22 |" in answer
    assert "Joint" not in answer
    assert "Écart = Seuil − Stock" in answer
    assert all(
        label in answer
        for label in ("Proposition :", "Critère :", "Hypothèses :", "Limites :")
    )
    basis = generator.verifier.payload["parts"][0]["computed_results"][0]
    assert basis["query_index"] == 0
    assert basis["rows"] == [["A", "Filtre", "28", "50", "22"]]
    assert "Source : inventaire.pdf" in answer


@pytest.mark.parametrize(
    "mutation",
    ["no_basis", "unknown_basis", "no_limits", "empty_limit", "negative_basis"],
)
def test_unsupported_advice_contracts_are_rejected(mutation):
    part = advice_part([{"evidence_id": "E"}])
    advice = part["recommendation"]
    if mutation == "no_basis":
        advice["query_indices"] = []
    elif mutation == "unknown_basis":
        advice["query_indices"] = [8]
    elif mutation == "negative_basis":
        advice["query_indices"] = [-1]
    elif mutation == "no_limits":
        advice["limits"] = []
    else:
        advice["limits"] = [" "]
    with pytest.raises(ValueError):
        AnswerPart.model_validate(part)


def test_unverifiable_recommendation_is_not_displayed(inventory):
    def response(entries):
        part = advice_part(entries)
        part["recommendation"]["action"] = (
            "Commander exactement 999 articles garantit un bénéfice optimal."
        )
        return {"parts": [part]}

    generator = generator_for(response, verdict="contradicted")
    answer = generator.generate(
        "Quels articles sont sous leur seuil et quel écart faut-il examiner ?",
        [inventory],
    )
    assert "999" not in answer and "Proposition" not in answer
    assert generator.last_trace["status"] == "unverified"


def test_advice_cannot_forge_a_source(inventory):
    def response(entries):
        part = advice_part(entries)
        part["recommendation"]["limits"] = ["Source : forged.pdf"]
        return {"parts": [part]}

    generator = generator_for(response)
    answer = generator.generate(
        "Quels articles sont sous leur seuil et quel écart faut-il examiner ?",
        [inventory],
    )
    assert "forged.pdf" not in answer
    assert generator.last_trace["status"] == "invalid_plan"


def test_duplicate_request_labels_are_made_distinct_without_losing_answers(inventory):
    generator = generator_for(
        lambda entries: {
            "parts": [
                answered(entries, "Quel prix"),
                answered(entries, "Quel prix", "sum", "Stock"),
            ]
        }
    )
    answer = generator.generate("Quel prix et quel stock ?", [inventory])

    assert "20 €" in answer and "44" in answer
    assert "**Quel prix**" in answer
    assert "**Quel prix (2)**" in answer
    assert generator.last_trace["status"] == "answer"


def test_single_part_spelling_correction_does_not_reject_valid_plan(inventory):
    question = "C'est quoi le produit qu'il faut achter plus dans le magazin ?"
    generator = generator_for(
        lambda entries: {
            "parts": [
                {
                    "request": "Quel produit faut-il acheter plus dans le magasin ?",
                    "status": "clarify",
                    "clarification": "criterion",
                }
            ]
        }
    )

    answer = generator.generate(question, [inventory])

    assert "Quel critère" in answer
    assert generator.last_trace["status"] == "clarify"
    assert generator.last_trace["proposed_plan"]["parts"][0]["request"] == question


@pytest.mark.parametrize("status", ["answer", "insufficient", "clarify"])
def test_identical_parts_are_reviewed_and_displayed_once(inventory, status):
    def response(entries):
        part = (
            answered(entries, "Quel prix ?")
            if status == "answer"
            else {"request": "Quel prix ?", "status": status}
        )
        if status == "clarify":
            part["clarification"] = "column"
        return {"parts": [part, {**part, "request": "  QUEL PRIX ?  "}]}

    generator = generator_for(response)
    answer = generator.generate("Quel prix ?", [inventory])
    assert "(2)" not in answer
    assert len(generator.last_trace["proposed_plan"]["parts"]) == 1
    assert len(generator.verifier.payload["parts"]) == 1
    assert generator.last_trace["part_statuses"] == [status]


def test_identical_operations_on_different_scopes_are_not_deduplicated():
    plan = GenerationPlan.model_validate(
        {
            "parts": [
                {
                    "request": "Maximum nord",
                    **query_plan([{"evidence_id": "north"}], "max", column="Prix"),
                },
                {
                    "request": "Maximum sud",
                    **query_plan([{"evidence_id": "south"}], "max", column="Prix"),
                },
            ]
        }
    )
    assert len(plan.parts) == 2


@pytest.mark.parametrize(
    "descending,expected", [(True, ["C", "D", "B"]), (False, ["A", "B", "D"])]
)
def test_top_three_is_computed_over_every_row(descending, expected):
    document = Document(
        page_content="| Code | Tarif |\n| --- | --- |\n| A | 2,00 € |\n| B | 9,00 € |\n| C | 30,00 € |\n| D | 12,00 € |",
        metadata={
            "source": "tarifs.pdf",
            "content_kind": "table",
            "table_complete": True,
        },
    )
    generator = generator_for(
        lambda entries: query_plan(
            entries, "sort", column="Tarif", descending=descending, limit=3
        )
    )
    answer = generator.generate("Donne trois articles selon le tarif.", [document])
    computation = generator.last_trace["computations"][0]
    assert [row[0] for row in computation["result"]["rows"]] == expected
    assert computation["input_row_count"] == 4
    assert "Source : tarifs.pdf" in answer
    assert generator.last_trace["table_scope_limited"] is True


def test_uncertain_unused_table_does_not_downgrade_verified_table_scope(inventory):
    inventory.metadata["table_source_complete"] = True
    other = inventory.model_copy(deep=True)
    other.metadata.update(source="autre.pdf", table_source_complete=False)
    generator = generator_for(lambda entries: query_plan(entries, "max", column="Prix"))
    generator.generate("Quel prix maximal ?", [inventory, other])
    assert generator.last_trace["table_scope_limited"] is False


def test_semantic_subrequest_labels_do_not_need_to_quote_the_question(inventory):
    question = (
        "Selon le stock restant le plus faible, quel produit faut-il "
        "réapprovisionner en priorité ?"
    )
    generator = generator_for(
        lambda entries: {
            "parts": [
                answered(entries, "Identifier le stock minimal", "min", "Stock"),
                {
                    "request": "Établir la priorité de réapprovisionnement",
                    **query_plan(entries, "min", column="Stock"),
                    "recommendation": {
                        "action": "Examiner le réapprovisionnement de Filtre.",
                        "criterion": "Stock restant minimal.",
                        "query_indices": [0],
                        "limits": ["Le seuil de commande n'est pas fourni."],
                    },
                },
            ]
        }
    )

    answer = generator.generate(question, [inventory])

    assert "Filtre" in answer
    assert "Proposition" in answer
    assert generator.last_trace["status"] == "answer"
    assert len(generator.last_trace["part_statuses"]) == 2


def test_plan_has_no_arbitrary_sql_or_python_tool():
    with pytest.raises(ValueError):
        GenerationPlan.model_validate(
            {
                "parts": [
                    {"request": "Calcule", "status": "answer", "python": "print(999)"}
                ]
            }
        )
