import pytest

from deal_voice.evaluation.generation import evaluate_generation


def test_generation_evaluation_checks_source_facts_and_language():
    result = evaluate_generation(
        "q1",
        "Quel produit ?",
        "call.mp3",
        ["fontaines", "bouteilles"],
        "Le produit concerne les fontaines et les bouteilles.\nSource : call.mp3",
    )
    assert result.source_present is True
    assert result.fact_recall == 1.0
    assert result.likely_french is True
    assert result.passed is True


def test_generation_evaluation_accepts_markdown_bold_source_label():
    result = evaluate_generation(
        "q-markdown-source",
        "Quel produit ?",
        "call.mp3",
        ["fontaines"],
        "Le produit concerne les fontaines.\n**Source** : call.mp3",
    )

    assert result.source_present is True
    assert result.citation_file_recall == 1.0
    assert result.passed is True


def test_generation_evaluation_rejects_english_answer_for_french_question():
    result = evaluate_generation(
        "q-language",
        "Combien de collaborateurs compte l'agence ?",
        "call.mp3",
        ["25"],
        "The agency has 25 employees.\nSource : call.mp3",
        expected_language="fr",
    )

    assert result.language_compliant is False
    assert result.passed is False


@pytest.mark.parametrize(
    ("answer", "is_french"),
    [
        ("L’agence compte 25 collaborateurs.", True),
        ("L'agence compte 25 collaborateurs.", True),
        ("25 collaborateurs travaillent au sein d’une agence.", True),
        ("25 collaborateurs travaillent au sein d'une agence.", True),
        (
            "The agency has 25 employees.\n"
            "Source : call.mp3 — section « Le suivi de l’agence »",
            False,
        ),
    ],
)
def test_generation_language_checks_answer_text_including_elisions(answer, is_french):
    result = evaluate_generation(
        "q-language-elision",
        "Combien de collaborateurs compte l’agence ?",
        "call.mp3",
        ["25"],
        answer + "\nSource : call.mp3",
        expected_language="fr",
    )

    assert result.likely_french is is_french
    assert result.language_compliant is is_french
    assert result.passed is is_french


def test_generation_evaluation_accepts_controlled_paraphrases():
    result = evaluate_generation(
        "q2",
        "Quel objectif ?",
        "call.mp3",
        ["fontaines a eau", "proposition"],
        "Le client propose un systeme de fontaines d'eau.\nSource : call.mp3",
        {
            "fontaines a eau": ["fontaines d'eau", "systeme de fontaines"],
            "proposition": ["propose", "proposer"],
        },
    )
    assert result.lexical_fact_recall == 0.0
    assert result.fact_recall == 1.0
    assert result.passed is True


@pytest.mark.parametrize(
    ("expected_fact", "variant", "answer"),
    [
        ("lendemain", "demain", "La proposition sera reçue dès demain par courrier."),
        ("17 h", "17 heures", "Le rendez-vous est fixé à 17 heures."),
        (
            "paquet de gâteaux",
            "paquet de gâteau",
            "Un paquet de gâteau a été rapporté.",
        ),
        (
            "produits disponibles",
            "produit disponible",
            "La liste présente chaque produit disponible.",
        ),
    ],
)
def test_generation_evaluation_accepts_declared_french_fact_variants(
    expected_fact, variant, answer
):
    result = evaluate_generation(
        "q-equivalence",
        "Que dit le document ?",
        "call.mp3",
        [expected_fact],
        answer + "\nSource : call.mp3",
        expected_fact_variants={expected_fact: [variant]},
    )

    assert result.facts_missing == []
    assert result.fact_recall == 1.0


def test_generation_evaluation_does_not_infer_undeclared_semantic_equivalence():
    result = evaluate_generation(
        "no-hidden-aliases",
        "Quand est le rendez-vous ?",
        "call.mp3",
        ["lendemain"],
        "Le rendez-vous est prévu demain.\nSource : call.mp3",
    )

    assert result.facts_missing == ["lendemain"]


@pytest.mark.parametrize(
    ("expected_fact", "answer"),
    [
        ("17 h", "Le document contient le nombre 17."),
        ("lendemain", "La proposition a été envoyée hier."),
        ("paquet de gâteaux", "Un paquet de café a été rapporté."),
        ("Éric Souplin", "L’appelant est Éric Souplein."),
    ],
)
def test_generation_evaluation_rejects_non_equivalent_facts(expected_fact, answer):
    result = evaluate_generation(
        "q-not-equivalent",
        "Que dit le document ?",
        "call.mp3",
        [expected_fact],
        answer + "\nSource : call.mp3",
    )

    assert result.facts_found == []
    assert result.fact_recall == 0.0


def test_generation_evaluation_accepts_grounded_refusal():
    result = evaluate_generation(
        "q3",
        "Quel est le chiffre d'affaires annuel ?",
        "call.mp3",
        [],
        "Je ne dispose pas de suffisamment d'informations autorisees dans la base de connaissances pour repondre a cette question.",
        expected_behavior="refusal",
    )
    assert result.refusal_detected is True
    assert result.passed is True


def test_generation_evaluation_flags_forbidden_claims():
    result = evaluate_generation(
        "q4",
        "Quel est le sujet ?",
        "call.mp3",
        ["fontaines"],
        "Le produit concerne les fontaines et les bouteilles. Il offre des garanties a vie.\nSource : call.mp3",
        forbidden_claims=["garanties a vie"],
    )
    assert result.forbidden_claims_found == ["garanties a vie"]
    assert result.passed is False


def test_generation_evaluation_does_not_match_a_fact_inside_another_word():
    result = evaluate_generation(
        "q5",
        "Que dit le document ?",
        "",
        ["art"],
        "La partie est decrite dans le document.",
    )
    assert result.facts_found == []
    assert result.facts_missing == ["art"]


@pytest.mark.parametrize("reverse", [False, True])
def test_citation_details_consider_every_reference_to_the_same_file(reverse):
    references = [
        "Source : rapport.pdf",
        "Source : rapport.pdf — table « ## 3. Catalogue » — page 3",
    ]
    if reverse:
        references.reverse()
    result = evaluate_generation(
        "multiple-citations",
        "Quels produits ?",
        "rapport.pdf",
        ["article"],
        "Un article est disponible.\n" + "\n".join(references),
        expected_citations=[{"file": "rapport.pdf", "table": "Catalogue", "page": 3}],
    )

    assert result.citation_detail_recall == 1.0
    assert result.invalid_citation_details == []
    assert result.passed


def test_several_expected_sections_in_one_pdf_are_not_mutually_invalid():
    result = evaluate_generation(
        "two-sections",
        "Que dit le rapport ?",
        "rapport.pdf",
        ["achats", "ventes"],
        "Le rapport décrit les achats et les ventes.\n"
        "Source : rapport.pdf — section « 2. Achats »\n"
        "Source : rapport.pdf — section « 5. Ventes »",
        expected_citations=[
            {"file": "rapport.pdf", "section": "Achats"},
            {"file": "rapport.pdf", "section": "Ventes"},
        ],
    )

    assert result.citation_detail_recall == 1.0
    assert result.invalid_citation_details == []
    assert result.passed


def test_table_title_cited_as_section_is_a_format_warning():
    result = evaluate_generation(
        "table-label",
        "Quel produit ?",
        "rapport.pdf",
        ["article"],
        "Un article est disponible.\nSource : rapport.pdf — section « 3. Catalogue »",
        expected_citations=[{"file": "rapport.pdf", "table": "Catalogue"}],
    )

    assert result.citation_detail_recall == 1.0
    assert result.invalid_citation_details == []
    assert result.citation_format_warnings
    assert result.passed


@pytest.mark.parametrize(
    "extra_reference",
    [
        "Source : rapport.pdf — section « Section inventée »",
        "Source : rapport.pdf — table « Catalogue » — page 8",
    ],
)
def test_valid_reference_does_not_hide_an_invalid_extra_reference(extra_reference):
    result = evaluate_generation(
        "invalid-extra-citation",
        "Quel produit ?",
        "rapport.pdf",
        ["article"],
        "Un article est disponible.\n"
        "Source : rapport.pdf — table « Catalogue » — page 3\n" + extra_reference,
        expected_citations=[{"file": "rapport.pdf", "table": "Catalogue", "page": 3}],
    )

    assert result.citation_detail_recall == 1.0
    assert result.invalid_citation_details
    assert not result.passed


def test_additional_citation_can_be_proven_by_available_evidence():
    result = evaluate_generation(
        "available-evidence",
        "Que dit le rapport ?",
        "rapport.pdf",
        ["article"],
        "Un article est disponible.\n"
        "Source : rapport.pdf — table « Catalogue »\n"
        "Source : rapport.pdf — section « Méthode »",
        expected_citations=[{"file": "rapport.pdf", "table": "Catalogue"}],
        available_citations=[
            {"file": "rapport.pdf", "table": "Catalogue"},
            {"file": "rapport.pdf", "section": "Méthode"},
        ],
    )

    assert result.invalid_citation_details == []
    assert result.passed


def test_structured_values_reject_swapped_entity_value_relations():
    result = evaluate_generation(
        "swapped-values",
        "Quels sont les montants ?",
        "rapport.pdf",
        [],
        "Les montants sont Alpha : 20 € ; Beta : 10 €.\nSource : rapport.pdf",
        expected_values=[
            {"entity": "Alpha", "field": "Montant", "value": "10 €"},
            {"entity": "Beta", "field": "Montant", "value": "20 €"},
        ],
        actual_values=[
            {"entity": "Alpha", "field": "Montant", "value": "20 €"},
            {"entity": "Beta", "field": "Montant", "value": "10 €"},
        ],
    )

    assert result.value_recall == 1.0  # Lexical presence alone cannot prove a relation.
    assert len(result.value_relations_missing) == 2
    assert not result.passed


@pytest.mark.parametrize("wrong_value", ["1", "-10"])
def test_relation_value_comparison_preserves_numeric_sign_and_magnitude(wrong_value):
    result = evaluate_generation(
        "numeric-relation",
        "Quel solde ?",
        "rapport.pdf",
        [],
        "Le solde est -1.\nSource : rapport.pdf",
        expected_values=[{"entity": "Compte", "field": "Solde", "value": "-1"}],
        actual_values=[{"entity": "Compte", "field": "Solde", "value": wrong_value}],
    )

    assert result.value_relations_missing
    assert not result.passed


def test_legacy_text_value_presence_does_not_claim_verified_relations():
    result = evaluate_generation(
        "unverified-relations",
        "Quel montant ?",
        "rapport.pdf",
        [],
        "Le montant est 10 €.\nSource : rapport.pdf",
        expected_values=[{"entity": "Article A", "field": "Montant", "value": "10 €"}],
    )

    assert result.value_recall == 1.0
    assert result.value_relations_found == []
    assert result.value_relations_missing == []
    assert result.value_relations_not_evaluable == ["Article A / Montant / 10 €"]


def test_missing_locator_in_available_metadata_does_not_prove_invented_locator():
    result = evaluate_generation(
        "unavailable-locator",
        "Quel produit ?",
        "rapport.pdf",
        ["article"],
        "Un article est disponible.\n"
        "Source : rapport.pdf — section « Section inventée » — page 8",
        expected_citations=[{"file": "rapport.pdf"}],
        available_citations=[{"file": "rapport.pdf"}],
    )

    assert len(result.invalid_citation_details) == 2
    assert not result.passed


@pytest.mark.parametrize(
    "locator", ["0.904–59.788 s", "1-12 s", "time 0.5–2.0 s", "horodatage 2–3 s"]
)
def test_audio_citation_with_timestamps_matches_transcript_reference(locator):
    result = evaluate_generation(
        "audio-time",
        "Quelle règle ?",
        "Reunion-dequipe.txt",
        ["parole"],
        f"La parole a le même poids.\nSource : Reunion dequipe.mp3 — {locator}",
    )
    assert result.citation_file_recall == 1.0
    assert result.passed


def test_hyphenated_filename_is_not_mistaken_for_a_citation_locator():
    result = evaluate_generation(
        "filename-dash",
        "Quel fait ?",
        "meeting — annual-review.mp3",
        ["parole"],
        "La parole est libre.\nSource : meeting — annual-review.mp3",
    )
    assert result.passed


def test_timestamp_does_not_make_a_wrong_audio_source_match():
    result = evaluate_generation(
        "wrong-audio",
        "Quel fait ?",
        "meeting.txt",
        ["parole"],
        "La parole est libre.\nSource : unrelated.mp3 — 1.2–3.4 s",
    )
    assert not result.source_present
    assert not result.passed
