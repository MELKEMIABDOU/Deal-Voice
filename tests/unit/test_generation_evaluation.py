from deal_voice.evaluation.generation import evaluate_generation


def test_generation_evaluation_checks_source_facts_and_language():
    result = evaluate_generation(
        "q1",
        "Quel produit ?",
        "call.mp3",
        ["fontaines", "bouteilles"],
        "La source est call.mp3. Le produit concerne les fontaines et les bouteilles.",
    )

    assert result.source_present is True
    assert result.fact_recall == 1.0
    assert result.likely_french is True
    assert result.passed is True


def test_generation_evaluation_accepts_controlled_paraphrases():
    result = evaluate_generation(
        "q2",
        "Quel objectif ?",
        "call.mp3",
        ["fontaines à eau", "proposition"],
        "La source est call.mp3. Le client propose un système de fontaines d'eau.",
        {
            "fontaines à eau": ["fontaines d'eau", "système de fontaines"],
            "proposition": ["propose", "proposer"],
        },
    )

    assert result.lexical_fact_recall == 0.0
    assert result.fact_recall == 1.0
    assert result.passed is True


def test_generation_evaluation_accepts_grounded_refusal():
    result = evaluate_generation(
        "q3",
        "Quel est le chiffre d'affaires annuel ?",
        "call.mp3",
        [],
        "Je ne dispose pas de suffisamment d'informations autorisées "
        "dans la base de connaissances pour répondre à cette question.",
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
        "La source est call.mp3. Les fontaines sont garanties à vie.",
        forbidden_claims=["garanties à vie"],
    )

    assert result.unsupported_claims_found == ["garanties à vie"]
    assert result.passed is False
