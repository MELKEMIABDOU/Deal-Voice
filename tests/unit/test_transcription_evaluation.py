from deal_voice.evaluation.transcription import (
    _count_repeated_phrases,
    error_rate,
    result_to_markdown,
)


def test_repeated_phrase_metric_counts_adjacent_repetitions():
    assert _count_repeated_phrases("bonjour bonjour bonjour") >= 1
    assert _count_repeated_phrases("bonjour comment allez-vous") == 0


def test_error_rates_are_zero_for_identical_text():
    assert error_rate("Bonjour le monde.", "bonjour le monde") == 0.0
    assert error_rate("Bonjour le monde.", "bonjour le monde", character=True) == 0.0


def test_markdown_report_contains_metrics():
    class Result:
        model = "base"
        duration_seconds = 10.0
        transcription_seconds = 20.0
        real_time_factor = 2.0
        segment_count = 3
        word_count = 12
        repeated_phrase_count = 1

    report = result_to_markdown([Result()])
    assert "| base | 10.0 | 20.0 | 2.00 | n/a | n/a | 3 | 12 | 1 |" in report
