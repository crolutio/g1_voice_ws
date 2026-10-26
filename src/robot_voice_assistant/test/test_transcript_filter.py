from robot_voice_assistant.transcript_filter import is_meaningful_transcript, normalize


def test_normalize_strips_punctuation_and_case() -> None:
    assert normalize("  Hello, ROBOT!! ") == "hello robot"


def test_real_sentence_is_kept() -> None:
    assert is_meaningful_transcript("Can you wave at me?", 0.05)


def test_known_hallucination_is_rejected() -> None:
    assert not is_meaningful_transcript(" Thank you.", 0.1)


def test_high_no_speech_probability_is_rejected() -> None:
    assert not is_meaningful_transcript("What is your name?", 0.9)


def test_punctuation_only_is_rejected() -> None:
    assert not is_meaningful_transcript(" ... ", 0.0)
