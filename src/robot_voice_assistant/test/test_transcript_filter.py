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


def test_babble_of_one_repeated_word_is_rejected() -> None:
    assert not is_meaningful_transcript("Ah, ah, ah, ah, ah, ah, ah, ah, ah.", 0.1)
    assert not is_meaningful_transcript("Whoa... Whoa... Whoa... Whoa...", 0.1)


def test_a_sentence_that_repeats_a_word_on_purpose_is_kept() -> None:
    assert is_meaningful_transcript("Walk, walk, walk forward please.", 0.1)


def test_babble_of_a_few_alternating_words_is_rejected() -> None:
    assert not is_meaningful_transcript("Uh-huh, uh-huh, uh-huh, uh-huh, uh-huh, uh-huh, uh-huh.", 0.1)


def test_a_normal_long_sentence_is_kept() -> None:
    assert is_meaningful_transcript("Tell me about the company and what the company does for the people.", 0.1)
