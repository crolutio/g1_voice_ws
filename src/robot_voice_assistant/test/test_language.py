from robot_voice_assistant.language import ARABIC, ENGLISH, choose_language, language_of


def test_english_and_arabic_sentences_are_told_apart() -> None:
    assert language_of("What is in my hand?") == ENGLISH
    assert language_of("ماذا يوجد في يدي؟") == ARABIC


def test_an_arabic_sentence_with_an_english_name_is_still_arabic() -> None:
    assert language_of("ما هي شركة Aideology؟") == ARABIC


def test_an_english_sentence_with_one_arabic_word_is_still_english() -> None:
    assert language_of("How do you say marhaba, مرحبا, in English?") == ENGLISH


def test_text_without_letters_counts_as_english() -> None:
    assert language_of("") == ENGLISH
    assert language_of("42!") == ENGLISH


def test_the_most_likely_allowed_language_is_chosen() -> None:
    probabilities = [("fa", 0.5), ("ar", 0.3), ("en", 0.2)]  # Whisper thinks Persian; only English and Arabic allowed
    assert choose_language(probabilities, allowed=("en", "ar")) == "ar"


def test_the_first_allowed_language_is_the_default_when_whisper_gives_none_of_them() -> None:
    assert choose_language([("fr", 0.9)], allowed=("en", "ar")) == "en"
    assert choose_language([], allowed=("en", "ar")) == "en"
