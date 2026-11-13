from robot_voice_assistant.wake_word import GREETING_FOR_BARE_WAKE_WORD, WakeWordGate

WAKE_WORDS = ("robot", "g1", "g one")


def _gate(**options: float) -> WakeWordGate:
    return WakeWordGate(WAKE_WORDS, **options)


def test_sentence_without_wake_word_is_rejected() -> None:
    gate = _gate()
    assert gate.accept("so then she said she would call later", now_s=0.0) is None
    assert not gate.in_conversation


def test_sentence_with_wake_word_is_accepted_and_stripped() -> None:
    assert _gate().accept("Hey robot, can you wave?", now_s=0.0) == "can you wave?"


def test_multi_word_wake_word_matches() -> None:
    assert _gate().accept("G one, what is your name?", now_s=0.0) == "what is your name?"


def test_wake_word_must_be_a_whole_word() -> None:
    assert _gate().accept("the robotics lab is closed", now_s=0.0) is None


def test_empty_wake_word_list_accepts_everything() -> None:
    gate = WakeWordGate(())
    assert gate.accept("anything at all", now_s=0.0) == "anything at all"


def test_wake_word_starts_a_conversation_that_needs_no_wake_word() -> None:
    gate = _gate()
    assert gate.accept("Robot, what is the capital of India?", now_s=0.0) == "what is the capital of India?"
    assert gate.in_conversation
    assert gate.accept("And what about France?", now_s=30.0) == "And what about France?"
    assert gate.accept("Walk forward.", now_s=600.0) == "Walk forward."  # no time limit by default


def test_saying_the_wake_word_inside_a_conversation_is_fine() -> None:
    gate = _gate()
    gate.accept("Robot, hello", now_s=0.0)
    assert gate.accept("Robot, turn left", now_s=5.0) == "turn left"


def test_goodbye_ends_the_conversation_and_is_still_answered() -> None:
    gate = _gate()
    gate.accept("Robot, hi there", now_s=0.0)
    assert gate.accept("Okay, goodbye!", now_s=10.0) == "Okay, goodbye!"
    assert not gate.in_conversation
    assert gate.accept("what time is it?", now_s=20.0) is None  # needs the wake word again


def test_other_ways_to_end_the_conversation() -> None:
    for phrase in ("Stop listening.", "That's all, thank you.", "Go to sleep.", "End the conversation.", "Bye robot!"):
        gate = _gate()
        gate.accept("Robot, hello", now_s=0.0)
        gate.accept(phrase, now_s=1.0)
        assert not gate.in_conversation, phrase


def test_stop_alone_stops_walking_but_keeps_the_conversation() -> None:
    gate = _gate()
    gate.accept("Robot, walk forward", now_s=0.0)
    assert gate.accept("Stop!", now_s=2.0) == "Stop!"
    assert gate.in_conversation


def test_a_long_sentence_that_mentions_goodbye_does_not_end_the_conversation() -> None:
    gate = _gate()
    gate.accept("Robot, hello", now_s=0.0)
    gate.accept("How do you say goodbye in French and in Spanish?", now_s=1.0)
    assert gate.in_conversation


def test_optional_idle_timeout_ends_a_quiet_conversation() -> None:
    gate = _gate(idle_timeout_s=60.0)
    gate.accept("Robot, hello", now_s=0.0)
    gate.note_reply_finished(now_s=5.0)
    assert gate.accept("still there?", now_s=50.0) == "still there?"
    assert gate.accept("anyone?", now_s=200.0) is None
    assert not gate.in_conversation


def test_bare_wake_word_waits_for_the_question_that_follows() -> None:
    gate = _gate(wake_word_wait_s=3.0)
    assert gate.accept("Robot.", now_s=10.0) is None  # held: the question may follow after a pause
    assert gate.in_conversation
    assert gate.poll(now_s=11.0) is None
    assert gate.accept("What is the capital of India?", now_s=11.5) == "What is the capital of India?"
    assert gate.poll(now_s=20.0) is None  # the greeting is no longer needed


def test_bare_wake_word_is_greeted_when_nothing_follows() -> None:
    gate = _gate(wake_word_wait_s=3.0)
    assert gate.accept("Hey robot!", now_s=10.0) is None
    assert gate.poll(now_s=12.0) is None
    assert gate.poll(now_s=13.1) == GREETING_FOR_BARE_WAKE_WORD
    assert gate.poll(now_s=14.0) is None  # only once


def test_greeting_waits_while_the_person_is_still_speaking() -> None:
    gate = _gate(wake_word_wait_s=3.0)
    gate.accept("Robot.", now_s=10.0)
    assert gate.poll(now_s=13.5, speech_in_progress=True) is None  # the question is still being spoken
    assert gate.poll(now_s=14.0, speech_in_progress=True) is None
    assert gate.accept("What is the capital of India?", now_s=14.6) == "What is the capital of India?"
    assert gate.poll(now_s=30.0) is None


def test_whisper_spellings_of_the_wake_word_are_recognised() -> None:
    assert _gate().accept("Robots, tell me about Aideology.", now_s=0.0) == "tell me about Aideology."
    assert _gate().accept("A robot, who is the CEO?", now_s=0.0) == "who is the CEO?"
    assert _gate().accept("Oh robot, turn left.", now_s=0.0) == "turn left."


def test_a_greeting_after_the_wake_word_counts_as_the_wake_word_alone() -> None:
    gate = _gate()
    assert gate.accept("Robot, hello!", now_s=0.0) is None  # greeted once no question follows
    assert gate.in_conversation


def test_words_after_a_greeting_are_kept() -> None:
    assert _gate().accept("Robot, hi there, how are you?", now_s=0.0) == "hi there, how are you?"


def test_a_sound_alike_of_the_wake_word_counts_at_the_start_of_a_sentence() -> None:
    gate = WakeWordGate(WAKE_WORDS, sound_alikes=("robert",))
    assert gate.accept("Hey Robert, where is Aideology based?", now_s=0.0) == "where is Aideology based?"
    assert gate.in_conversation


def test_a_sound_alike_in_the_middle_of_a_sentence_is_someone_else() -> None:
    gate = WakeWordGate(WAKE_WORDS, sound_alikes=("robert",))
    assert gate.accept("I told Robert about the meeting.", now_s=0.0) is None
    assert not gate.in_conversation


def test_greeting_a_person_starts_a_conversation_without_the_wake_word() -> None:
    gate = _gate()
    gate.start_conversation(now_s=10.0)  # the robot said "Hello, may I help you?" to someone walking up
    assert gate.in_conversation
    assert gate.accept("What is this?", now_s=12.0) == "What is this?"


ARABIC_WAKE_WORDS = ("robot", "روبوت", "الروبوت")


def test_the_arabic_wake_word_starts_a_conversation() -> None:
    gate = WakeWordGate(ARABIC_WAKE_WORDS)
    assert gate.accept("يا روبوت، ماذا يوجد في يدي؟", now_s=0.0) == "ماذا يوجد في يدي؟"
    assert gate.in_conversation
    assert gate.accept("وما لونه؟", now_s=5.0) == "وما لونه؟"


def test_the_wake_word_with_the_arabic_article_counts() -> None:
    assert WakeWordGate(ARABIC_WAKE_WORDS).accept("أيها الروبوت، من أنت؟", now_s=0.0) is not None


def test_an_arabic_goodbye_ends_the_conversation() -> None:
    gate = WakeWordGate(ARABIC_WAKE_WORDS)
    gate.accept("روبوت، مرحبا", now_s=0.0)
    for goodbye in ("مع السلامة", "إلى اللقاء!", "وداعا"):
        gate.start_conversation(now_s=1.0)
        assert gate.accept(goodbye, now_s=2.0) is not None  # still answered
        assert not gate.in_conversation, goodbye


def test_the_arabic_wake_word_alone_is_greeted_in_arabic() -> None:
    from robot_voice_assistant.wake_word import ARABIC_GREETING_FOR_BARE_WAKE_WORD

    gate = WakeWordGate(ARABIC_WAKE_WORDS, wake_word_wait_s=3.0)
    assert gate.accept("يا روبوت", now_s=0.0) is None
    assert gate.poll(now_s=3.5) == ARABIC_GREETING_FOR_BARE_WAKE_WORD
    assert gate.accept("Robot", now_s=10.0) is None
    assert gate.poll(now_s=13.5) == GREETING_FOR_BARE_WAKE_WORD



def test_whisper_s_other_arabic_spelling_of_robot_is_a_wake_word_too() -> None:
    from robot_voice_assistant.wake_word import DEFAULT_SOUND_ALIKES

    gate = WakeWordGate(ARABIC_WAKE_WORDS, sound_alikes=DEFAULT_SOUND_ALIKES)
    assert gate.accept("يا ربوت ممكن تقولي What is the capital of Italy?", now_s=0.0) == (
        "ممكن تقولي What is the capital of Italy?")


def test_goodbye_in_arabic_written_in_latin_letters_ends_the_conversation() -> None:
    for goodbye in ("Thank you. Ma'a Salama.", "maa salama", "Ma salama!"):
        gate = _gate()
        gate.start_conversation(now_s=0.0)
        gate.accept(goodbye, now_s=1.0)
        assert not gate.in_conversation, goodbye
