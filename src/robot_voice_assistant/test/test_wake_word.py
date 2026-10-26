from robot_voice_assistant.wake_word import WakeWordGate

WAKE_WORDS = ("robot", "g1", "g one")


def test_sentence_without_wake_word_is_rejected() -> None:
    gate = WakeWordGate(WAKE_WORDS, follow_up_s=8.0)
    assert gate.accept("so then she said she would call later", now_s=0.0) is None


def test_sentence_with_wake_word_is_accepted_and_stripped() -> None:
    gate = WakeWordGate(WAKE_WORDS, follow_up_s=8.0)
    assert gate.accept("Hey robot, can you wave?", now_s=0.0) == "can you wave?"


def test_multi_word_wake_word_matches() -> None:
    gate = WakeWordGate(WAKE_WORDS, follow_up_s=8.0)
    assert gate.accept("G one, what is your name?", now_s=0.0) == "what is your name?"


def test_wake_word_must_be_a_whole_word() -> None:
    gate = WakeWordGate(WAKE_WORDS, follow_up_s=8.0)
    assert gate.accept("the robotics lab is closed", now_s=0.0) is None


def test_follow_up_within_window_needs_no_wake_word() -> None:
    gate = WakeWordGate(WAKE_WORDS, follow_up_s=8.0)
    gate.note_reply_finished(now_s=10.0)
    assert gate.accept("and what about tomorrow?", now_s=15.0) == "and what about tomorrow?"


def test_follow_up_window_expires() -> None:
    gate = WakeWordGate(WAKE_WORDS, follow_up_s=8.0)
    gate.note_reply_finished(now_s=10.0)
    assert gate.accept("and what about tomorrow?", now_s=19.0) is None


def test_empty_wake_word_list_accepts_everything() -> None:
    gate = WakeWordGate((), follow_up_s=8.0)
    assert gate.accept("anything at all", now_s=0.0) == "anything at all"


def test_only_wake_word_returns_greeting_prompt() -> None:
    gate = WakeWordGate(WAKE_WORDS, follow_up_s=8.0)
    assert gate.accept("Robot!", now_s=0.0) == "Hello!"
