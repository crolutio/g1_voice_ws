import json
from collections.abc import Sequence
from datetime import datetime

import pytest

from robot_voice_assistant.conversation import (
    OFFLINE_COMMAND_REPLIES,
    Conversation,
    ExampleTurn,
    looks_truncated,
    quote_user,
    split_sentences,
)
from robot_voice_assistant.ollama_client import ChatMessage

GESTURES = {"none": "stand still", "nod": "nod yes", "wave_right": "wave"}
FIXED_NOW = datetime(2026, 10, 23, 14, 30)


class FakeBackend:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage], response_schema: dict[str, object]) -> str:
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _conversation(
    backend: FakeBackend,
    knowledge: str = "",
    history_turns: int = 6,
    examples: Sequence[ExampleTurn] = (),
) -> Conversation:
    return Conversation(
        backend,
        robot_name="G1",
        gesture_descriptions=GESTURES,
        background_knowledge=knowledge,
        history_turns=history_turns,
        examples=examples,
        clock=lambda: FIXED_NOW,
    )


def test_examples_come_before_the_real_conversation() -> None:
    backend = FakeBackend(['{"reply": "Hi.", "gesture": "none"}'])
    example = ExampleTurn("What is 7 times 8?", "Seven times eight is fifty-six.", "nod")
    _conversation(backend, examples=[example]).respond("Hello")
    messages = backend.calls[0]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == quote_user("What is 7 times 8?")
    assert json.loads(messages[2]["content"]) == {"reply": "Seven times eight is fifty-six.", "gesture": "nod"}
    assert messages[3]["content"] == quote_user("Hello")


def test_default_examples_use_known_gestures() -> None:
    from robot_voice_assistant.conversation import DEFAULT_EXAMPLES
    from robot_voice_assistant.gestures import GESTURE_DESCRIPTIONS

    assert DEFAULT_EXAMPLES
    assert all(example.gesture in GESTURE_DESCRIPTIONS for example in DEFAULT_EXAMPLES)


def test_answer_and_gesture_are_returned() -> None:
    backend = FakeBackend(['{"reply": "The capital of India is New Delhi.", "gesture": "nod"}'])
    response = _conversation(backend).respond("What is the capital of India?")
    assert response.reply == "The capital of India is New Delhi."
    assert response.gesture == "nod"


def test_system_prompt_contains_date_and_knowledge() -> None:
    backend = FakeBackend(['{"reply": "Hi.", "gesture": "none"}'])
    _conversation(backend, knowledge="Aideology is based in Dubai.").respond("Hello")
    system_prompt = backend.calls[0][0]["content"]
    assert backend.calls[0][0]["role"] == "system"
    assert "Friday, 23 October 2026" in system_prompt
    assert "Aideology is based in Dubai." in system_prompt


def test_follow_up_question_sees_earlier_turns() -> None:
    backend = FakeBackend(
        [
            '{"reply": "New Delhi.", "gesture": "nod"}',
            '{"reply": "Beijing.", "gesture": "nod"}',
        ]
    )
    conversation = _conversation(backend)
    conversation.respond("What is the capital of India?")
    conversation.respond("And of China?")
    second_call = backend.calls[1]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    assert second_call[1]["content"] == quote_user("What is the capital of India?")
    assert json.loads(second_call[2]["content"]) == {"reply": "New Delhi.", "gesture": "nod"}


def test_history_is_trimmed_to_the_last_turns() -> None:
    backend = FakeBackend([f'{{"reply": "Answer {i}.", "gesture": "none"}}' for i in range(5)])
    conversation = _conversation(backend, history_turns=2)
    for i in range(5):
        conversation.respond(f"Question {i}")
    last_call = backend.calls[-1]
    # system + 2 earlier turns (user/assistant pairs) + the new question
    assert len(last_call) == 1 + 2 * 2 + 1
    assert last_call[1]["content"] == quote_user("Question 2")


def test_failed_request_is_not_kept_in_history() -> None:
    backend = FakeBackend([ConnectionError("down"), '{"reply": "Hi again.", "gesture": "none"}'])
    conversation = _conversation(backend)
    with pytest.raises(ConnectionError):
        conversation.respond("Hello?")
    conversation.respond("Hello again")
    assert [m["role"] for m in backend.calls[1]] == ["system", "user"]


def test_truncated_reply_is_retried_once() -> None:
    backend = FakeBackend(
        [
            '{"reply": "I\'m sorry, I don", "gesture": "none"}',
            '{"reply": "I\'m sorry, I don\'t know that.", "gesture": "none"}',
        ]
    )
    response = _conversation(backend).respond("Where are my keys?")
    assert response.reply == "I'm sorry, I don't know that."
    assert len(backend.calls) == 2


def test_reset_forgets_the_conversation() -> None:
    backend = FakeBackend(['{"reply": "A.", "gesture": "none"}', '{"reply": "B.", "gesture": "none"}'])
    conversation = _conversation(backend)
    conversation.respond("First")
    conversation.reset()
    conversation.respond("Second")
    assert [m["role"] for m in backend.calls[1]] == ["system", "user"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("I'm sorry, I don", True),
        ("That was just a joke, I wasn", True),
        ("It isn", True),
        ("The capital of India is New Delhi.", False),
        ("Sure, here you go", False),
        ("I don't know.", False),
    ],
)
def test_looks_truncated(reply: str, expected: bool) -> None:
    assert looks_truncated(reply) is expected


def test_user_words_are_clearly_marked() -> None:
    # Mistral puts the system prompt right before the last user turn; quoting keeps the two apart.
    assert quote_user("Thank you, goodbye!") == 'The person says: "Thank you, goodbye!"'


def test_calculation_result_is_given_to_the_model() -> None:
    backend = FakeBackend(['{"reply": "23 times 17 is 391.", "gesture": "nod"}', '{"reply": "Ok.", "gesture": "none"}'])
    conversation = _conversation(backend)
    conversation.respond("What is 23 times 17?")
    last_user_message = backend.calls[0][-1]["content"]
    assert last_user_message.startswith(quote_user("What is 23 times 17?"))
    assert "23 × 17 = 391" in last_user_message
    # The hint is not kept in the history.
    conversation.respond("Thanks")
    assert backend.calls[1][1]["content"] == quote_user("What is 23 times 17?")


def test_no_calculator_hint_for_normal_questions() -> None:
    backend = FakeBackend(['{"reply": "New Delhi.", "gesture": "nod"}'])
    _conversation(backend).respond("What is the capital of India?")
    assert backend.calls[0][-1]["content"] == quote_user("What is the capital of India?")


LONG_REPLY = "One. Two is here. Three is next. Four follows. Five ends it."


def test_split_sentences_keeps_decimals_and_abbreviations_together() -> None:
    assert split_sentences("Everest is 8,848.86 metres tall. It is in Nepal.") == [
        "Everest is 8,848.86 metres tall.",
        "It is in Nepal.",
    ]
    assert split_sentences("Hi! How are you? Fine.") == ["Hi!", "How are you?", "Fine."]


def test_long_reply_is_shortened_by_the_model() -> None:
    backend = FakeBackend(
        [
            json.dumps({"reply": LONG_REPLY, "gesture": "think"}),
            json.dumps({"reply": "Short version. Done.", "gesture": "think"}),
        ]
    )
    response = _conversation(backend, knowledge="Aideology facts.").respond("Tell me everything")
    assert response.reply == "Short version. Done."
    assert len(backend.calls) == 2
    # Shortening is a separate request with only the long answer: no conversation, no background knowledge.
    shorten_call = backend.calls[1]
    assert [m["role"] for m in shorten_call] == ["system", "user"]
    assert LONG_REPLY in shorten_call[1]["content"]
    assert "Aideology" not in shorten_call[0]["content"] + shorten_call[1]["content"]


def test_reply_that_stays_long_is_cut_to_four_sentences() -> None:
    backend = FakeBackend([json.dumps({"reply": LONG_REPLY, "gesture": "think"})] * 2)
    response = _conversation(backend).respond("Tell me everything")
    assert response.reply == "One. Two is here. Three is next. Four follows."


def test_bare_number_reply_becomes_a_sentence() -> None:
    backend = FakeBackend(['{"reply": "391", "gesture": "nod"}'])
    response = _conversation(backend).respond("What is 23 times 17?")
    assert response.reply == "23 times 17 is 391."


WALK_MOTIONS = {"none": "stay", "stop": "stop walking", "walk_forward": "walk forward one metre"}


def test_walking_mode_describes_motions_and_returns_them() -> None:
    backend = FakeBackend(['{"reply": "On my way!", "gesture": "none", "motion": "walk_forward"}'])
    conversation = Conversation(
        backend, "G1", GESTURES, motion_descriptions=WALK_MOTIONS, examples=(), clock=lambda: FIXED_NOW
    )
    response = conversation.respond("Come here please")
    system_prompt = backend.calls[0][0]["content"]
    assert "walk forward one metre" in system_prompt
    assert "cannot walk" not in system_prompt
    assert response.motion == "walk_forward"


def test_standing_mode_says_it_cannot_walk() -> None:
    backend = FakeBackend(['{"reply": "Sorry.", "gesture": "none"}'])
    _conversation(backend).respond("Come here")
    assert "cannot walk" in backend.calls[0][0]["content"]


def test_walking_examples_use_known_motions() -> None:
    from robot_voice_assistant.conversation import WALKING_EXAMPLES
    from robot_voice_assistant.motions import MOTION_DESCRIPTIONS

    assert WALKING_EXAMPLES
    assert all(example.motion in MOTION_DESCRIPTIONS for example in WALKING_EXAMPLES)


class FakeRetriever:
    def __init__(self) -> None:
        self.remembered: list[tuple[str, str]] = []

    def context_for(self, question: str) -> str:
        return f"Relevant facts for: {question}"

    def remember(self, user: str, reply: str) -> None:
        self.remembered.append((user, reply))

    def forget(self) -> None:
        self.remembered.clear()


class FakeMatcher:
    def __init__(self, command: str | None) -> None:
        self.command = command

    def match(self, text: str, allowed: Sequence[str] | None = None) -> str | None:
        return self.command if allowed is None or self.command in allowed else None


def test_retrieved_context_goes_into_the_system_prompt_and_exchanges_are_remembered() -> None:
    backend = FakeBackend(['{"reply": "Jorge Colotto.", "gesture": "nod"}'])
    retriever = FakeRetriever()
    conversation = Conversation(backend, "G1", GESTURES, examples=(), clock=lambda: FIXED_NOW, retriever=retriever)
    conversation.respond("Who is the CEO of Aideology?")
    assert "Relevant facts for: Who is the CEO of Aideology?" in backend.calls[0][0]["content"]
    assert retriever.remembered == [("Who is the CEO of Aideology?", "Jorge Colotto.")]


def test_command_matcher_fills_in_a_missed_motion() -> None:
    backend = FakeBackend(['{"reply": "Sure!", "gesture": "none", "motion": "none"}'])
    conversation = Conversation(
        backend, "G1", GESTURES, motion_descriptions=WALK_MOTIONS, examples=(), clock=lambda: FIXED_NOW,
        command_matcher=FakeMatcher("walk_forward"),
    )
    assert conversation.respond("Come over to me").motion == "walk_forward"


def test_command_matcher_does_not_override_the_model() -> None:
    backend = FakeBackend(['{"reply": "Turning.", "gesture": "none", "motion": "stop"}'])
    conversation = Conversation(
        backend, "G1", GESTURES, motion_descriptions=WALK_MOTIONS, examples=(), clock=lambda: FIXED_NOW,
        command_matcher=FakeMatcher("walk_forward"),
    )
    assert conversation.respond("Stop and come here").motion == "stop"


def test_clear_commands_still_work_when_no_language_model_answers() -> None:
    backend = FakeBackend([ConnectionError("groq and llama both down")])
    conversation = Conversation(
        backend, "G1", GESTURES, motion_descriptions=WALK_MOTIONS, examples=(), clock=lambda: FIXED_NOW,
        command_matcher=FakeMatcher("walk_forward"),
    )
    response = conversation.respond("Come here")
    assert response.motion == "walk_forward"
    assert response.reply


def test_without_a_matching_command_a_backend_failure_is_raised() -> None:
    backend = FakeBackend([ConnectionError("down")])
    conversation = Conversation(
        backend, "G1", GESTURES, examples=(), clock=lambda: FIXED_NOW, command_matcher=FakeMatcher(None)
    )
    with pytest.raises(ConnectionError):
        conversation.respond("Tell me a story")


def test_forget_memories_clears_history_and_long_term_memory() -> None:
    backend = FakeBackend(['{"reply": "Hi Radhika.", "gesture": "none"}', '{"reply": "Hello.", "gesture": "none"}'])
    retriever = FakeRetriever()
    conversation = Conversation(backend, "G1", GESTURES, examples=(), clock=lambda: FIXED_NOW, retriever=retriever)
    conversation.respond("My name is Radhika")
    conversation.forget_memories()
    assert retriever.remembered == []
    conversation.respond("Hello")
    assert [m["role"] for m in backend.calls[1]] == ["system", "user"]


def test_each_answer_records_how_long_its_steps_took() -> None:
    backend = FakeBackend(['{"reply": "Jorge Colotto.", "gesture": "nod"}'])
    conversation = Conversation(backend, "G1", GESTURES, examples=(), clock=lambda: FIXED_NOW, retriever=FakeRetriever())
    assert conversation.last_timing is None
    conversation.respond("Who is the CEO of Aideology?")
    assert conversation.last_timing is not None
    steps = [name for name, _ in conversation.last_timing.steps]
    assert steps[:2] == ["knowledge search", "language model"]


def test_timing_is_recorded_when_the_language_model_fails() -> None:
    backend = FakeBackend([ConnectionError("groq and llama both down")])
    conversation = Conversation(
        backend, "G1", GESTURES, motion_descriptions=WALK_MOTIONS, examples=(), clock=lambda: FIXED_NOW,
        command_matcher=FakeMatcher("walk_forward"),
    )
    conversation.respond("Walk forward to the door")  # not a plain command: the model is asked
    assert conversation.last_timing is not None
    assert [name for name, _ in conversation.last_timing.steps][:2] == ["knowledge search", "language model"]


def test_system_prompt_explains_how_a_conversation_ends() -> None:
    backend = FakeBackend(['{"reply": "Goodbye!", "gesture": "wave_right"}'])
    _conversation(backend).respond("Goodbye.")
    prompt = backend.calls[0][0]["content"]
    assert "say 'robot'" in prompt
    assert "stop listening" in prompt


def test_system_prompt_warns_that_names_may_be_misheard() -> None:
    backend = FakeBackend(['{"reply": "In Dubai.", "gesture": "none"}'])
    _conversation(backend).respond("Where is ideology based?")
    assert "ideology" in backend.calls[0][0]["content"].lower()


TURN_MOTIONS = {**WALK_MOTIONS, "turn_left": "turn 90 degrees to the left", "walk_backward": "step back"}
PHRASES = {"turn_left": ["turn left"], "walk_forward": ["come here", "walk forward"], "walk_backward": ["step back"],
           "wave_right": ["wave at me"], "nod": ["nod"]}


def _commands(backend: FakeBackend, motions: dict[str, str] | None = TURN_MOTIONS) -> Conversation:
    return Conversation(backend, "G1", GESTURES, motion_descriptions=motions, examples=(), clock=lambda: FIXED_NOW,
                        command_phrases=PHRASES)


def test_a_plain_command_is_done_without_the_language_model() -> None:
    backend = FakeBackend([])  # any call to the model would fail
    conversation = _commands(backend)
    for said, motion in [("Please turn left.", "turn_left"), ("Come here!", "walk_forward"),
                         ("Step back a little.", "walk_backward"), ("Okay R1, walk forward please", "walk_forward")]:
        response = conversation.respond(said)
        assert (response.motion, response.reply) == (motion, OFFLINE_COMMAND_REPLIES[motion]), said
    assert backend.calls == [] and conversation.last_timing.steps[0][0] == "command (no language model)"


def test_a_plain_gesture_request_is_done_without_the_language_model() -> None:
    backend = FakeBackend([])
    response = _commands(backend).respond("Can you wave at me?")
    assert response.gesture == "wave_right" and response.reply and backend.calls == []


def test_questions_greetings_and_longer_requests_still_go_to_the_language_model() -> None:
    for said in ["What does turn left mean?", "Hello!", "Turn left and tell me what you see", "Move closer",
                 "Turn around again so you face me"]:
        backend = FakeBackend(['{"reply": "Okay.", "gesture": "none", "motion": "none"}'])
        _commands(backend).respond(said)
        assert len(backend.calls) == 1, said


def test_the_language_model_still_hears_about_commands_done_without_it() -> None:
    backend = FakeBackend(['{"reply": "You asked me to turn left.", "gesture": "none", "motion": "none"}'])
    conversation = _commands(backend)
    conversation.respond("Turn left")
    conversation.respond("What did I just ask you to do?")
    assert any("Turn left" in message["content"] for message in backend.calls[0][1:])


def test_the_prompt_stays_small_so_groq_s_token_limits_go_further() -> None:
    from robot_voice_assistant.conversation import WALKING_EXAMPLES, build_system_prompt
    from robot_voice_assistant.gestures import GESTURE_DESCRIPTIONS
    from robot_voice_assistant.motions import MOTION_DESCRIPTIONS

    prompt = build_system_prompt("R1", GESTURE_DESCRIPTIONS, "", FIXED_NOW, MOTION_DESCRIPTIONS,
                                 surroundings="a person right in front of you, close, holding an apple")
    examples = "".join(m["content"] for example in WALKING_EXAMPLES for m in example.as_messages(True))
    # about 4 characters a token: ~1,000 tokens of instructions and ~400 of examples (was ~1,280 and ~720)
    assert len(prompt) < 4_000, len(prompt)
    assert len(WALKING_EXAMPLES) <= 10 and len(examples) < 1_600, (len(WALKING_EXAMPLES), len(examples))


def test_the_examples_teach_what_the_model_still_decides_about_moving() -> None:
    from robot_voice_assistant.conversation import WALKING_EXAMPLES

    both = [e for e in WALKING_EXAMPLES if e.motion != "none" and e.gesture != "none"]
    assert both  # walking and a gesture in one request ("come over here and wave")
    assert any(e.motion == "none" and "can't" in e.reply for e in WALKING_EXAMPLES)  # something it cannot do
