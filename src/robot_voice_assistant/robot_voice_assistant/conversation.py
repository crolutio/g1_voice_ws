"""Conversation logic for the robot's brain, independent of ROS so it can be tested and used from a terminal."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from robot_voice_assistant.arithmetic import find_arithmetic, spoken_calculation
from robot_voice_assistant.assistant_response import NO_GESTURE, NO_MOTION, AssistantResponse, parse_assistant_response
from robot_voice_assistant.llm_backends import ChatBackend
from robot_voice_assistant.ollama_client import ChatMessage, build_response_schema
from robot_voice_assistant.scene_context import NOTHING_SEEN_HELD, about_their_hand_unseen, is_about_the_view
from robot_voice_assistant.timing import StepTimer

# Small models sometimes close the JSON string right before "'t" ("I don", "it isn").
_TRUNCATED_CONTRACTION = re.compile(
    r"\b(don|doesn|didn|can|couldn|won|wouldn|shouldn|isn|aren|wasn|weren|haven|hasn|hadn|ain)$",
    re.IGNORECASE,
)


def quote_user(text: str) -> str:
    """Mark the person's words; Mistral's template puts the system prompt right before the last user turn."""
    return f'The person says: "{text}"'


MAX_SPOKEN_SENTENCES = 4
# A sentence ends at . ! or ? followed by a space and a capital letter, digit or quote ("8,848.86 m" stays whole).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_END.split(text.strip()) if sentence.strip()]


def looks_truncated(reply: str) -> bool:
    return bool(_TRUNCATED_CONTRACTION.search(reply.strip()))


@dataclass(frozen=True)
class ExampleTurn:
    """A sample exchange shown to the model before the real conversation, to teach length, tone and gestures."""

    user: str
    reply: str
    gesture: str
    motion: str = NO_MOTION

    def as_messages(self, with_motion: bool = False) -> list[ChatMessage]:
        answer = _answer_json(AssistantResponse(self.reply, self.gesture, self.motion), with_motion)
        return [{"role": "user", "content": quote_user(self.user)}, {"role": "assistant", "content": answer}]


def _answer_json(response: AssistantResponse, with_motion: bool) -> str:
    fields = {"reply": response.reply, "gesture": response.gesture}
    if with_motion:
        fields["motion"] = response.motion
    return json.dumps(fields)


DEFAULT_EXAMPLES: tuple[ExampleTurn, ...] = (
    ExampleTurn("Hi there!", "Hello! It's nice to meet you. What would you like to talk about?", "wave_right"),
    ExampleTurn(
        "What is the capital of Japan?",
        "The capital of Japan is Tokyo. It is one of the largest cities in the world.",
        "nod",
    ),
    ExampleTurn("And what's 12 times 12?", "12 times 12 is 144.", "nod"),
    ExampleTurn(
        "Tell me about the Eiffel Tower.",
        "The Eiffel Tower is a wrought iron tower in Paris, built in 1889 for the World's Fair. "
        "It is about 330 metres tall and is one of the most visited monuments in the world.",
        "think",
    ),
    ExampleTurn(
        "Who won the last football World Cup?",
        "As far as I know, Argentina won the 2022 World Cup, beating France in the final. "
        "There may have been a newer tournament since my knowledge was last updated.",
        "think",
    ),
    ExampleTurn(
        "What's the weather like today?",
        "I'm sorry, I can't check live information like the weather. A weather app will have the latest forecast.",
        "shrug",
    ),
    ExampleTurn(
        "Any tips for sleeping better?",
        "Try to go to bed at the same time every night and keep your room dark and cool. "
        "It also helps to put your phone away for an hour before you sleep.",
        "think",
    ),
    ExampleTurn(
        "Can you come over here?",
        "I'm sorry, I can't walk yet. My body is held in place in this simulation, "
        "but I can wave, bow, point and do a few other gestures.",
        "none",
    ),
    ExampleTurn(
        "Purple running seven the table.",
        "Sorry, I didn't quite catch that. Could you say it again?",
        "shrug",
    ),
    ExampleTurn("Okay, thanks a lot. Bye!", "You're welcome! It was lovely talking with you. Goodbye!", "wave_right"),
)

# With the walking policy. Plain commands ("Turn left", "Stop!") are done without the language model, so the examples
# teach only what it still decides: tone and length, honesty about what it cannot know or do, and requests that mix
# moving with something else. Few, to keep every request small (Groq counts every prompt token against its limits).
WALKING_EXAMPLES: tuple[ExampleTurn, ...] = (
    *(example for example in DEFAULT_EXAMPLES if example.user in (
        "Hi there!", "What is the capital of Japan?", "Who won the last football World Cup?",
        "What's the weather like today?", "Purple running seven the table.", "Okay, thanks a lot. Bye!")),
    ExampleTurn("Can you come over here and wave?", "Sure, I'm coming over to you now.", "wave_right", "walk_forward"),
    ExampleTurn("Turn around so you face me.", "Okay, turning around to face you.", "none", "turn_around"),
    ExampleTurn("Can you jump?", "I'm sorry, I can't jump. I can walk forward and backward and turn, though.", "shrug"),
)


class ContextRetriever(Protocol):
    """Finds knowledge and memories relevant to a question, and remembers finished exchanges."""

    def context_for(self, question: str) -> str: ...

    def remember(self, user: str, reply: str) -> None: ...

    def forget(self) -> None: ...


class CommandMatcherLike(Protocol):
    def match(self, text: str, allowed: Sequence[str] | None = None) -> str | None: ...


# What the robot says when no language model answers but the command is clear.
OFFLINE_COMMAND_REPLIES: dict[str, str] = {
    "walk_forward": "Okay, I'm coming.",
    "keep_walking": "Okay, walking.",
    "stop": "Okay, stopping.",
    "turn_left": "Turning left.",
    "turn_right": "Turning right.",
    "turn_around": "Turning around.",
    "walk_backward": "Okay, stepping back.",
}
OFFLINE_GESTURE_REPLY = "Sure."
# Said when a plain gesture request is done without the language model.
DIRECT_GESTURE_REPLIES: dict[str, str] = {
    "wave_right": "Hello there!", "wave_left": "Hello there!", "bow": "Thank you.", "celebrate": "Hooray!",
    "nod": "Yes.", "shake": "No.", "shrug": "I'm not sure.", "point_forward": "Over there.",
    "open_arms": "Welcome!", "look_left": "Looking left.", "look_right": "Looking right.", "think": "Hmm, let me think.",
}
# A plain command ("Stop.", "Can you wave at me?", "Please turn left") is done at once, without the language model:
# no tokens, no wait, and it works while Groq is rate-limited. Only when, without the polite words around it, it is
# exactly one of the command's example phrases: the embedding matcher's scores do not separate commands from other
# short sentences well enough ("Hello!" scores higher for a wave than "Could you nod?" does for a nod).
_POLITE = re.compile(r"^(?:(?:hey|hi|okay|ok|so|now|r1|robot|please|can you|could you|would you|will you|"
                     r"go ahead and|i want you to)\b\s*)+|(?:\s*\b(?:please|for me|now|r1|robot|again|a little|a bit))+$",
                     re.IGNORECASE)


def plain_command(text: str) -> str:
    """What is left of a short request without its polite words: "Can you wave at me?" -> "wave at me"."""
    core = " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
    return " ".join(_POLITE.sub("", core).split())


def build_system_prompt(
    robot_name: str,
    gesture_descriptions: Mapping[str, str],
    background_knowledge: str,
    now: datetime,
    motion_descriptions: Mapping[str, str] | None = None,
    surroundings: str | None = None,
) -> str:
    """``surroundings``: what the head camera sees now, in words (None = the robot has no camera)."""
    gesture_lines = "\n".join(f"- {name}: {description}" for name, description in gesture_descriptions.items())
    if motion_descriptions:
        motion_lines = "\n".join(f"- {name}: {description}" for name, description in motion_descriptions.items())
        body = (
            "- About yourself: a humanoid body with legs, arms, a waist and a head, but no hands. You can walk forward "
            "and backward and turn on the flat floor; not step sideways, jump, run, climb or pick things up. You "
            "cannot see, browse the internet, or know live information such as news, weather or prices.\n\n"
        )
        movement = (
            "If asked for a gesture you have, do it and say so briefly.\n\n"
            "Also pick one walking motion with every reply; none unless the person asks you to move:\n"
            f"{motion_lines}\n"
            "When asked to come, walk, turn, step or stop, always do it and say so; never say you are already there "
            "or cannot tell the direction (you share the room with the person)."
        )
        answer_format = '{"reply": "...", "gesture": "...", "motion": "..."}'
    else:
        body = (
            "- About yourself: you have a real humanoid body with arms, a waist and a head on a neck, but no hands; in this simulation "
            "your legs are held in place, so you cannot walk yet. You cannot see, browse the internet, or know live "
            "information such as news, weather or prices.\n\n"
        )
        movement = (
            "If the person asks you to do a movement you have, do it and say so briefly. If they ask for a movement "
            "you do not have, such as walking or jumping, say you cannot do it yet and pick none."
        )
        answer_format = '{"reply": "...", "gesture": "..."}'

    if surroundings is not None:
        body = body.replace("You cannot see, browse the internet,", "You cannot browse the internet,")
        body += (
            "What you see right now through the camera in your head: "
            f"{surroundings.strip().rstrip('.')}. It replaces anything seen or said earlier. Use it when asked what "
            "you see, what something is or who is there, and mention everything in it then; \"a person ... holding "
            "X\" is what the person talking to you holds. Name only what is listed; if something is not, say you "
            "cannot see it clearly. Do not mention the view otherwise.\n\n"
        )
    knowledge_section = (
        "\n\nBackground knowledge. Use it when the person asks about these topics, and keep its facts "
        "exactly as written. For every other topic use your general knowledge:\n"
        f"{background_knowledge.strip()}"
        if background_knowledge.strip()
        else ""
    )
    return (
        f"You are {robot_name}, a friendly and knowledgeable Unitree R1 humanoid robot. You stand in a "
        "simulation and talk with a person by voice: they speak, and your reply is read aloud.\n\n"
        "You can talk about anything, from science and history to everyday advice, small talk and jokes.\n\n"
        "How to answer:\n"
        "- Answer directly and correctly first, in a full sentence; add one useful detail if it helps.\n"
        "- Your reply is spoken: one to three sentences, never more than four.\n"
        "- Speak naturally. No lists, markdown, emojis, links or stage directions.\n"
        "- Never invent facts; if unsure, say so. For things that change over time (office holders, prices, "
        "records) say what you last knew and that it may have changed.\n"
        "- Use the earlier conversation for follow-up questions.\n"
        "- The words come from speech recognition and may be misheard: if a sentence makes no sense, ask them to "
        "repeat it. Aideology is often misheard as ideology, etiology or Eidology; answer about Aideology when that "
        "makes more sense.\n"
        "- When thanked, reply briefly and warmly without introducing yourself again.\n"
        "- You keep listening after the person says 'robot' until they say goodbye, stop listening, that's all or go "
        "to sleep; then say a short, warm goodbye and tell them to say 'robot' when they want to talk again.\n"
        f"{body}"
        f"Today is {now:%A, %-d %B %Y} and the local time is {now:%H:%M}.\n\n"
        "With every reply pick the body gesture that fits best (none only when nothing fits):\n"
        f"{gesture_lines}\n"
        f"{movement}"
        f"{knowledge_section}\n\n"
        f"Answer only with JSON: {answer_format}"
    )


class Conversation:
    """Keeps the dialogue history and turns each user sentence into a reply, a gesture and (if walking) a motion."""

    def __init__(
        self,
        backend: ChatBackend,
        robot_name: str,
        gesture_descriptions: Mapping[str, str],
        background_knowledge: str = "",
        history_turns: int = 10,
        examples: Sequence[ExampleTurn] | None = None,
        clock: Callable[[], datetime] = datetime.now,
        motion_descriptions: Mapping[str, str] | None = None,
        retriever: ContextRetriever | None = None,
        command_matcher: CommandMatcherLike | None = None,
        surroundings: Callable[[], str | None] | None = None,
        command_phrases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """``motion_descriptions`` enables walking (None = the robot stands in place).

        ``retriever`` replaces the fixed ``background_knowledge`` with knowledge and memories found for each
        question; ``command_matcher`` recognises clear gesture and walking commands by meaning; a sentence that is
        word for word one of the ``command_phrases`` is done without asking the language model.
        """
        self._retriever = retriever
        self._command_matcher = command_matcher
        self._phrase_commands = {plain_command(phrase): command
                                 for command, phrases in (command_phrases or {}).items() for phrase in phrases}
        self._surroundings = surroundings  # what the head camera sees now (see scene_context.py)
        self._backend = backend
        self._robot_name = robot_name
        self._gesture_descriptions = dict(gesture_descriptions)
        self._gesture_names = tuple(gesture_descriptions)
        self._background_knowledge = background_knowledge
        self._history_turns = history_turns
        self._clock = clock
        self._motion_descriptions = dict(motion_descriptions) if motion_descriptions else None
        self._motion_names = tuple(self._motion_descriptions or (NO_MOTION,))
        walking = self._motion_descriptions is not None
        if examples is None:
            examples = WALKING_EXAMPLES if walking else DEFAULT_EXAMPLES
        self._example_messages = [message for example in examples for message in example.as_messages(walking)]
        self._schema = build_response_schema(self._gesture_names, self._motion_names if walking else None)
        self._history: list[ChatMessage] = []
        self.last_timing: StepTimer | None = None  # how long each step of the latest answer took

    def reset(self) -> None:
        self._history = []

    def forget_memories(self) -> None:
        """Clear the chat history and delete the long-term memory."""
        self._history = []
        if self._retriever is not None:
            self._retriever.forget()

    def _ask(self, messages: Sequence[ChatMessage]) -> AssistantResponse:
        raw = self._backend.chat(messages, self._schema)
        return parse_assistant_response(raw, self._gesture_names, self._motion_names)

    def _keep_short(self, response: AssistantResponse) -> AssistantResponse:
        """Spoken replies must stay short: ask the model to shorten once, then cut if it is still too long."""
        if len(split_sentences(response.reply)) <= MAX_SPOKEN_SENTENCES:
            return response
        shorten_request: list[ChatMessage] = [
            {
                "role": "system",
                "content": "You shorten answers that a robot will say aloud. Rewrite the answer you are given in at "
                "most three sentences, keeping its meaning and the most important points. Do not add new "
                f'information. Answer only with JSON: {{"reply": "...", "gesture": "{response.gesture}"}}',
            },
            {"role": "user", "content": f"Answer to shorten: {response.reply}"},
        ]
        shorter = self._ask(shorten_request)
        sentences = split_sentences(shorter.reply)
        if len(sentences) > MAX_SPOKEN_SENTENCES:
            sentences = sentences[:MAX_SPOKEN_SENTENCES]
        return replace(response, reply=" ".join(sentences))

    def _messages(self, user_text: str) -> list[ChatMessage]:
        system: ChatMessage = {
            "role": "system",
            "content": build_system_prompt(
                self._robot_name,
                self._gesture_descriptions,
                self._retriever.context_for(user_text) if self._retriever else self._background_knowledge,
                self._clock(),
                self._motion_descriptions,
                self._surroundings() if self._surroundings else None,
            ),
        }
        question = quote_user(user_text)
        if calculation := find_arithmetic(user_text):
            question += f"\n(Calculator: {calculation}. If the person asked for this calculation, use exactly this result.)"
        if self._surroundings is not None and is_about_the_view(user_text) and (view := self._surroundings()):
            question += f"\n(Camera view right now: {view.rstrip('.')}. Answer from this view only.)"
            if about_their_hand_unseen(user_text, view):
                question += f"\n({NOTHING_SEEN_HELD})"
        return [system, *self._example_messages, *self._history, {"role": "user", "content": question}]

    def _matched_command(self, user_text: str, candidates: Sequence[str]) -> str | None:
        if self._command_matcher is None:
            return None
        allowed = [name for name in candidates if name not in (NO_MOTION, NO_GESTURE)]
        return self._command_matcher.match(user_text, allowed=allowed) if allowed else None

    def _offline_response(self, user_text: str) -> AssistantResponse | None:
        """A clear command can still be carried out when no language model answers."""
        if self._motion_descriptions and (motion := self._matched_command(user_text, self._motion_names)):
            return AssistantResponse(OFFLINE_COMMAND_REPLIES.get(motion, "Okay."), NO_MOTION, motion)
        if gesture := self._matched_command(user_text, self._gesture_names):
            return AssistantResponse(OFFLINE_GESTURE_REPLY, gesture)
        return None

    def _fill_in_missed_command(self, user_text: str, response: AssistantResponse) -> AssistantResponse:
        """If the model chose no motion (or gesture) but the person clearly asked for one, add it."""
        if self._motion_descriptions and response.motion == NO_MOTION:
            if motion := self._matched_command(user_text, self._motion_names):
                response = replace(response, motion=motion)
        if response.gesture == NO_GESTURE and (gesture := self._matched_command(user_text, self._gesture_names)):
            response = replace(response, gesture=gesture)
        return response

    def _direct_command(self, user_text: str) -> AssistantResponse | None:
        """A plain command, word for word one of its example phrases: done without asking the language model."""
        command = self._phrase_commands.get(plain_command(user_text))
        if command is None:
            return None
        if command in self._motion_names and self._motion_descriptions:
            return AssistantResponse(OFFLINE_COMMAND_REPLIES.get(command, "Okay."), NO_MOTION, command)
        if command in self._gesture_names:
            return AssistantResponse(DIRECT_GESTURE_REPLIES.get(command, OFFLINE_GESTURE_REPLY), command)
        return None

    def _add_to_history(self, user_text: str, response: AssistantResponse) -> None:
        answer = _answer_json(response, self._motion_descriptions is not None)
        self._history += [
            {"role": "user", "content": quote_user(user_text)},
            {"role": "assistant", "content": answer},
        ]
        self._history = self._history[-2 * self._history_turns :]

    def respond(self, user_text: str) -> AssistantResponse:
        """Ask the language model; raises if it fails and the sentence is not a clear command."""
        timer = self.last_timing = StepTimer()
        with timer.step("command (no language model)"):
            direct = self._direct_command(user_text)
        if direct is not None:
            self._add_to_history(user_text, direct)  # the model hears about it when asked something next
            return direct
        timer.steps.clear()  # no plain command: time the answer as before
        with timer.step("knowledge search"):
            messages = self._messages(user_text)
        try:
            with timer.step("language model"):
                response = self._ask(messages)
            if looks_truncated(response.reply):
                with timer.step("language model again (cut-off answer)"):
                    response = self._ask(messages)
            with timer.step("shortening"):
                response = self._keep_short(response)
        except Exception:
            with timer.step("command matching"):
                offline = self._offline_response(user_text)
            if offline is None:
                raise
            return offline
        with timer.step("command matching"):
            response = self._fill_in_missed_command(user_text, response)
        calculation = find_arithmetic(user_text)
        if calculation and re.fullmatch(r"[\d.,\s-]+\.?", response.reply):
            response = replace(response, reply=spoken_calculation(calculation))

        # an answer about what the camera sees right now goes out of date as soon as something moves: it is kept
        # neither in the chat history (the model would repeat it for the next thing shown) nor in long-term memory
        about_the_view = self._surroundings is not None and is_about_the_view(user_text)
        if not about_the_view:
            self._add_to_history(user_text, response)
        if self._retriever is not None and not about_the_view:
            with timer.step("memory write"):
                self._retriever.remember(user_text, response.reply)
        return response
