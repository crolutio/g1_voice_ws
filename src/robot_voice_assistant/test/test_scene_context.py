from robot_voice_assistant.scene_context import Detection, PersonGreeter, describe_scene, people_nearby


def detection(label: str, x: float = 0.5, height: float = 0.3, confidence: float = 0.8) -> Detection:
    return Detection(label=label, confidence=confidence, x_center=x, height=height)


def test_nothing_seen_is_said_plainly() -> None:
    assert describe_scene([]) == "Nothing in particular."


def test_objects_are_described_with_where_they_are() -> None:
    text = describe_scene([detection("apple", x=0.5, height=0.5), detection("person", x=0.1), detection("mug", x=0.9)])
    assert "an apple right in front of you, close" in text
    assert "a person on your left" in text
    assert "a mug on your right" in text


def test_several_of_the_same_kind_are_counted() -> None:
    text = describe_scene([detection("person", x=0.2), detection("person", x=0.8), detection("person", x=0.5)])
    assert text.startswith("3 people")


def test_unsure_detections_are_left_out() -> None:
    assert describe_scene([detection("banana", confidence=0.1)]) == "Nothing in particular."


def test_a_person_arriving_is_greeted_once() -> None:
    greeter = PersonGreeter(confirm_frames=2, absent_s=5.0, cooldown_s=20.0)
    assert not greeter.update(0.0, people_nearby=1)  # first glimpse: not yet sure
    assert greeter.update(0.3, people_nearby=1)  # seen twice: greet
    assert not greeter.update(0.6, people_nearby=1)  # still there: no second greeting
    assert not greeter.update(3.0, people_nearby=0)


def test_a_new_person_after_a_while_is_greeted_again_but_not_too_often() -> None:
    greeter = PersonGreeter(confirm_frames=2, absent_s=5.0, cooldown_s=20.0)
    greeter.update(0.0, 1), greeter.update(0.3, 1)
    greeter.update(1.0, 0)
    assert not (greeter.update(8.0, 1) or greeter.update(8.3, 1))  # gone 7 s, but within the 20 s cooldown
    greeter.update(9.0, 0)
    assert not greeter.update(30.0, 1)
    assert greeter.update(30.3, 1)  # gone long enough and past the cooldown


def test_only_people_who_come_close_count_not_the_crowd_behind() -> None:
    crowd = [detection("person", height=0.12), detection("person", height=0.29), detection("apple", height=0.5)]
    assert people_nearby(crowd) == 0
    assert people_nearby(crowd + [detection("person", height=0.4)]) == 1


def test_in_a_busy_hall_each_new_person_walking_up_is_greeted() -> None:
    greeter = PersonGreeter(confirm_frames=2, absent_s=5.0, cooldown_s=10.0)
    greeter.update(0.0, 1), greeter.update(0.5, 1)  # the first visitor is greeted
    assert not greeter.update(5.0, 1)  # still talking to them
    assert not greeter.update(12.0, 2)  # a second one walks up: first glimpse
    assert greeter.update(12.5, 2)  # greeted, though the first never left
    assert not greeter.update(13.0, 2)
    for t in (14.0, 16.0, 19.5):
        greeter.update(t, 1)  # one leaves and stays gone
    assert not greeter.update(25.0, 2) and greeter.update(25.5, 2)  # the next newcomer is greeted too


def test_a_crowd_flickering_in_and_out_of_view_is_not_greeted_twice() -> None:
    greeter = PersonGreeter(confirm_frames=2, absent_s=5.0, cooldown_s=10.0)
    assert any([greeter.update(float(t), 2) for t in range(11)])  # two people stay in front for 10 s
    for t in (11.0, 12.0, 13.0):
        greeter.update(t, 1)  # one is hidden for a moment
    assert not any(greeter.update(t, 2) for t in (14.0, 14.5, 15.0))  # back within 5 s: the same people


def test_the_prompt_tells_the_robot_what_it_sees() -> None:
    from datetime import datetime

    from robot_voice_assistant.conversation import build_system_prompt

    seeing = build_system_prompt("R1", {"nod": "nod"}, "", datetime(2026, 1, 1), {"stop": "stop"},
                                 surroundings="an apple right in front of you, close")
    assert "an apple right in front of you, close" in seeing
    assert "You cannot see" not in seeing
    blind = build_system_prompt("R1", {"nod": "nod"}, "", datetime(2026, 1, 1), {"stop": "stop"})
    assert "You cannot see" in blind


def test_two_models_seeing_the_same_thing_count_once() -> None:
    from robot_voice_assistant.scene_context import merge_detections

    coco = [Detection("apple", 0.75, 0.5, 0.2, box=(0.45, 0.6, 0.55, 0.8)), Detection("kite", 0.7, 0.9, 0.05, box=(0.88, 0.1, 0.92, 0.15))]
    world = [Detection("apple", 0.4, 0.5, 0.2, box=(0.46, 0.61, 0.56, 0.8)), Detection("plant", 0.5, 0.1, 0.3, box=(0.0, 0.3, 0.2, 0.6))]
    merged = merge_detections([coco, world], ignore={"kite"})
    assert sorted(d.label for d in merged) == ["apple", "plant"]
    assert next(d for d in merged if d.label == "apple").confidence == 0.75  # the more confident one is kept


def test_coco_names_are_said_the_everyday_way() -> None:
    from robot_voice_assistant.scene_context import merge_detections

    merged = merge_detections([[Detection("dining table", 0.6, 0.5, 0.3, box=(0.3, 0.5, 0.7, 0.9))]], ignore=set())
    assert merged[0].label == "table"


def test_things_named_in_the_plural_are_said_naturally() -> None:
    assert describe_scene([detection("scissors", x=0.5)]).startswith("a pair of scissors right in front of you")


def test_questions_about_the_view_right_now_are_recognised() -> None:
    from robot_voice_assistant.scene_context import is_about_the_view

    for question in ("What is this?", "What can you see in front of you?", "And what is this one?", "What am I holding up now?",
                     "Who is there?", "Do you see anyone?", "What's that on the table?", "Look at this, what is it?",
                     "What fruit is this?", "What colour is that?", "Is this an apple?", "What kind of thing is this?"):
        assert is_about_the_view(question), question
    for question in ("What is the capital of France?", "Tell me a joke.", "What is Aideology?", "Walk forward."):
        assert not is_about_the_view(question), question


def test_answers_about_the_view_are_not_kept_in_long_term_memory() -> None:
    from robot_voice_assistant.conversation import Conversation

    class Backend:
        def chat(self, messages, *_args, **_kwargs):
            return '{"reply": "I see an apple.", "gesture": "none", "motion": "none"}'

    class Retriever:
        remembered: list[str] = []

        def context_for(self, question: str) -> str:
            return ""

        def remember(self, user: str, reply: str) -> None:
            self.remembered.append(user)

    retriever = Retriever()
    conversation = Conversation(Backend(), "R1", {"none": "no gesture"}, retriever=retriever,
                                motion_descriptions={"none": "no motion"}, surroundings=lambda: "an apple right in front of you")
    conversation.respond("What is this?")
    conversation.respond("What is the capital of France?")
    assert retriever.remembered == ["What is the capital of France?"]


def test_a_question_about_the_view_carries_the_view_and_old_views_are_not_kept() -> None:
    from robot_voice_assistant.conversation import Conversation

    replies = iter(['{"reply": "I see a banana.", "gesture": "none", "motion": "none"}',
                    '{"reply": "I see an apple.", "gesture": "none", "motion": "none"}'])
    sent: list[list[dict]] = []

    class Backend:
        def chat(self, messages, *_args, **_kwargs):
            sent.append(messages)
            return next(replies)

    view = ["a banana right in front of you"]
    conversation = Conversation(Backend(), "R1", {"none": "no gesture"}, motion_descriptions={"none": "no motion"},
                                surroundings=lambda: view[0])
    conversation.respond("What is this?")
    view[0] = "an apple right in front of you"
    conversation.respond("What is this?")
    last = sent[-1]
    assert "an apple right in front of you" in last[-1]["content"]  # the current view comes with the question
    assert not any("banana" in message["content"] for message in last[1:])  # the old answer is not in the history


def test_one_thing_given_two_names_is_said_once() -> None:
    from robot_voice_assistant.scene_context import merge_detections

    mug = Detection("mug", 0.6, 0.5, 0.2, box=(0.40, 0.60, 0.55, 0.80))
    cup = Detection("cup", 0.4, 0.5, 0.2, box=(0.41, 0.61, 0.55, 0.80))  # the other model's name for the same thing
    apple = Detection("apple", 0.7, 0.2, 0.2, box=(0.10, 0.60, 0.25, 0.80))
    assert sorted(d.label for d in merge_detections([[mug, apple], [cup]], ignore=set())) == ["apple", "mug"]
