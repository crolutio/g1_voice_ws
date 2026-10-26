from robot_voice_assistant.assistant_response import AssistantResponse, parse_assistant_response

GESTURES = ("none", "wave_right", "nod")


def test_parses_valid_json() -> None:
    raw = '{"reply": "Hi there!", "gesture": "wave_right"}'
    assert parse_assistant_response(raw, GESTURES) == AssistantResponse("Hi there!", "wave_right")


def test_unknown_gesture_falls_back_to_none() -> None:
    raw = '{"reply": "Sure.", "gesture": "backflip"}'
    assert parse_assistant_response(raw, GESTURES) == AssistantResponse("Sure.", "none")


def test_missing_gesture_falls_back_to_none() -> None:
    assert parse_assistant_response('{"reply": "Okay."}', GESTURES) == AssistantResponse("Okay.", "none")


def test_plain_text_becomes_reply() -> None:
    assert parse_assistant_response("  Hello human.  ", GESTURES) == AssistantResponse("Hello human.", "none")


def test_json_embedded_in_text_is_extracted() -> None:
    raw = 'Here you go: {"reply": "Nodding now.", "gesture": "nod"} done'
    assert parse_assistant_response(raw, GESTURES) == AssistantResponse("Nodding now.", "nod")


def test_gesture_is_case_and_space_insensitive() -> None:
    raw = '{"reply": "Hello!", "gesture": " Wave_Right "}'
    assert parse_assistant_response(raw, GESTURES).gesture == "wave_right"


def test_empty_reply_gets_default_text() -> None:
    response = parse_assistant_response('{"reply": "", "gesture": "nod"}', GESTURES)
    assert response.reply != ""
    assert response.gesture == "nod"


MOTIONS = ("none", "stop", "walk_forward", "turn_left")


def test_motion_is_parsed_when_allowed() -> None:
    raw = '{"reply": "Coming!", "gesture": "wave_right", "motion": "walk_forward"}'
    response = parse_assistant_response(raw, GESTURES, MOTIONS)
    assert (response.reply, response.gesture, response.motion) == ("Coming!", "wave_right", "walk_forward")


def test_motion_defaults_to_none() -> None:
    assert parse_assistant_response('{"reply": "Hi.", "gesture": "nod"}', GESTURES, MOTIONS).motion == "none"
    assert parse_assistant_response('{"reply": "Hi.", "gesture": "nod", "motion": "fly"}', GESTURES, MOTIONS).motion == "none"


def test_motion_is_ignored_when_walking_is_not_available() -> None:
    raw = '{"reply": "Coming!", "gesture": "none", "motion": "walk_forward"}'
    assert parse_assistant_response(raw, GESTURES).motion == "none"
