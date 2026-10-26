import pytest

from robot_voice_assistant.arithmetic import find_arithmetic, spoken_calculation


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is 23 times 17?", "23 × 17 = 391"),
        ("What is fifteen times twelve?", "15 × 12 = 180"),
        ("what's 7 x 8", "7 × 8 = 56"),
        ("Calculate 144 divided by 12", "144 ÷ 12 = 12"),
        ("What is 10 divided by 4?", "10 ÷ 4 = 2.5"),
        ("What is 1,250 plus 3,750?", "1250 + 3750 = 5000"),
        ("what is one hundred minus thirty seven", "100 − 37 = 63"),
        ("What is twenty-five multiplied by four?", "25 × 4 = 100"),
        ("What is 12 squared?", "12² = 144"),
        ("What is the square root of 81?", "√81 = 9"),
        ("What is 20 percent of 150?", "20% of 150 = 30"),
        ("What is 3.5 times 2?", "3.5 × 2 = 7"),
        ("What is 10 divided by 3?", "10 ÷ 3 ≈ 3.333"),
    ],
)
def test_finds_and_solves_arithmetic(question: str, expected: str) -> None:
    assert find_arithmetic(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of India?",
        "I have two cats and one dog.",
        "Tell me about the year 1969.",
        "What is 5 divided by 0?",
        "Can you wave at me?",
    ],
)
def test_ignores_questions_without_a_calculation(question: str) -> None:
    assert find_arithmetic(question) is None


@pytest.mark.parametrize(
    ("calculation", "spoken"),
    [
        ("23 × 17 = 391", "23 times 17 is 391."),
        ("10 ÷ 3 ≈ 3.333", "10 divided by 3 is about 3.333."),
        ("100 − 37 = 63", "100 minus 37 is 63."),
        ("√81 = 9", "The square root of 81 is 9."),
        ("12² = 144", "12 squared is 144."),
        ("20% of 150 = 30", "20 percent of 150 is 30."),
    ],
)
def test_calculation_as_a_spoken_sentence(calculation: str, spoken: str) -> None:
    assert spoken_calculation(calculation) == spoken
