"""A tiny calculator for spoken arithmetic, because language models get multiplication wrong."""

from __future__ import annotations

import math
import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"and"}

_NUMBER = r"(-?\d+(?:\.\d+)?)"
_BINARY_OPERATORS: dict[str, tuple[str, str]] = {
    r"(?:times|multiplied by|x|\*|×)": ("×", "multiply"),
    r"(?:plus|\+|added to)": ("+", "add"),
    r"(?:minus|-|−|take away)": ("−", "subtract"),
    r"(?:divided by|/|÷|over)": ("÷", "divide"),
}


def _words_to_number(words: list[str]) -> int | None:
    total, current, seen_number = 0, 0, False
    for word in words:
        if word == "and":
            continue
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word in _SCALES:
            total += max(current, 1) * _SCALES[word]
            current = 0
        else:
            return None
        seen_number = True
    return total + current if seen_number else None


def _replace_number_words(text: str) -> str:
    """'twenty-five times four' -> '25 times 4'."""
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text.replace("-", " - "))
    output: list[str] = []
    run: list[str] = []

    def flush() -> None:
        while run and run[-1] == "and":
            run.pop()
        if run:
            value = _words_to_number(run)
            output.append(str(value) if value is not None else " ".join(run))
        run.clear()

    for token in tokens:
        word = token.lower()
        if word in _NUMBER_WORDS and (word != "and" or run):
            run.append(word)
        elif run and token.strip() in ("", "-"):
            continue  # spaces or hyphens inside a number ("twenty-five")
        else:
            flush()
            output.append(token)
    flush()
    return re.sub(r"\s+", " ", "".join(output)).replace(" - ", " minus ").strip()


def _format(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _result(value: float) -> str:
    exact = math.isclose(value, round(value, 3), abs_tol=1e-9)
    return f"{'=' if exact else '≈'} {_format(value)}"


def find_arithmetic(question: str) -> str | None:
    """Return a worked result such as '23 × 17 = 391' when the question asks for a calculation."""
    text = _replace_number_words(question.replace(",", ""))

    if match := re.search(rf"square root of {_NUMBER}", text, re.IGNORECASE):
        number = float(match.group(1))
        return f"√{_format(number)} {_result(math.sqrt(number))}" if number >= 0 else None
    if match := re.search(rf"{_NUMBER} squared", text, re.IGNORECASE):
        number = float(match.group(1))
        return f"{_format(number)}² {_result(number**2)}"
    if match := re.search(rf"{_NUMBER} ?(?:%|percent) of {_NUMBER}", text, re.IGNORECASE):
        percent, base = float(match.group(1)), float(match.group(2))
        return f"{_format(percent)}% of {_format(base)} {_result(percent / 100 * base)}"

    for pattern, (symbol, operation) in _BINARY_OPERATORS.items():
        match = re.search(rf"{_NUMBER} ?{pattern} ?{_NUMBER}", text, re.IGNORECASE)
        if match is None:
            continue
        left, right = float(match.group(1)), float(match.group(2))
        if operation == "multiply":
            value = left * right
        elif operation == "add":
            value = left + right
        elif operation == "subtract":
            value = left - right
        elif right == 0:
            return None
        else:
            value = left / right
        return f"{_format(left)} {symbol} {_format(right)} {_result(value)}"
    return None


def spoken_calculation(calculation: str) -> str:
    """'23 × 17 = 391' -> '23 times 17 is 391.'"""
    text = calculation
    if text.startswith("√"):
        text = "The square root of " + text[1:]
    for symbol, words in (("²", " squared"), ("% of", " percent of"), (" × ", " times "), (" ÷ ", " divided by "),
                          (" + ", " plus "), (" − ", " minus "), (" ≈ ", " is about "), (" = ", " is ")):
        text = text.replace(symbol, words)
    return f"{text}."
