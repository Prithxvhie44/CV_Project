from __future__ import annotations

import re

from .config import ALLOWED_CHARS

# Common Indian private/commercial formats without separators.
INDIAN_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$"),
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$"),
]

LETTER_LIKE = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
DIGIT_LIKE = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}


def clean_text(raw: str) -> str:
    return "".join(ch for ch in raw.upper() if ch in ALLOWED_CHARS)


def indian_pattern_score(text: str) -> float:
    t = clean_text(text)
    if not t:
        return 0.0

    if any(p.match(t) for p in INDIAN_PATTERNS):
        return 1.0

    score = 0.0
    if len(t) >= 2 and t[0].isalpha() and t[1].isalpha():
        score += 0.35
    if len(t) >= 4 and t[2].isdigit() and t[3].isdigit():
        score += 0.35
    if len(t) >= 6 and any(ch.isalpha() for ch in t[4:-1]):
        score += 0.15
    if len(t) >= 6 and t[-1].isdigit():
        score += 0.15
    return min(score, 0.99)


def normalize_indian_plate(text: str) -> str:
    t = clean_text(text)
    if len(t) < 6:
        return t

    chars = list(t)

    def force_letter(i: int) -> None:
        if 0 <= i < len(chars):
            chars[i] = LETTER_LIKE.get(chars[i], chars[i])

    def force_digit(i: int) -> None:
        if 0 <= i < len(chars):
            chars[i] = DIGIT_LIKE.get(chars[i], chars[i])

    # Prefix: state + district expected as LLDD.
    force_letter(0)
    force_letter(1)
    force_digit(2)
    force_digit(3)

    # Suffix in most formats is numeric tail.
    for i in range(max(0, len(chars) - 4), len(chars)):
        force_digit(i)

    return clean_text("".join(chars))
