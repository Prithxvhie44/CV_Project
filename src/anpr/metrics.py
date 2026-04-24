from __future__ import annotations

from difflib import SequenceMatcher


def normalize_plate_text(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def text_similarity(pred: str, truth: str) -> float:
    pred_n = normalize_plate_text(pred)
    truth_n = normalize_plate_text(truth)
    if not pred_n and not truth_n:
        return 1.0
    if not pred_n or not truth_n:
        return 0.0
    return SequenceMatcher(None, pred_n, truth_n).ratio()


def char_accuracy(pred: str, truth: str) -> float:
    return text_similarity(pred, truth)


def exact_plate_match(pred: str, truth: str) -> bool:
    return normalize_plate_text(pred) == normalize_plate_text(truth)
