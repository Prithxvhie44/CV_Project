from __future__ import annotations

from dataclasses import dataclass

import cv2
import easyocr
import numpy as np

from .config import ALLOWED_CHARS
from .postprocess import clean_text, indian_pattern_score, normalize_indian_plate


@dataclass
class OCRResult:
    text: str
    confidence: float


def _clean_text(raw: str) -> str:
    return clean_text(raw)


def _candidate_views(plate_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 75, 75)
    eq = cv2.equalizeHist(denoised)
    up = cv2.resize(eq, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(
        up,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    inv = cv2.bitwise_not(otsu)
    return [up, otsu, adapt, inv]


def preprocess_plate_image(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 75, 75)
    eq = cv2.equalizeHist(denoised)
    resized = cv2.resize(eq, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


class PlateOCR:
    def __init__(self, gpu: bool = False):
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def predict(self, plate_bgr: np.ndarray) -> OCRResult:
        candidates: list[OCRResult] = []
        for view in _candidate_views(plate_bgr):
            results = self.reader.readtext(view, allowlist=ALLOWED_CHARS, detail=1, paragraph=False)
            if not results:
                continue
            for item in results:
                raw = str(item[1])
                conf = float(item[2])
                cleaned = _clean_text(raw)
                if not cleaned:
                    continue
                normalized = normalize_indian_plate(cleaned)
                candidates.append(OCRResult(text=normalized, confidence=conf))

        if not candidates:
            return OCRResult(text="", confidence=0.0)

        # Vote by text using summed confidence and format-likelihood bonus.
        score_by_text: dict[str, float] = {}
        best_conf_by_text: dict[str, float] = {}
        for c in candidates:
            pattern_bonus = 0.15 * indian_pattern_score(c.text)
            score_by_text[c.text] = score_by_text.get(c.text, 0.0) + c.confidence + pattern_bonus
            best_conf_by_text[c.text] = max(best_conf_by_text.get(c.text, 0.0), c.confidence)

        best_text = max(score_by_text.items(), key=lambda kv: kv[1])[0]
        return OCRResult(text=best_text, confidence=best_conf_by_text.get(best_text, 0.0))
