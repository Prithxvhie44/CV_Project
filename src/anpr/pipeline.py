from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import crop_plate
from .detector import PlateDetector
from .ocr import PlateOCR


@dataclass
class PlatePrediction:
    bbox: tuple[int, int, int, int]
    det_confidence: float
    text: str
    ocr_confidence: float


class ANPRPipeline:
    def __init__(
        self,
        detector_weights: Path,
        detector_conf: float = 0.25,
        use_gpu_ocr: bool = False,
        ocr_min_conf: float = 0.0,
    ):
        self.detector = PlateDetector(detector_weights, conf=detector_conf)
        self.ocr = PlateOCR(gpu=use_gpu_ocr)
        self.ocr_min_conf = ocr_min_conf

    def predict(self, image_bgr: np.ndarray) -> list[PlatePrediction]:
        detections = self.detector.predict(image_bgr)
        predictions: list[PlatePrediction] = []

        for det in detections:
            crop = crop_plate(image_bgr, det.bbox)
            if crop is None:
                continue
            ocr_res = self.ocr.predict(crop)
            if ocr_res.confidence < self.ocr_min_conf:
                continue
            predictions.append(
                PlatePrediction(
                    bbox=det.bbox,
                    det_confidence=det.confidence,
                    text=ocr_res.text,
                    ocr_confidence=ocr_res.confidence,
                )
            )

        predictions.sort(
            key=lambda p: (0.7 * p.det_confidence) + (0.3 * p.ocr_confidence),
            reverse=True,
        )
        return predictions

    @staticmethod
    def draw_predictions(image_bgr: np.ndarray, preds: list[PlatePrediction]) -> np.ndarray:
        out = image_bgr.copy()
        for pred in preds:
            xmin, ymin, xmax, ymax = pred.bbox
            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{pred.text or 'UNKNOWN'} | D:{pred.det_confidence:.2f} O:{pred.ocr_confidence:.2f}"
            cv2.putText(out, label, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return out
