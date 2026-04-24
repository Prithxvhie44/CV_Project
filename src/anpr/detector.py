from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float


class PlateDetector:
    def __init__(self, weights_path: Path, conf: float = 0.25):
        self.conf = conf
        self.model = None
        if weights_path.exists():
            from ultralytics import YOLO

            self.model = YOLO(str(weights_path))

    def available(self) -> bool:
        return self.model is not None

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        if self.model is None:
            return []

        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, confs, strict=False):
            xmin, ymin, xmax, ymax = [int(round(v)) for v in box.tolist()]
            detections.append(Detection((xmin, ymin, xmax, ymax), float(score)))
        return detections
