from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anpr.config import Paths
from src.anpr.data import iter_xml_files, load_image, parse_voc_xml
from src.anpr.metrics import char_accuracy, exact_plate_match
from src.anpr.pipeline import ANPRPipeline


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def eval_config(det_conf: float, ocr_min_conf: float, limit: int | None) -> dict:
    paths = Paths()
    pipe = ANPRPipeline(paths.detector_weights, detector_conf=det_conf, use_gpu_ocr=False, ocr_min_conf=ocr_min_conf)

    rows = []
    for xml in iter_xml_files(paths.dataset_ocr_xml):
        objects = parse_voc_xml(xml)
        if not objects:
            continue

        image = load_image(paths.dataset_ocr_images / objects[0].image_filename)
        if image is None:
            continue

        preds = pipe.predict(image)
        for gt in objects:
            if not gt.plate_text:
                continue
            best_text = ""
            best_iou = 0.0
            for pred in preds:
                ov = iou(pred.bbox, gt.bbox)
                if ov > best_iou:
                    best_iou = ov
                    best_text = pred.text
            if best_iou < 0.1:
                best_text = ""

            rows.append((best_text, gt.plate_text))
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break

    if not rows:
        return {
            "det_conf": det_conf,
            "ocr_min_conf": ocr_min_conf,
            "samples": 0,
            "char_accuracy": 0.0,
            "full_plate_accuracy": 0.0,
        }

    char_acc = sum(char_accuracy(p, t) for p, t in rows) / len(rows)
    full_acc = sum(1 for p, t in rows if exact_plate_match(p, t)) / len(rows)
    return {
        "det_conf": det_conf,
        "ocr_min_conf": ocr_min_conf,
        "samples": len(rows),
        "char_accuracy": char_acc,
        "full_plate_accuracy": full_acc,
    }


def calibrate(limit: int | None = None) -> None:
    det_grid = [0.2, 0.25, 0.3, 0.35, 0.4]
    ocr_grid = [0.2, 0.3, 0.4, 0.5]

    results = [eval_config(d, o, limit=limit) for d, o in product(det_grid, ocr_grid)]
    best = max(results, key=lambda r: (r["full_plate_accuracy"], r["char_accuracy"]))

    print(json.dumps({"best": best, "all": results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate detector/OCR thresholds for ANPR.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    calibrate(limit=args.limit)
