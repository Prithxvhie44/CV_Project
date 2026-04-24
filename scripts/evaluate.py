from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.anpr.config import Paths
from src.anpr.data import crop_plate, iter_xml_files, load_image, parse_voc_xml
from src.anpr.metrics import char_accuracy, exact_plate_match
from src.anpr.ocr import PlateOCR
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

    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_ocr_only(limit: int | None = None) -> dict:
    paths = Paths()
    ocr = PlateOCR(gpu=False)

    pairs = []
    for xml in iter_xml_files(paths.dataset_ocr_xml):
        objects = parse_voc_xml(xml)
        if not objects:
            continue
        image = load_image(paths.dataset_ocr_images / objects[0].image_filename)
        if image is None:
            continue

        for obj in objects:
            if not obj.plate_text:
                continue
            crop = crop_plate(image, obj.bbox)
            if crop is None:
                continue
            pred = ocr.predict(crop).text
            truth = obj.plate_text
            pairs.append((pred, truth))
            if limit and len(pairs) >= limit:
                break
        if limit and len(pairs) >= limit:
            break

    if not pairs:
        return {"samples": 0, "char_accuracy": 0.0, "full_plate_accuracy": 0.0}

    char_acc = sum(char_accuracy(p, t) for p, t in pairs) / len(pairs)
    full_acc = sum(1 for p, t in pairs if exact_plate_match(p, t)) / len(pairs)
    return {"samples": len(pairs), "char_accuracy": char_acc, "full_plate_accuracy": full_acc}


def evaluate_end_to_end(
    limit: int | None = None,
    detector_conf: float = 0.25,
    ocr_min_conf: float = 0.0,
) -> dict:
    paths = Paths()
    pipe = ANPRPipeline(paths.detector_weights, detector_conf=detector_conf, ocr_min_conf=ocr_min_conf)
    if not pipe.detector.available():
        raise FileNotFoundError(
            f"Detector weights not found at {paths.detector_weights}. Train detector first."
        )

    rows = []
    for xml in iter_xml_files(paths.dataset_ocr_xml):
        objects = parse_voc_xml(xml)
        if not objects:
            continue

        image_name = objects[0].image_filename
        image = load_image(paths.dataset_ocr_images / image_name)
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

            # If no box overlaps, count as miss.
            if best_iou < 0.1:
                best_text = ""

            rows.append({
                "image": image_name,
                "truth": gt.plate_text,
                "pred": best_text,
                "char_acc": char_accuracy(best_text, gt.plate_text),
                "exact": exact_plate_match(best_text, gt.plate_text),
            })
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break

    if not rows:
        return {"samples": 0, "char_accuracy": 0.0, "full_plate_accuracy": 0.0}

    df = pd.DataFrame(rows)
    return {
        "samples": int(len(df)),
        "char_accuracy": float(df["char_acc"].mean()),
        "full_plate_accuracy": float(df["exact"].mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ANPR accuracy.")
    parser.add_argument("--mode", choices=["ocr-only", "end2end"], default="ocr-only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--ocr-min-conf", type=float, default=0.0)
    args = parser.parse_args()

    if args.mode == "ocr-only":
        result = evaluate_ocr_only(limit=args.limit)
    else:
        result = evaluate_end_to_end(
            limit=args.limit,
            detector_conf=args.detector_conf,
            ocr_min_conf=args.ocr_min_conf,
        )

    print(json.dumps(result, indent=2))
