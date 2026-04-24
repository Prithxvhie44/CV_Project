from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd
import yaml

from src.anpr.config import CLASS_NAME, Paths
from src.anpr.data import crop_plate, iter_xml_files, load_image, parse_voc_xml, yolo_line_from_bbox


def ensure_dirs(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    (paths.processed_dir / "plates").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (paths.yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (paths.yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def prepare(seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
    paths = Paths()
    ensure_dirs(paths)

    xml_files = list(iter_xml_files(paths.dataset_ocr_xml))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {paths.dataset_ocr_xml}")

    random.Random(seed).shuffle(xml_files)
    total = len(xml_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_map: dict[str, list[Path]] = {
        "train": xml_files[:train_end],
        "val": xml_files[train_end:val_end],
        "test": xml_files[val_end:],
    }

    crop_rows: list[dict] = []

    for split, files in split_map.items():
        for xml_path in files:
            objects = parse_voc_xml(xml_path)
            if not objects:
                continue

            image_name = objects[0].image_filename
            image_path = paths.dataset_ocr_images / image_name
            image = load_image(image_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            yolo_lines = []

            for idx, obj in enumerate(objects):
                yolo_lines.append(yolo_line_from_bbox(obj.bbox, w, h, class_id=0))

                crop = crop_plate(image, obj.bbox)
                if crop is None:
                    continue
                crop_name = f"{Path(image_name).stem}_{idx}.jpg"
                crop_path = paths.processed_dir / "plates" / crop_name
                cv2.imwrite(str(crop_path), crop)

                crop_rows.append(
                    {
                        "crop_path": str(crop_path),
                        "source_image": image_name,
                        "plate_text": obj.plate_text or "",
                        "xmin": obj.bbox[0],
                        "ymin": obj.bbox[1],
                        "xmax": obj.bbox[2],
                        "ymax": obj.bbox[3],
                    }
                )

            out_img_path = paths.yolo_dir / "images" / split / image_name
            cv2.imwrite(str(out_img_path), image)

            out_label_path = paths.yolo_dir / "labels" / split / f"{Path(image_name).stem}.txt"
            out_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    labels_csv = paths.processed_dir / "labels.csv"
    pd.DataFrame(crop_rows).to_csv(labels_csv, index=False)

    yolo_yaml = {
        "path": str(paths.yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: CLASS_NAME},
    }
    (paths.yolo_dir / "yolo.yaml").write_text(yaml.safe_dump(yolo_yaml, sort_keys=False), encoding="utf-8")

    print(f"Prepared {len(xml_files)} images for YOLO and {len(crop_rows)} OCR crops.")
    print(f"OCR labels: {labels_csv}")
    print(f"YOLO config: {paths.yolo_dir / 'yolo.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ANPR dataset for detector + OCR.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    prepare(seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
