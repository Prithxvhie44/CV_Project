from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    dataset_ocr_images: Path = root / "dataset" / "number_plate_images_ocr" / "number_plate_images_ocr"
    dataset_ocr_xml: Path = root / "dataset" / "number_plate_annos_ocr" / "number_plate_annos_ocr"
    dataset_extra_xml: Path = root / "dataset" / "Annotations" / "Annotations"

    processed_dir: Path = root / "data" / "processed"
    yolo_dir: Path = root / "data" / "yolo"

    detector_weights: Path = root / "artifacts" / "detector" / "best.pt"


CLASS_NAME = "number_plate"
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
