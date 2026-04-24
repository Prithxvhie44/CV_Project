from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.anpr.config import Paths


def train(epochs: int, imgsz: int, batch: int, device: str, model_size: str, patience: int) -> None:
    paths = Paths()
    data_yaml = paths.yolo_dir / "yolo.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run scripts/prepare_dataset.py first."
        )

    model_ckpt = f"yolov8{model_size}.pt"
    model = YOLO(model_ckpt)
    run = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        lr0=0.003,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.08,
        scale=0.4,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        project=str((paths.root / "artifacts").resolve()),
        name="detector",
        exist_ok=True,
    )

    best = run.save_dir / "weights" / "best.pt"
    paths.detector_weights.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        paths.detector_weights.write_bytes(best.read_bytes())
        print(f"Saved best detector weights to: {paths.detector_weights}")
    else:
        print("Training finished, but best.pt was not found in run directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO detector for number plates.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="s")
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        model_size=args.model_size,
        patience=args.patience,
    )
