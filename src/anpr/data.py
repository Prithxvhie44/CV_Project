from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class PlateObject:
    image_filename: str
    bbox: tuple[int, int, int, int]
    plate_text: str | None


def _to_int(value: str | float) -> int:
    return int(round(float(value)))


def parse_voc_xml(xml_path: Path) -> list[PlateObject]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default="")
    objects: list[PlateObject] = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = _to_int(box.findtext("xmin", "0"))
        ymin = _to_int(box.findtext("ymin", "0"))
        xmax = _to_int(box.findtext("xmax", "0"))
        ymax = _to_int(box.findtext("ymax", "0"))

        plate_text = None
        attrs = obj.find("attributes")
        if attrs is not None:
            for attr in attrs.findall("attribute"):
                if (attr.findtext("name") or "").strip() == "number_plate_text":
                    plate_text = (attr.findtext("value") or "").strip()
                    break

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append(
            PlateObject(
                image_filename=filename,
                bbox=(xmin, ymin, xmax, ymax),
                plate_text=plate_text or None,
            )
        )
    return objects


def iter_xml_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.xml"))


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(0, min(xmax, width))
    ymax = max(0, min(ymax, height))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def crop_plate(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    h, w = image.shape[:2]
    clamped = clamp_bbox(bbox, w, h)
    if clamped is None:
        return None
    xmin, ymin, xmax, ymax = clamped
    crop = image[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        return None
    return crop


def yolo_line_from_bbox(bbox: tuple[int, int, int, int], width: int, height: int, class_id: int = 0) -> str:
    xmin, ymin, xmax, ymax = bbox
    cx = ((xmin + xmax) / 2.0) / width
    cy = ((ymin + ymax) / 2.0) / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def load_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path))
    return image
