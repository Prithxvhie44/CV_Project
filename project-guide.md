# Project Guide: Hybrid ANPR (YOLOv8 + EasyOCR + Streamlit)

## 1) Project Purpose
This project implements an end-to-end Automatic Number Plate Recognition (ANPR) workflow for vehicle images.

It does the following:
- Detects number plates using a YOLOv8 detector.
- Crops detected plate regions.
- Reads plate text using EasyOCR with multiple preprocessing views.
- Applies postprocessing for Indian plate-like normalization.
- Ranks predictions and shows results in a Streamlit UI.
- Evaluates OCR-only and end-to-end accuracy through scripts.

---

## 2) Repository Structure (Important Folders)
- `app.py`
  - Streamlit UI entry point.
- `src/anpr/`
  - Core ANPR modules:
    - `config.py`: paths/constants.
    - `data.py`: XML parsing, bbox utils, crop logic.
    - `detector.py`: YOLO detector wrapper.
    - `ocr.py`: EasyOCR wrapper + image view generation.
    - `postprocess.py`: text cleaning and Indian plate heuristics.
    - `pipeline.py`: detector + OCR orchestration.
    - `metrics.py`: similarity and exact-match metrics.
- `scripts/`
  - `prepare_dataset.py`: builds YOLO split and OCR label artifacts.
  - `train_detector.py`: trains YOLOv8 and copies best weights.
  - `evaluate.py`: OCR-only and end-to-end evaluation.
  - `calibrate_thresholds.py`: threshold tuning for detector/OCR confidence.
- `data/`
  - Processed artifacts, YOLO training structure, labels.
- `dataset/`
  - Original datasets (XML annotations and source images).
- `artifacts/detector/`
  - Training outputs, weights, and training logs.

---

## 3) Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8
- EasyOCR
- Streamlit
- Pandas / NumPy
- PyYAML / Pillow

---

## 4) End-to-End System Flow
1. Input image is loaded in Streamlit UI.
2. Detector predicts candidate plate bounding boxes.
3. Each bounding box region is cropped.
4. OCR runs on multiple transformed views of each crop.
5. Text candidates are cleaned and normalized.
6. Confidence + pattern score voting selects best plate text.
7. Predictions are ranked by combined detector/OCR confidence.
8. UI displays image overlay, table, top result, and optional GT similarity.

---

## 5) Data Preparation Flow
1. Read VOC-style XML files.
2. Parse bounding boxes and optional plate text labels.
3. Split samples into train/val/test.
4. Write YOLO image-label pairs into split directories.
5. Crop plate regions and save OCR metadata to CSV.
6. Generate `data/yolo/yolo.yaml`.

Output artifacts:
- `data/processed/labels.csv`
- `data/processed/plates/*.jpg`
- `data/yolo/images/{train,val,test}`
- `data/yolo/labels/{train,val,test}`
- `data/yolo/yolo.yaml`

---

## 6) Model Training Flow
1. Load YOLO base checkpoint (`yolov8{size}.pt`).
2. Train on prepared dataset YAML.
3. Store run logs and model weights in `artifacts/detector/`.
4. Copy best weight to `artifacts/detector/best.pt` for inference.

Key tuneable parameters:
- `epochs`
- `imgsz`
- `batch`
- `device`
- `model-size`
- `patience`

---

## 7) Evaluation Flow
### OCR-only mode
- Uses GT plate crops and compares OCR output against GT text.
- Reports:
  - Character similarity accuracy
  - Full plate exact-match accuracy

### End-to-end mode
- Runs detector + OCR on full image.
- Matches prediction to GT by IoU.
- Reports:
  - Character similarity accuracy
  - Full plate exact-match accuracy

### Threshold calibration
- Grid-search over detector confidence and OCR minimum confidence.
- Selects best configuration by full-plate accuracy, then char accuracy.

---

## 8) How to Run
Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

Prepare dataset:
```powershell
python scripts/prepare_dataset.py
```

Train detector:
```powershell
python scripts/train_detector.py --epochs 100 --imgsz 1280 --batch 8 --device 0 --model-size s --patience 30
```

Evaluate OCR-only:
```powershell
python scripts/evaluate.py --mode ocr-only
```

Evaluate end-to-end:
```powershell
python scripts/evaluate.py --mode end2end
```

Calibrate thresholds:
```powershell
python scripts/calibrate_thresholds.py --limit 200
```

Run UI:
```powershell
streamlit run app.py
```

---

## 9) What to Include in Your Flow Diagram Image
Use this as a checklist for the image you need to create.

### A) Main pipeline blocks (left to right)
- Input Vehicle Image
- Image Preprocessing (resize/color conversion)
- YOLOv8 Plate Detection
- Bounding Box Filtering (detector confidence threshold)
- Plate Crop Extraction
- OCR Preprocessing Views (gray, histogram equalization, Otsu/adaptive threshold)
- EasyOCR Inference
- Text Postprocessing (clean + normalize + Indian pattern score)
- Candidate Voting/Selection
- Ranked Final Predictions
- UI Output (bbox + text + confidence + metrics)

### B) Training branch (top or bottom lane)
- Raw Dataset (images + XML)
- Dataset Parser
- Train/Val/Test Split
- YOLO Label Writer
- Detector Training (YOLOv8)
- Best Weight Export (`best.pt`)
- Feedback arrow to Inference (model artifact reused in runtime)

### C) Evaluation branch
- OCR-only Evaluation
- End-to-End Evaluation (IoU-based match)
- Metrics:
  - Character Accuracy
  - Full Plate Accuracy
- Threshold Calibration Grid Search
- Recommended Thresholds arrow back to runtime settings

### D) Label every edge with data passed
Examples:
- `image_bgr`
- `bbox(xmin,ymin,xmax,ymax)`
- `plate_crop`
- `text candidates + confidence`
- `final_text`

### E) Include key runtime controls in a side panel
- Detector confidence
- OCR min confidence
- GPU OCR on/off
- Max results

### F) Recommended visual style
- Three swimlanes:
  - Data Preparation
  - Training/Evaluation
  - Runtime Inference/UI
- Use distinct colors:
  - Blue for data
  - Orange for models
  - Green for inference/output
- Use dashed arrows for feedback loops (calibration and retraining)

---

## 10) Suggested One-Line Caption for Diagram
"Hybrid ANPR architecture showing dataset preparation, YOLOv8 detector training, OCR-enhanced inference, and closed-loop threshold calibration."

---

## 11) Optional Add-ons for Better Presentation
- Add small icons for dataset, model, OCR, and dashboard.
- Add a legend for arrow types:
  - Solid: primary execution flow
  - Dashed: optimization/feedback loop
- Add target KPI box near output:
  - Character Accuracy
  - Full Plate Accuracy
  - Inference latency per image
