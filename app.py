from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.anpr.config import Paths
from src.anpr.metrics import exact_plate_match, text_similarity
from src.anpr.pipeline import ANPRPipeline

st.set_page_config(page_title="ANPR High Accuracy UI", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 45%, #0ea5e9 100%);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
        color: #f8fafc;
    }
    .hero h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .hero p {
        margin-top: 6px;
        opacity: 0.95;
    }
    .small-note {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Hybrid ANPR: Detection + OCR</h1>
        <p>Upload a vehicle image to detect number plates and read text with confidence scores.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

paths = Paths()
weights_path = paths.detector_weights

with st.sidebar:
    st.header("Inference Settings")
    conf = st.slider("Detector confidence", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
    ocr_min_conf = st.slider("OCR min confidence", min_value=0.0, max_value=0.95, value=0.2, step=0.05)
    use_gpu_ocr = st.checkbox("Use GPU for OCR (if available)", value=False)
    max_results = st.slider("Max results", min_value=1, max_value=10, value=5, step=1)
    gt_text = st.text_input("Optional ground truth plate", value="")
    st.markdown("<span class='small-note'>Tip: Start with detector 0.25 and OCR 0.20, then tune based on validation.</span>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def build_pipeline(weights: Path, detector_conf: float, use_gpu_ocr: bool, ocr_min_conf: float) -> ANPRPipeline:
    return ANPRPipeline(
        weights,
        detector_conf=detector_conf,
        use_gpu_ocr=use_gpu_ocr,
        ocr_min_conf=ocr_min_conf,
    )

if not weights_path.exists():
    st.warning(
        "Detector weights not found. Train detector first using scripts/train_detector.py. "
        f"Expected: {weights_path}"
    )

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Running detector + OCR..."):
        pipeline = build_pipeline(
            weights_path,
            detector_conf=conf,
            use_gpu_ocr=use_gpu_ocr,
            ocr_min_conf=ocr_min_conf,
        )
    preds = pipeline.predict(bgr)[:max_results]
    vis_bgr = pipeline.draw_predictions(bgr, preds)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    summary_tab, details_tab = st.tabs(["Result", "Details"])

    col1, col2 = summary_tab.columns([1.7, 1.3])
    with col1:
        st.image(vis_rgb, caption="Predictions", use_container_width=True)

    with col2:
        if preds:
            rows = [
                {
                    "plate_text": p.text,
                    "det_conf": round(p.det_confidence, 4),
                    "ocr_conf": round(p.ocr_confidence, 4),
                    "bbox": p.bbox,
                }
                for p in preds
            ]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            best = preds[0]
            st.success(f"Best plate: {best.text or 'UNKNOWN'}")
            k1, k2 = st.columns(2)
            with k1:
                st.metric("Detector confidence", f"{best.det_confidence:.2f}")
            with k2:
                st.metric("OCR confidence", f"{best.ocr_confidence:.2f}")

            if gt_text.strip():
                sim = text_similarity(best.text, gt_text)
                exact = exact_plate_match(best.text, gt_text)
                st.metric("Similarity", f"{sim * 100:.2f}%")
                st.write("Exact match:", "Yes" if exact else "No")
        else:
            st.error("No plate detected.")

    with details_tab:
        st.markdown("### Processing Notes")
        st.markdown(
            "- Detector proposes plate regions using YOLOv8.\n"
            "- OCR runs on multiple enhanced plate views.\n"
            "- Postprocessing normalizes candidate text and ranks results by confidence."
        )
else:
    st.info("Upload an image to start inference.")

st.markdown("---")
st.markdown(
    "**Accuracy target (80-90%) tips:** train detector with more epochs, add harder augmentations, and evaluate with scripts/evaluate.py."
)
