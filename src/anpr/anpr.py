"""
src/anpr/anpr.py
Automatic Number Plate Recognition pipeline:
  Step 1 — YOLO detects plate bounding box
  Step 2 — Crop + grayscale + bilateral filter + adaptive threshold
  Step 3 — EasyOCR reads plate text
  Step 4 — 3-tier confidence gate:
            >= 0.90  → auto-accept (ticket issued)
            0.50-0.90→ low_confidence (manual review)
            < 0.50   → discard

Handles multi-line Indian plates by joining text blocks
left-to-right, top-to-bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Compile once — matches Indian plate patterns (e.g. MH12AB1234)
_PLATE_PATTERN = re.compile(r"[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,2}[\s\-]?\d{4}")


# ══════════════════════════════════════════════════════════════════════
# Result Dataclass
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PlateResult:
    """
    Output of ANPRSystem.recognize().

    status values:
      "accepted"       — conf >= auto_accept_threshold  → ticket can be issued
      "low_confidence" — conf in [manual_review, auto_accept) → manual review
      "discarded"      — conf < manual_review_threshold → unusable
      "not_detected"   — no plate found in frame
    """
    text:       str
    confidence: float
    bbox:       Optional[Tuple[int, int, int, int]]
    status:     str   # "accepted" | "low_confidence" | "discarded" | "not_detected"

    @property
    def is_usable(self) -> bool:
        return self.status in ("accepted", "low_confidence")

    @property
    def needs_review(self) -> bool:
        return self.status == "low_confidence"


# ══════════════════════════════════════════════════════════════════════
# ANPR System
# ══════════════════════════════════════════════════════════════════════

class ANPRSystem:
    """
    Two-stage ANPR: YOLO plate detection → EasyOCR.

    Instantiate once; call recognize() per violation frame.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 'anpr' section from settings.yaml.
        """
        self._model_path    = cfg.get("model_path", "yolov8n.pt")
        self._languages     = cfg.get("ocr_languages", ["en"])
        self._auto_conf     = cfg.get("auto_accept_confidence", 0.90)
        self._review_conf   = cfg.get("manual_review_confidence", 0.50)

        self._yolo          = None
        self._reader        = None

        self._init_yolo()
        self._init_ocr()

    # ── Initialisers ─────────────────────────────────────────────────

    def _init_yolo(self) -> None:
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self._model_path)
            logger.info(f"ANPR: YOLO plate detector loaded ('{self._model_path}').")
        except Exception as e:
            logger.error(f"ANPR: YOLO init failed — {e}. Plate detection disabled.")

    def _init_ocr(self) -> None:
        try:
            import easyocr
            self._reader = easyocr.Reader(
                self._languages,
                gpu=False,   # set True if CUDA available
                verbose=False,
            )
            logger.info(f"ANPR: EasyOCR initialised ({self._languages}).")
        except Exception as e:
            logger.error(f"ANPR: EasyOCR init failed — {e}. OCR disabled.")

    # ── Step 1: Plate Detection ───────────────────────────────────────

    def _detect_plate_bbox(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Run YOLO to find the license plate bounding box.

        Args:
            frame: BGR vehicle crop or full frame.

        Returns:
            (x1, y1, x2, y2) of highest-confidence plate detection, or None.
        """
        if self._yolo is None or frame is None or frame.size == 0:
            return None

        try:
            results = self._yolo.predict(
                source=frame,
                conf=0.30,
                verbose=False,
            )
            best_conf = 0.0
            best_bbox = None
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        best_bbox = (int(xyxy[0]), int(xyxy[1]),
                                     int(xyxy[2]), int(xyxy[3]))
            return best_bbox
        except Exception as e:
            logger.warning(f"ANPR plate detection error: {e}")
            return None

    # ── Step 2: Plate Preprocessing ──────────────────────────────────

    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Prepare plate crop for OCR:
          - Resize to fixed height (64 px), preserve aspect ratio
          - Convert to grayscale
          - Bilateral filter (preserves edges / characters)
          - Adaptive threshold (works under varied lighting)

        Args:
            plate_img: BGR crop of the plate region.

        Returns:
            Binary (thresholded) grayscale image for OCR.
        """
        if plate_img is None or plate_img.size == 0:
            return plate_img

        # Resize to height=64
        h, w = plate_img.shape[:2]
        if h == 0:
            return plate_img
        target_h = 64
        scale    = target_h / h
        new_w    = max(1, int(w * scale))
        resized  = cv2.resize(plate_img, (new_w, target_h),
                              interpolation=cv2.INTER_LINEAR)

        # Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Bilateral filter (denoise while preserving plate characters)
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Adaptive thresholding for robustness under different lighting
        binary = cv2.adaptiveThreshold(
            bilateral, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )
        return binary

    # ── Step 3: OCR ───────────────────────────────────────────────────

    def _run_ocr(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Run EasyOCR on preprocessed plate image.
        Joins multiple detected text blocks left-to-right, top-to-bottom
        (handles multi-line Indian plates).

        Args:
            plate_img: Preprocessed (binary) plate image.

        Returns:
            (plate_text, mean_confidence)
        """
        if self._reader is None or plate_img is None or plate_img.size == 0:
            return "", 0.0

        try:
            # EasyOCR expects grayscale or RGB — use binary as-is (1 channel)
            results = self._reader.readtext(plate_img, detail=1)

            if not results:
                return "", 0.0

            # Sort by top-left y then x (top-to-bottom, left-to-right)
            results_sorted = sorted(
                results,
                key=lambda r: (r[0][0][1], r[0][0][0])
            )

            texts  = [r[1].strip().upper() for r in results_sorted]
            confs  = [float(r[2]) for r in results_sorted]

            # Clean and join
            raw_text  = " ".join(texts).replace(" ", "")
            mean_conf = sum(confs) / len(confs) if confs else 0.0

            # Basic sanitisation: keep only alphanumeric
            clean_text = re.sub(r"[^A-Z0-9]", "", raw_text)

            return clean_text, mean_conf

        except Exception as e:
            logger.warning(f"ANPR OCR error: {e}")
            return "", 0.0

    # ── Step 4: Confidence Gate ───────────────────────────────────────

    def _gate(self, text: str, confidence: float) -> str:
        """
        Apply 3-tier confidence gate:
          >= auto_accept_confidence  → "accepted"
          >= manual_review_confidence → "low_confidence"
          else                       → "discarded"
        """
        if not text:
            return "discarded"
        if confidence >= self._auto_conf:
            return "accepted"
        if confidence >= self._review_conf:
            return "low_confidence"
        return "discarded"

    # ── Public API ────────────────────────────────────────────────────

    def recognize(
        self,
        frame: np.ndarray,
        vehicle_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> PlateResult:
        """
        Full ANPR pipeline on one frame.

        Args:
            frame:        Full BGR frame or pre-cropped vehicle region.
            vehicle_bbox: If provided, crops to vehicle region before
                          plate detection (improves accuracy).

        Returns:
            PlateResult with text, confidence, bbox, status.
        """
        if frame is None or frame.size == 0:
            return PlateResult("", 0.0, None, "not_detected")

        # Crop to vehicle region if bbox provided
        work_frame = frame
        if vehicle_bbox is not None:
            h, w = frame.shape[:2]
            x1  = max(0, vehicle_bbox[0])
            y1  = max(0, vehicle_bbox[1])
            x2  = min(w, vehicle_bbox[2])
            y2  = min(h, vehicle_bbox[3])
            if x2 > x1 and y2 > y1:
                work_frame = frame[y1:y2, x1:x2]

        # Step 1 — Detect plate bbox
        plate_bbox = self._detect_plate_bbox(work_frame)

        if plate_bbox is None:
            # No plate found — try OCR directly on the lower portion of crop
            h, w = work_frame.shape[:2]
            fallback = work_frame[int(h * 0.6):h, :]
            text, conf = self._run_ocr(self._preprocess_plate(fallback))
            status = self._gate(text, conf)
            if status == "discarded":
                logger.debug("ANPR: no plate detected and fallback OCR discarded.")
                return PlateResult("", 0.0, None, "not_detected")
            logger.debug(f"ANPR: fallback OCR → '{text}' conf={conf:.2f} [{status}]")
            return PlateResult(text, conf, None, status)

        # Step 2 — Crop + preprocess
        px1, py1, px2, py2 = plate_bbox
        h, w = work_frame.shape[:2]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        plate_crop = work_frame[py1:py2, px1:px2]

        preprocessed = self._preprocess_plate(plate_crop)

        # Step 3 — OCR
        text, conf = self._run_ocr(preprocessed)

        # Step 4 — Gate
        status = self._gate(text, conf)

        if status == "discarded":
            logger.debug(f"ANPR: discarded '{text}' conf={conf:.2f}")
        else:
            logger.info(f"ANPR: '{text}' conf={conf:.2f} [{status}]")

        # Adjust plate bbox back to full-frame coords if we cropped
        abs_bbox = plate_bbox
        if vehicle_bbox is not None:
            vx1, vy1 = vehicle_bbox[0], vehicle_bbox[1]
            abs_bbox = (
                px1 + vx1, py1 + vy1,
                px2 + vx1, py2 + vy1,
            )

        return PlateResult(
            text=text,
            confidence=conf,
            bbox=abs_bbox,
            status=status,
        )

    def recognize_batch(
        self,
        frames_and_bboxes: List[Tuple[np.ndarray, Optional[Tuple]]],
    ) -> List[PlateResult]:
        """
        Run recognition on multiple frames (e.g. batch pipeline).

        Args:
            frames_and_bboxes: List of (frame, vehicle_bbox) tuples.

        Returns:
            List of PlateResult in same order.
        """
        return [self.recognize(f, b) for f, b in frames_and_bboxes]
