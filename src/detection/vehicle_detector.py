"""
src/detection/vehicle_detector.py
YOLOv8-based multi-class vehicle detector.
Detects: bicycle, motorcycle, auto-rickshaw (custom), car, bus, truck.
Also runs secondary sub-classifiers on crops for helmet / seatbelt / person count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Detection Dataclass
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Single object detection result."""
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    class_id:    int
    class_name:  str
    confidence:  float
    centroid:    Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Extract bounding-box sub-image from frame."""
        x1, y1, x2, y2 = self.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]


# ══════════════════════════════════════════════════════════════════════
# Vehicle Detector
# ══════════════════════════════════════════════════════════════════════

class VehicleDetector:
    """
    Wraps ultralytics YOLOv8 for multi-class vehicle detection.

    On first import, ultralytics will auto-download yolov8n.pt if not present.
    Swap model_path in config for larger models (yolov8s, yolov8m, ...).
    """

    # Human-readable class names from config keys → display strings
    CLASS_DISPLAY: Dict[int, str] = {}

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 'detection' section from settings.yaml.
        """
        self._conf_thresh  = cfg.get("confidence_threshold", 0.40)
        self._iou_thresh   = cfg.get("iou_threshold", 0.45)
        self._device       = cfg.get("device", "cpu")
        self._model_path   = cfg.get("model_path", "yolov8n.pt")
        self._person_id    = cfg.get("person_class_id", 0)

        # Vehicle class IDs from config
        vc = cfg.get("vehicle_classes", {})
        self._vehicle_class_ids = set(vc.values())

        # Two-wheeler / three-wheeler / heavy for rule checks
        self._two_wheeler_ids   = set(cfg.get("two_wheeler_ids", [1, 3]))
        self._three_wheeler_ids = set(cfg.get("three_wheeler_ids", []))
        self._heavy_ids         = set(cfg.get("heavy_vehicle_ids", [5, 7]))

        # Build display name dict
        for name, cid in vc.items():
            VehicleDetector.CLASS_DISPLAY[cid] = name.replace("_", " ").title()
        VehicleDetector.CLASS_DISPLAY[self._person_id] = "Person"

        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8 model (downloads automatically on first run)."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info(
                f"VehicleDetector: loaded '{self._model_path}' "
                f"on device='{self._device}'"
            )
        except Exception as e:
            logger.error(f"VehicleDetector: failed to load model — {e}")
            raise

    # ── Helpers ───────────────────────────────────────────────────────

    def _parse_results(self, results) -> List[Detection]:
        """Parse ultralytics Results objects into Detection list."""
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid   = int(box.cls[0])
                conf  = float(box.conf[0])
                xyxy  = box.xyxy[0].cpu().numpy().astype(int)
                bbox  = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                name  = VehicleDetector.CLASS_DISPLAY.get(cid, f"class_{cid}")
                detections.append(Detection(
                    bbox=bbox,
                    class_id=cid,
                    class_name=name,
                    confidence=conf,
                ))
        return detections

    # ── Public API ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a BGR frame.

        Returns all detected vehicles and persons above confidence threshold.

        Args:
            frame: Preprocessed BGR uint8 image.

        Returns:
            List of Detection objects (vehicles + persons).
        """
        if self._model is None or frame is None or frame.size == 0:
            return []

        try:
            results = self._model.predict(
                source=frame,
                conf=self._conf_thresh,
                iou=self._iou_thresh,
                device=self._device,
                verbose=False,
            )
            all_dets = self._parse_results(results)

            # Keep only vehicle + person classes
            kept = [
                d for d in all_dets
                if d.class_id in self._vehicle_class_ids
                or d.class_id == self._person_id
            ]
            return kept

        except Exception as e:
            logger.error(f"VehicleDetector.detect failed: {e}")
            return []

    def detect_vehicles_only(self, frame: np.ndarray) -> List[Detection]:
        """Return only vehicle detections (no persons)."""
        return [d for d in self.detect(frame) if d.class_id != self._person_id]

    def detect_persons_only(self, frame: np.ndarray) -> List[Detection]:
        """Return only person detections."""
        return [d for d in self.detect(frame) if d.class_id == self._person_id]

    def persons_inside_bbox(
        self,
        persons: List[Detection],
        vehicle_bbox: Tuple[int, int, int, int],
        iou_threshold: float = 0.3,
    ) -> List[Detection]:
        """
        Return persons whose bounding box is mostly inside the vehicle bounding box.
        Used for triple-riding detection.

        Args:
            persons:       List of person Detection objects.
            vehicle_bbox:  (x1, y1, x2, y2) of the vehicle.
            iou_threshold: Minimum overlap to count (centroid check + IoU).

        Returns:
            Filtered list of persons that are inside the vehicle bbox.
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        # Minimal padding (5%) horizontally to avoid grabbing pedestrians next to the bike.
        # Moderate padding (10%) vertically for tall riders.
        w_pad = (vx2 - vx1) * 0.05
        h_pad = (vy2 - vy1) * 0.10
        bx1, by1 = vx1 - w_pad, vy1 - h_pad
        bx2, by2 = vx2 + w_pad, vy2 + h_pad

        inside = []
        for p in persons:
            # Require higher confidence for triple riding to avoid hallucinated people
            if p.confidence < 0.60:
                continue
                
            px1, py1, px2, py2 = p.bbox
            person_area = max(0, px2 - px1) * max(0, py2 - py1)
            if person_area == 0:
                continue

            # Intersection of person with padded vehicle bbox
            ix1 = max(bx1, px1)
            iy1 = max(by1, py1)
            ix2 = min(bx2, px2)
            iy2 = min(by2, py2)
            
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            
            # A person is "on" the bike if at least 60% of their body is inside 
            # the bike's bounding box OR their centroid is strictly inside 
            # and they have moderate intersection.
            overlap_ratio = inter_area / person_area
            
            cx, cy = p.centroid
            centroid_inside = (bx1 <= cx <= bx2 and by1 <= cy <= by2)
            
            if overlap_ratio > 0.60 or (centroid_inside and overlap_ratio > 0.40):
                inside.append(p)

        return inside

    # ── Sub-classifiers (OpenCV heuristics) ─────────────────────────

    def classify_helmet(self, head_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Classify whether a rider is wearing a helmet using HSV color analysis.

        Algorithm:
          1. Convert head crop to HSV.
          2. Exclude skin-tone pixels (hue 0–25 OR 160–180, high saturation).
          3. Measure non-skin coverage ratio over the crop.
          4. High non-skin coverage + large contiguous region → helmet present.

        Accuracy note: Heuristic — works well for solid-coloured helmets.
        Replace with a fine-tuned YOLOv8-cls model for production accuracy.
        Return: (has_helmet: bool, confidence: float 0–1)
        """
        if head_crop is None or head_crop.size == 0:
            return True, 0.0   # unknown → conservative (assume helmet)

        try:
            # Resize to a fixed height for speed
            h, w = head_crop.shape[:2]
            if h < 5 or w < 5:
                return True, 0.0
            scale = 64 / max(h, 1)
            resized = cv2.resize(
                head_crop,
                (max(1, int(w * scale)), 64),
                interpolation=cv2.INTER_AREA,
            )

            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            total_px = resized.shape[0] * resized.shape[1]

            # Skin-tone mask: hue 0-20 or 160-180 with moderate saturation
            skin_lo1 = np.array([0,   40,  60],  dtype=np.uint8)
            skin_hi1 = np.array([20,  255, 255], dtype=np.uint8)
            skin_lo2 = np.array([160, 40,  60],  dtype=np.uint8)
            skin_hi2 = np.array([180, 255, 255], dtype=np.uint8)
            skin_mask = (
                cv2.inRange(hsv, skin_lo1, skin_hi1)
                | cv2.inRange(hsv, skin_lo2, skin_hi2)
            )

            # Black/dark pixels (common helmet colors)
            dark_mask = cv2.inRange(hsv,
                np.array([0, 0, 0],   dtype=np.uint8),
                np.array([180, 80, 80], dtype=np.uint8),
            )

            non_skin_px = total_px - int(np.count_nonzero(skin_mask))
            dark_px     = int(np.count_nonzero(dark_mask))
            non_skin_ratio = non_skin_px / max(total_px, 1)
            dark_ratio     = dark_px     / max(total_px, 1)

            # Find the largest non-skin contiguous region
            non_skin_img = cv2.bitwise_not(skin_mask)
            contours, _  = cv2.findContours(
                non_skin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            largest_area = max((cv2.contourArea(c) for c in contours), default=0)
            largest_ratio = largest_area / max(total_px, 1)

            # Helmet heuristic:
            #  - Non-skin covers most of the crop  (non_skin_ratio > 0.55)
            #  - A single large region dominates  (largest_ratio  > 0.40)
            has_helmet   = (non_skin_ratio > 0.55) and (largest_ratio > 0.40)
            # Confidence: blend of how strongly the two conditions are met
            conf = float(np.clip(
                (non_skin_ratio * 0.6 + largest_ratio * 0.4) * 1.2, 0.0, 0.98
            ))
            # Dark helmets boost confidence
            if dark_ratio > 0.35:
                conf = min(conf + 0.08, 0.98)

            return has_helmet, round(conf, 3)

        except Exception as e:
            logger.debug(f"classify_helmet exception: {e}")
            return True, 0.0   # fail-safe: assume helmet

    def classify_seatbelt(self, driver_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Classify seatbelt presence using diagonal-band detection.

        Algorithm:
          1. Convert to grayscale + Canny edges.
          2. Run probabilistic Hough transform to find line segments.
          3. A seatbelt appears as a strong diagonal line in the
             driver's torso region (angle 30°–60° from horizontal).
          4. Score by number and length of matching diagonal segments.

        Accuracy note: Works best on front-facing camera views.
        Replace with a fine-tuned YOLOv8-cls model for production accuracy.
        Return: (has_seatbelt: bool, confidence: float 0–1)
        """
        if driver_crop is None or driver_crop.size == 0:
            return True, 0.0   # unknown → conservative

        try:
            h, w = driver_crop.shape[:2]
            if h < 10 or w < 10:
                return True, 0.0

            # Resize to fixed height
            scale   = 80 / max(h, 1)
            resized = cv2.resize(
                driver_crop,
                (max(1, int(w * scale)), 80),
                interpolation=cv2.INTER_AREA,
            )
            gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
            edges    = cv2.Canny(filtered, 30, 100)

            rh, rw = resized.shape[:2]
            # Hough lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=15,
                minLineLength=int(rh * 0.25),
                maxLineGap=int(rh * 0.15),
            )

            if lines is None:
                return False, 0.62   # no lines → likely no seatbelt

            diagonal_score = 0.0
            total_len      = 0.0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                seg_len = math.sqrt(dx * dx + dy * dy)
                if seg_len < 5:
                    continue
                # Angle from horizontal (degrees)
                angle = abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6)))
                # Seatbelt diagonal: 20° to 70° from horizontal
                if 20 <= angle <= 70:
                    # Weight by length relative to crop diagonal
                    weight        = seg_len / max(math.sqrt(rh**2 + rw**2), 1)
                    diagonal_score += weight
                    total_len     += seg_len

            # Normalise score
            crop_diag    = math.sqrt(rh ** 2 + rw ** 2)
            norm_score   = float(np.clip(diagonal_score * 2.5, 0.0, 1.0))
            has_seatbelt = norm_score > 0.35   # threshold
            conf         = round(norm_score * 0.90 + 0.05, 3)   # range ~[0.05, 0.95]

            return has_seatbelt, float(np.clip(conf, 0.0, 0.98))

        except Exception as e:
            logger.debug(f"classify_seatbelt exception: {e}")
            return True, 0.0   # fail-safe: assume seatbelt

    def classify_phone(self, driver_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Classify whether driver is holding a phone.

        ┌─────────────────────────────────────────────────────────────┐
        │  SWAP POINT — Fine-tuned Phone Detector                     │
        │  Replace with YOLOv8-cls trained on phone-use data.         │
        └─────────────────────────────────────────────────────────────┘

        Current: returns not-detected (phone use is disabled by default).
        """
        if driver_crop is None or driver_crop.size == 0:
            return False, 0.0
        return False, 0.3  # feature-flagged off in config by default

    # ── Class Group Helpers ───────────────────────────────────────────

    def is_two_wheeler(self, class_id: int) -> bool:
        return class_id in self._two_wheeler_ids

    def is_three_wheeler(self, class_id: int) -> bool:
        return class_id in self._three_wheeler_ids

    def is_heavy(self, class_id: int) -> bool:
        return class_id in self._heavy_ids

    def is_restricted_vehicle(self, class_id: int) -> bool:
        """Two- or three-wheelers are restricted on highways."""
        return self.is_two_wheeler(class_id) or self.is_three_wheeler(class_id)
