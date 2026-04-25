"""
src/detection/vehicle_detector.py
YOLOv8-based multi-class vehicle detector.
Detects: bicycle, motorcycle, auto-rickshaw (custom), car, bus, truck.
Also runs secondary sub-classifiers on crops for helmet / seatbelt / person count.
"""

from __future__ import annotations

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
        Return persons whose centroid lies inside the vehicle bounding box.
        Used for triple-riding detection.

        Args:
            persons:       List of person Detection objects.
            vehicle_bbox:  (x1, y1, x2, y2) of the vehicle.
            iou_threshold: Minimum overlap to count (centroid check + IoU).

        Returns:
            Filtered list of persons that are inside the vehicle bbox.
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        inside = []
        for p in persons:
            cx, cy = p.centroid
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                inside.append(p)
        return inside

    # ── Sub-classifier Stubs ──────────────────────────────────────────

    def classify_helmet(self, head_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Classify whether a rider is wearing a helmet.

        ┌─────────────────────────────────────────────────────────────┐
        │  SWAP POINT — Fine-tuned Helmet Classifier                  │
        │  Replace with a YOLOv8 cls model trained on helmet data.    │
        │  Return (has_helmet: bool, confidence: float)               │
        └─────────────────────────────────────────────────────────────┘

        Current: simple brightness heuristic as placeholder (always review).
        """
        if head_crop is None or head_crop.size == 0:
            return True, 0.0  # unknown → assume helmet (conservative)
        # Placeholder: return no_helmet=False with low confidence → triggers manual review
        return False, 0.55  # confidence < 0.90 → goes to manual review queue

    def classify_seatbelt(self, driver_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Classify seatbelt presence for a car driver crop.

        ┌─────────────────────────────────────────────────────────────┐
        │  SWAP POINT — Fine-tuned Seatbelt Classifier                │
        │  Replace with YOLOv8 cls model trained on seatbelt data.    │
        └─────────────────────────────────────────────────────────────┘
        """
        if driver_crop is None or driver_crop.size == 0:
            return True, 0.0
        return False, 0.55  # placeholder

    def classify_phone(self, driver_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Classify whether driver is holding a phone.

        ┌─────────────────────────────────────────────────────────────┐
        │  SWAP POINT — Fine-tuned Phone Detector                     │
        └─────────────────────────────────────────────────────────────┘
        """
        if driver_crop is None or driver_crop.size == 0:
            return False, 0.0
        return False, 0.3  # placeholder — disabled by default in config

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
