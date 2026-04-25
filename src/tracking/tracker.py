"""
src/tracking/tracker.py
ByteTrack-based multi-object tracker via lapx.
Assigns stable unique IDs across frames and maintains per-track
trajectory history and class majority-vote.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Counter, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.detection.vehicle_detector import Detection
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Track Dataclass
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Track:
    """One confirmed tracked vehicle."""
    id:          int
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    class_id:    int
    class_name:  str
    centroid:    Tuple[int, int]
    confidence:  float
    age:         int   # frames since first seen
    hits:        int   # frames with a detection match
    timestamp:   float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════
# Vehicle Tracker
# ══════════════════════════════════════════════════════════════════════

class VehicleTracker:
    """
    Wraps ByteTrack (via lapx) to provide:
      - Stable integer track IDs
      - Per-track centroid trajectory history (world coords after homography)
      - Per-track pixel centroid history (for lane crossing counting)
      - Class majority-vote to reduce label flicker
      - Frame timestamp history for accurate Δt calculation
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 'tracking' section of settings.yaml.
        """
        self._max_age        = cfg.get("max_age", 30)
        self._min_hits       = cfg.get("min_hits", 3)
        self._iou_threshold  = cfg.get("iou_threshold", 0.3)
        self._hist_len       = cfg.get("trajectory_history", 90)

        # Per-track state
        # pixel centroid history   → for lane crossing, zigzag sign changes
        self.pixel_history:  Dict[int, Deque[Tuple[int, int]]]   = defaultdict(lambda: deque(maxlen=self._hist_len))
        # world centroid history   → for speed / tailgating (metres)
        self.world_history:  Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=self._hist_len))
        # timestamp history        → for Δt computation
        self.ts_history:     Dict[int, Deque[float]]              = defaultdict(lambda: deque(maxlen=self._hist_len))
        # class votes              → majority vote per track
        self.class_votes:    Dict[int, Counter]                   = defaultdict(Counter)
        # confirmed track ages
        self._ages:          Dict[int, int]                       = defaultdict(int)
        self._hits:          Dict[int, int]                       = defaultdict(int)

        self._tracker = None
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialise the ByteTrack tracker from lapx."""
        try:
            from lapx.byte_tracker import BYTETracker

            class _Args:
                track_thresh  = 0.35
                track_buffer  = 30
                match_thresh  = 0.8
                mot20         = False

            self._tracker = BYTETracker(_Args(), frame_rate=15)
            logger.info("VehicleTracker: ByteTrack initialised via lapx.")
        except ImportError:
            logger.warning(
                "lapx not installed — tracker will run in DETECTION-ONLY mode "
                "(each frame re-starts IDs). Install: pip install lapx"
            )
            self._tracker = None

    # ── Fallback: simple centroid re-ID ──────────────────────────────

    _NEXT_ID = 1

    def _fallback_update(
        self,
        detections: List[Detection],
        frame_ts: float,
    ) -> List[Track]:
        """Assign sequential IDs when ByteTrack is unavailable."""
        tracks = []
        for det in detections:
            tid = VehicleTracker._NEXT_ID
            VehicleTracker._NEXT_ID += 1
            self.pixel_history[tid].append(det.centroid)
            self.ts_history[tid].append(frame_ts)
            self.class_votes[tid][det.class_id] += 1
            tracks.append(Track(
                id=tid, bbox=det.bbox, class_id=det.class_id,
                class_name=det.class_name, centroid=det.centroid,
                confidence=det.confidence, age=1, hits=1,
                timestamp=frame_ts,
            ))
        return tracks

    # ── Public API ────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_ts: Optional[float] = None,
        homography=None,
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: Vehicle detections from this frame.
            frame:      Current BGR frame (for frame size info).
            frame_ts:   Epoch timestamp of this frame.
            homography: Optional HomographyCalibrator for world coords.

        Returns:
            List of confirmed Track objects.
        """
        if frame_ts is None:
            frame_ts = time.time()

        if self._tracker is None:
            return self._fallback_update(detections, frame_ts)

        if not detections:
            # Still call tracker update to age out stale tracks
            try:
                self._tracker.update(
                    np.empty((0, 5), dtype=np.float32),
                    frame.shape[:2],
                    frame.shape[:2],
                )
            except Exception:
                pass
            return []

        # Build input array: [x1, y1, x2, y2, confidence]
        det_arr = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
            for d in detections
        ], dtype=np.float32)

        try:
            online_targets = self._tracker.update(
                det_arr, frame.shape[:2], frame.shape[:2]
            )
        except Exception as e:
            logger.warning(f"ByteTrack update error: {e} — using fallback.")
            return self._fallback_update(detections, frame_ts)

        tracks: List[Track] = []

        for t in online_targets:
            tid    = int(t.track_id)
            tlwh   = t.tlwh
            x1     = int(tlwh[0])
            y1     = int(tlwh[1])
            x2     = int(tlwh[0] + tlwh[2])
            y2     = int(tlwh[1] + tlwh[3])
            bbox   = (x1, y1, x2, y2)
            conf   = float(t.score)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Match to nearest detection for class ID
            matched_class_id   = self._match_class(detections, bbox)
            matched_class_name = next(
                (d.class_name for d in detections
                 if self._bbox_overlap(d.bbox, bbox) > 0.3),
                "vehicle"
            )

            # Update class votes (majority vote for stable labels)
            self.class_votes[tid][matched_class_id] += 1
            stable_class_id = self.class_votes[tid].most_common(1)[0][0]
            stable_class_name = next(
                (d.class_name for d in detections if d.class_id == stable_class_id),
                matched_class_name
            )

            # Update pixel history
            self.pixel_history[tid].append((cx, cy))

            # Update world history via homography
            if homography is not None and homography.is_calibrated:
                wx, wy = homography.pixel_to_world(cx, cy)
                self.world_history[tid].append((wx, wy))

            # Update timestamp history
            self.ts_history[tid].append(frame_ts)

            # Update age and hits
            self._ages[tid]  += 1
            self._hits[tid]  += 1

            tracks.append(Track(
                id=tid, bbox=bbox,
                class_id=stable_class_id, class_name=stable_class_name,
                centroid=(cx, cy), confidence=conf,
                age=self._ages[tid], hits=self._hits[tid],
                timestamp=frame_ts,
            ))

        # Prune state for old tracks not seen recently
        active_ids = {t.id for t in tracks}
        for tid in list(self._ages.keys()):
            if tid not in active_ids:
                self._ages[tid] += 1
                if self._ages[tid] > self._max_age * 2:
                    # Full cleanup to release memory
                    for store in (self.pixel_history, self.world_history,
                                  self.ts_history, self.class_votes,
                                  self._ages, self._hits):
                        store.pop(tid, None)

        return tracks

    # ── Helpers ───────────────────────────────────────────────────────

    def _match_class(
        self,
        detections: List[Detection],
        bbox: Tuple[int, int, int, int],
    ) -> int:
        """Match a tracker bbox to nearest detection by IoU for class ID."""
        best_iou  = 0.0
        best_cid  = 0
        for d in detections:
            iou = self._bbox_overlap(d.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_cid = d.class_id
        return best_cid

    @staticmethod
    def _bbox_overlap(
        b1: Tuple[int, int, int, int],
        b2: Tuple[int, int, int, int],
    ) -> float:
        """Simple IoU between two bboxes."""
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter == 0:
            return 0.0
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)
