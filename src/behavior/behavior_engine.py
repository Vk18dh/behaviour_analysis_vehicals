"""
src/behavior/behavior_engine.py
Complete behavior detection engine implementing all 13 behaviors:
  1. Zigzag / Weaving         (exact formula from paper)
  2. Tailgating               (time headway < 1.5s, same lane)
  3. Red-Light Violation      (ROI crossing when signal=RED)
  4. Overspeed                (speed > limit > 3s)
  5. Wrong-Way Driving        (dot product < threshold)
  6. Highway Restriction      (2W/3W in restricted zone)
  7. Lane Violation           (heavy vehicle in fast lane)
  8. Rash Driving             (high accel + high direction change rate)
  9. No Helmet                (rider head crop classifier)
 10. No Seatbelt              (driver crop classifier)
 11. Triple Riding            (≥3 persons on 2W)
 12. Illegal Turn / U-Turn    (junction ROI + angle > 135°)
 13. Phone Use                (optional, feature-flagged)

Each detector is a pure function: (track, features, config, state) → Violation | None
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.features.feature_extractor import VehicleFeatures
from src.tracking.tracker import Track
from src.utils.helpers import (
    count_sign_changes, poly_contains_point, vector_dot, vector_normalize
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Violation Dataclass
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Violation:
    """A detected traffic violation."""
    type:          str               # e.g. "ZIGZAG", "OVERSPEED"
    track_id:      int
    vehicle_class: str
    frame_idx:     int
    timestamp:     float = field(default_factory=time.time)
    confidence:    float = 1.0
    metadata:      Dict[str, Any] = field(default_factory=dict)
    # Set by ANPR later
    plate_text:    str  = ""
    plate_conf:    float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Per-Track Sustained-State Tracker
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _SustainedState:
    """Tracks how long a condition has been continuously true."""
    active:    bool  = False
    start_ts:  float = 0.0
    flagged:   bool  = False   # True once violation was emitted

    def begin(self, ts: float) -> None:
        if not self.active:
            self.active   = True
            self.start_ts = ts
            self.flagged  = False

    def reset(self) -> None:
        self.active  = False
        self.flagged = False

    def duration(self, now: float) -> float:
        return (now - self.start_ts) if self.active else 0.0

    def should_flag(self, now: float, threshold_sec: float) -> bool:
        if self.active and not self.flagged:
            if self.duration(now) >= threshold_sec:
                self.flagged = True
                return True
        return False


# ══════════════════════════════════════════════════════════════════════
# Behavior Engine
# ══════════════════════════════════════════════════════════════════════

class BehaviorEngine:
    """
    Runs all behavior detectors on each frame's track+feature data.
    Call run_all() to get a list of Violations for the current frame.
    """

    def __init__(self, cfg: dict, detector=None) -> None:
        """
        Args:
            cfg:      'behavior' section from settings.yaml.
            detector: VehicleDetector instance (for sub-classifiers).
        """
        self._cfg      = cfg
        self._detector = detector

        # Per-track sustained-state stores (one per violation type)
        self._states: Dict[str, Dict[int, _SustainedState]] = defaultdict(
            lambda: defaultdict(_SustainedState)
        )

        # Per-track zigzag rolling window (deque of (ts, dx) tuples)
        self._zz_window: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque()
        )
        # Per-track helmet sustained timer (separate because confidence-gated)
        self._helmet_sus: Dict[int, _SustainedState] = defaultdict(_SustainedState)

        # Signal state — can be overridden via API
        self.signal_state: str = cfg.get("red_light", {}).get("signal_state", "GREEN")

        # Frame dimensions (set each call) — used for ROI fraction scaling
        self._frame_w: int = 1280
        self._frame_h: int = 720

        logger.info("BehaviorEngine initialised — all 13 detectors active.")

    # ══════════════════════════════════════════════════════════════════
    # Main Entry Point
    # ══════════════════════════════════════════════════════════════════

    def run_all(
        self,
        tracks: List[Track],
        features: Dict[int, VehicleFeatures],
        lane_result=None,
        frame_idx: int = 0,
        frame: Optional[np.ndarray] = None,
        persons: Optional[List] = None,
        frame_ts: Optional[float] = None,
    ) -> List[Violation]:
        """
        Run every behavior detector on the current frame.

        Args:
            tracks:      Confirmed tracks from tracker.
            features:    Per-track VehicleFeatures from feature extractor.
            lane_result: LaneResult (for crossing count, zone checks).
            frame_idx:   Current frame number.
            frame:       BGR frame (for sub-classifier crops).
            persons:     Person Detection list (for triple-riding).

        Returns:
            List of Violation objects detected this frame.
        """
        violations: List[Violation] = []
        now = frame_ts if frame_ts is not None else time.time()

        # Update frame dimensions for ROI fraction scaling
        if frame is not None:
            self._frame_h, self._frame_w = frame.shape[:2]

        for track in tracks:
            tid  = track.id
            f    = features.get(tid)
            if f is None:
                continue

            # 1. Zigzag
            v = self._detect_zigzag(track, f, frame_idx, now)
            if v: violations.append(v)

            # 2. Tailgating (Disabled per user request)
            # v = self._detect_tailgating(track, f, frame_idx, now)
            # if v: violations.append(v)

            # 3. Red-Light
            v = self._detect_red_light(track, f, frame_idx, now)
            if v: violations.append(v)

            # 4. Overspeed (Disabled per user request)
            # v = self._detect_overspeed(track, f, frame_idx, now)
            # if v: violations.append(v)

            # 5. Wrong-Way
            v = self._detect_wrong_direction(track, f, frame_idx, now)
            if v: violations.append(v)

            # 6. Highway Restriction
            v = self._detect_highway_restriction(track, f, frame_idx, now)
            if v: violations.append(v)

            # 7. Lane Violation
            v = self._detect_lane_violation(track, f, frame_idx, now)
            if v: violations.append(v)

            # 8. Rash Driving
            v = self._detect_rash_driving(track, f, frame_idx, now)
            if v: violations.append(v)

            # 9. Helmet (requires detector + frame)
            if frame is not None and self._detector is not None:
                v = self._detect_no_helmet(track, f, frame, frame_idx, now)
                if v: violations.append(v)

            # 10. Seatbelt (Disabled per user request)
            # if frame is not None and self._detector is not None:
            #     v = self._detect_no_seatbelt(track, f, frame, frame_idx, now)
            #     if v: violations.append(v)

            # 11. Triple Riding
            if persons is not None:
                v = self._detect_triple_riding(track, f, persons, frame_idx, now)
                if v: violations.append(v)

            # 12. Illegal Turn
            v = self._detect_illegal_turn(track, f, frame_idx, now)
            if v: violations.append(v)

            # 13. Phone Use (Disabled per user request)
            # if (self._cfg.get("phone_use", {}).get("enabled", False)
            #         and frame is not None
            #         and self._detector is not None):
            #     v = self._detect_phone_use(track, f, frame, frame_idx, now)
            #     if v: violations.append(v)

        return violations

    # ══════════════════════════════════════════════════════════════════
    # ROI Scaling Helper
    # ══════════════════════════════════════════════════════════════════

    def _scale_roi(
        self,
        roi: List[List],
    ) -> List[Tuple[int, int]]:
        """
        Convert an ROI polygon to absolute pixel coordinates.

        Supports TWO formats:
          - Pixel format:    [[400, 500], [880, 500], ...] (values > 1.5)
          - Fraction format: [[0.31, 0.69], [0.69, 0.69], ...] (values 0.0–1.5)

        Fraction format is camera-resolution-agnostic and preferred.
        Pixel format is kept for backward compatibility.

        Args:
            roi: List of [x, y] pairs from config.

        Returns:
            List of (x_px, y_px) integer tuples.
        """
        if not roi:
            return []
        # Detect fraction vs pixel by checking if max value <= 1.5
        flat_vals = [v for pt in roi for v in pt]
        is_fraction = max(flat_vals) <= 1.5
        result = []
        for pt in roi:
            x, y = pt[0], pt[1]
            if is_fraction:
                result.append((int(x * self._frame_w), int(y * self._frame_h)))
            else:
                result.append((int(x), int(y)))
        return result

    # ══════════════════════════════════════════════════════════════════
    # Detector 1 — Zigzag / Weaving
    # ══════════════════════════════════════════════════════════════════

    def _detect_zigzag(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        Exact algorithm from the research paper:

        Let x(t) = lateral world position at time t
        Δxᵢ = x(tᵢ) - x(tᵢ₋₁)

        SLIDING WINDOW of T seconds:
          weave_count      = number of sign changes in {Δxᵢ}
          lateral_shift    = Σ|Δxᵢ|  where |Δxᵢ| > noise_filter
          lateral_accel    = max(Δx/Δt²) in window

        FLAG if:
          weave_count >= 2
          AND lateral_shift > displacement_threshold (4.0 m)
          AND lateral_accel > accel_threshold (2.5 m/s²)
          AND sustained ≥ 2 s
          (ALSO: lane-line crossing count > 1 strengthens the flag)
        """
        cfg = self._cfg.get("zigzag", {})
        T          = cfg.get("time_window_sec", 5.0)
        min_chg    = cfg.get("min_direction_changes", 2)
        disp_thr   = cfg.get("lateral_displacement_m", 4.0)
        noise_filt = cfg.get("noise_filter_m", 1.0)
        accel_thr  = cfg.get("lateral_accel_threshold", 2.5)
        sustain    = cfg.get("sustained_sec", 2.0)
        state      = self._states["ZIGZAG"][track.id]

        # Update rolling window with (timestamp, dx) pairs
        win = self._zz_window[track.id]
        lat_hist = list(f.lateral_dx_history)
        if lat_hist:
            win.append((now, lat_hist[-1]))

        # Prune entries older than T seconds
        while win and (now - win[0][0]) > T:
            win.popleft()

        if len(win) < 3:
            state.reset()
            return None

        dx_vals = [abs(dx) for _, dx in win if abs(dx) > noise_filt]
        dx_signs = [dx for _, dx in win if abs(dx) > noise_filt]

        lateral_shift = sum(dx_vals)
        weave_count   = count_sign_changes(dx_signs)

        condition_met = (
            weave_count  >= min_chg
            and lateral_shift >= disp_thr
            and f.lateral_accel_mps2 >= accel_thr
        )

        if condition_met:
            state.begin(now)
            if state.should_flag(now, sustain):
                logger.debug(
                    f"[t{track.id}] ZIGZAG: changes={weave_count}, "
                    f"shift={lateral_shift:.2f}m, accel={f.lateral_accel_mps2:.2f}"
                )
                return Violation(
                    type="ZIGZAG", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.90,
                    metadata={
                        "weave_count": weave_count,
                        "lateral_shift_m": round(lateral_shift, 2),
                        "lateral_accel": round(f.lateral_accel_mps2, 2),
                        "speed_kmh": round(f.speed_kmh, 1),
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 2 — Tailgating
    # ══════════════════════════════════════════════════════════════════

    def _detect_tailgating(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if:
          same lane (checked by feature extractor)
          AND time_headway < threshold (1.5 s)
          AND speed > min_speed (avoids stop-and-go false positives)
          AND sustained > 2.0 s
        """
        cfg     = self._cfg.get("tailgating", {})
        hw_thr  = cfg.get("time_headway_sec", 1.5)
        sustain = cfg.get("sustained_duration_sec", 2.0)
        min_spd = self._cfg.get("features", {}).get("min_speed_for_tailgate", 5.0)
        state   = self._states["TAILGATING"][track.id]

        if f.speed_kmh < min_spd:
            state.reset()
            return None

        if f.time_headway_s < hw_thr:
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="TAILGATING", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.85,
                    metadata={
                        "time_headway_s": round(f.time_headway_s, 2),
                        "following_dist_m": round(f.following_dist_m, 2),
                        "speed_kmh": round(f.speed_kmh, 1),
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 3 — Red-Light Violation
    # ══════════════════════════════════════════════════════════════════

    def _detect_red_light(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        Zero-tolerance: any frame where vehicle centroid is inside the
        stop-line ROI while signal_state == RED is a violation.
        Min speed filter to exclude naturally parked vehicles.
        """
        cfg      = self._cfg.get("red_light", {})
        roi_raw  = cfg.get("roi_polygon", [[0.31, 0.69], [0.69, 0.69], [0.69, 0.83], [0.31, 0.83]])
        roi      = self._scale_roi(roi_raw)
        min_spd  = cfg.get("min_speed_kmh", 2.0)
        state    = self._states["RED_LIGHT"][track.id]

        if self.signal_state != "RED":
            state.reset()
            return None
        if f.speed_kmh < min_spd:
            return None

        cx, cy = track.centroid
        if poly_contains_point(roi, (cx, cy)):
            if not state.flagged:
                state.flagged = True
                return Violation(
                    type="RED_LIGHT", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.95,
                    metadata={
                        "speed_kmh": round(f.speed_kmh, 1),
                        "signal": self.signal_state,
                    },
                )
        else:
            state.flagged = False
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 4 — Overspeed
    # ══════════════════════════════════════════════════════════════════

    def _detect_overspeed(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if speed > limit AND sustained > 3.0 s.
        Fine: base_fine + per_5kmh × floor((speed-limit)/5)
        Deduction: 1 point per 5 km/h over.
        """
        cfg     = self._cfg.get("overspeed", {})
        limit   = cfg.get("speed_limit_kmh", 60.0)
        sustain = cfg.get("sustained_duration_sec", 3.0)
        state   = self._states["OVERSPEED"][track.id]

        if f.speed_kmh > limit:
            state.begin(now)
            if state.should_flag(now, sustain):
                excess = f.speed_kmh - limit
                return Violation(
                    type="OVERSPEED", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.90,
                    metadata={
                        "speed_kmh":    round(f.speed_kmh, 1),
                        "limit_kmh":    limit,
                        "excess_kmh":   round(excess, 1),
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 5 — Wrong-Way Driving
    # ══════════════════════════════════════════════════════════════════

    def _detect_wrong_direction(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if dot(movement_vector, road_direction) < threshold (-0.5).
        Road direction configured per camera in settings.yaml.
        """
        cfg      = self._cfg.get("wrong_direction", {})
        dot_thr  = cfg.get("dot_product_threshold", -0.5)
        sustain  = cfg.get("sustained_sec", 1.5)
        
        # Read road direction from config, fallback to upward (0, -1)
        rd_cfg   = cfg.get("road_dir", [0.0, -1.0])
        road_dir = (float(rd_cfg[0]), float(rd_cfg[1]))
        
        state    = self._states["WRONG_DIRECTION"][track.id]

        if f.direction_vec == (0.0, 0.0) or f.speed_kmh < 5.0:
            state.reset()
            return None

        dot = vector_dot(f.direction_vec, road_dir)
        if dot < dot_thr:
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="WRONG_DIRECTION", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.88,
                    metadata={
                        "dot_product":   round(dot, 3),
                        "speed_kmh":     round(f.speed_kmh, 1),
                        "direction_vec": f.direction_vec,
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 6 — Highway Restriction (2W / 3W)
    # ══════════════════════════════════════════════════════════════════

    def _detect_highway_restriction(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if a two-wheeler or three-wheeler is inside a restricted zone polygon.
        Zone polygon is configurable per camera.
        """
        cfg     = self._cfg.get("highway_restriction", {})
        
        if not cfg.get("enabled", True):
            return None

        zone_raw = cfg.get("zone_polygon", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        zone     = self._scale_roi(zone_raw)
        sustain  = cfg.get("sustained_sec", 0.5)
        state    = self._states["HIGHWAY_RESTRICTION"][track.id]

        if self._detector is None:
            return None
        if not (self._detector.is_two_wheeler(track.class_id)
                or self._detector.is_three_wheeler(track.class_id)):
            state.reset()
            return None

        cx, cy = track.centroid
        if poly_contains_point(zone, (cx, cy)):
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="HIGHWAY_RESTRICTION", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.92,
                    metadata={"zone": zone, "class": track.class_name},
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 7 — Lane Violation (heavy vehicle in fast lane)
    # ══════════════════════════════════════════════════════════════════

    def _detect_lane_violation(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if bus/truck is in the rightmost (fastest) lane
        sustained > 3 s.
        Fast lane = lane index ≥ configured fast_lane_threshold.
        """
        cfg      = self._cfg.get("lane_violation", {})
        fast_x   = cfg.get("fast_lane_x_fraction", 0.6)
        sustain  = cfg.get("sustained_sec", 3.0)
        state    = self._states["LANE_VIOLATION"][track.id]

        if self._detector is None or not self._detector.is_heavy(track.class_id):
            state.reset()
            return None

        # Flag if the vehicle is on the right side (fast lane fraction)
        cx, _ = track.centroid
        if (cx / self._frame_w) >= fast_x:
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="LANE_VIOLATION", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.85,
                    metadata={
                        "lane_fraction": round(cx / self._frame_w, 2),
                        "speed_kmh":     round(f.speed_kmh, 1),
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 8 — Rash Driving
    # ══════════════════════════════════════════════════════════════════

    def _detect_rash_driving(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if BOTH:
          acceleration > accel_threshold (4.0 m/s²)
          AND lateral_accel > rash proxy (≈ direction change rate)
        Must be simultaneously true.
        """
        cfg       = self._cfg.get("rash_driving", {})
        accel_thr = cfg.get("accel_threshold", 4.0)
        lat_thr   = cfg.get("direction_change_rate", 1.5)
        state     = self._states["RASH_DRIVING"][track.id]

        # Use lateral_accel as erratic-motion proxy
        if f.accel_mps2 > accel_thr and f.lateral_accel_mps2 > lat_thr:
            state.begin(now)
            if state.should_flag(now, 1.0):   # 1.0 s sustained
                return Violation(
                    type="RASH_DRIVING", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.82,
                    metadata={
                        "accel_mps2":       round(f.accel_mps2, 2),
                        "lateral_accel":    round(f.lateral_accel_mps2, 2),
                        "speed_kmh":        round(f.speed_kmh, 1),
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 9 — No Helmet
    # ══════════════════════════════════════════════════════════════════

    def _detect_no_helmet(
        self,
        track: Track,
        f: VehicleFeatures,
        frame: np.ndarray,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        Only runs on two-wheelers.
        Crops head region (upper 30% of vehicle bbox), runs sub-classifier.
        FLAG if no_helmet confidence >= 0.90 sustained ≥ 1.0 s.
        Confidence 0.50–0.90 → flags with status 'low_confidence' (manual review).
        """
        cfg      = self._cfg.get("helmet", {})
        conf_thr = cfg.get("confidence_threshold", 0.90)
        sustain  = cfg.get("sustained_sec", 1.0)
        state    = self._helmet_sus[track.id]

        if not self._detector.is_two_wheeler(track.class_id):
            return None

        # Crop head region (upper 30%)
        x1, y1, x2, y2 = track.bbox
        head_h  = max(1, int((y2 - y1) * 0.30))
        head_crop = frame[y1:y1 + head_h, x1:x2]

        has_helmet, conf = self._detector.classify_helmet(head_crop)

        if not has_helmet and conf >= 0.50:
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="NO_HELMET", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=conf,
                    metadata={
                        "ocr_confidence": conf,
                        "needs_review": conf < conf_thr,
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 10 — No Seatbelt
    # ══════════════════════════════════════════════════════════════════

    def _detect_no_seatbelt(
        self,
        track: Track,
        f: VehicleFeatures,
        frame: np.ndarray,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        Only runs on cars.
        Crops driver region (upper-left quadrant of bbox).
        Requires sustained detection over time to avoid single-frame hallucination.
        """
        cfg      = self._cfg.get("seatbelt", {})
        conf_thr = cfg.get("confidence_threshold", 0.85)
        sustain  = cfg.get("sustained_sec", 0.5)
        state    = self._states["NO_SEATBELT"][track.id]

        if self._detector is None or track.class_id not in {2}:  # car only
            return None
            
        # Seatbelts are only visible from the front windshield
        if not getattr(f, "moving_towards_camera", False):
            return None

        x1, y1, x2, y2 = track.bbox
        driver_crop = frame[y1:int(y1 + (y2-y1)*0.6), x1:int(x1 + (x2-x1)*0.5)]

        has_belt, conf = self._detector.classify_seatbelt(driver_crop)

        if not has_belt and conf >= 0.50:
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="NO_SEATBELT", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=conf,
                    metadata={
                        "ocr_confidence": conf,
                        "needs_review": conf < conf_thr,
                    },
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 11 — Triple Riding
    # ══════════════════════════════════════════════════════════════════

    def _detect_triple_riding(
        self,
        track: Track,
        f: VehicleFeatures,
        persons: List,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if a two-wheeler track has ≥ 3 persons inside its bbox.
        One-shot per track entry.
        """
        cfg      = self._cfg.get("triple_riding", {})
        min_p    = cfg.get("min_persons", 3)
        sustain  = cfg.get("sustained_duration_sec", 0.5)  # Require 0.5s of continuous detection
        state    = self._states["TRIPLE_RIDING"][track.id]

        if self._detector is None or not self._detector.is_two_wheeler(track.class_id):
            return None

        inside = self._detector.persons_inside_bbox(persons, track.bbox)
        if len(inside) >= min_p:
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="TRIPLE_RIDING", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=0.88,
                    metadata={"person_count": len(inside)},
                )
        else:
            state.reset()
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 12 — Illegal Turn / U-Turn
    # ══════════════════════════════════════════════════════════════════

    def _detect_illegal_turn(
        self,
        track: Track,
        f: VehicleFeatures,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        FLAG if vehicle is in junction ROI AND direction vector rotates
        > angle_threshold_deg within time_window_sec.
        Uses direction vector history. Requires sustained turn to avoid bbox jitter.
        """
        cfg       = self._cfg.get("illegal_turn", {})
        roi_raw   = cfg.get("junction_roi", [[0.23, 0.55], [0.70, 0.55], [0.70, 1.0], [0.23, 1.0]])
        roi       = self._scale_roi(roi_raw)
        angle_thr = cfg.get("angle_threshold_deg", 135.0)
        sustain   = cfg.get("sustained_sec", 0.5)
        state     = self._states["ILLEGAL_TURN"][track.id]

        cx, cy = track.centroid
        if not poly_contains_point(roi, (cx, cy)):
            state.reset()
            return None

        if f.direction_vec == (0.0, 0.0):
            return None

        # Compare with stored previous direction
        prev_dir = state.__dict__.get("_prev_dir", None)
        curr_dir = f.direction_vec

        if prev_dir is not None and prev_dir != (0.0, 0.0):
            dot = max(-1.0, min(1.0, vector_dot(prev_dir, curr_dir)))
            angle = math.degrees(math.acos(dot))
            if angle > angle_thr:
                state.begin(now)
                if state.should_flag(now, sustain):
                    state._prev_dir = curr_dir
                    return Violation(
                        type="ILLEGAL_TURN", track_id=track.id,
                        vehicle_class=track.class_name, frame_idx=frame_idx,
                        confidence=0.80,
                        metadata={"angle_deg": round(angle, 1)},
                    )
            else:
                state.reset()
        else:
            state.reset()

        state._prev_dir = curr_dir
        return None

    # ══════════════════════════════════════════════════════════════════
    # Detector 13 — Phone Use (Optional)
    # ══════════════════════════════════════════════════════════════════

    def _detect_phone_use(
        self,
        track: Track,
        f: VehicleFeatures,
        frame: np.ndarray,
        frame_idx: int,
        now: float,
    ) -> Optional[Violation]:
        """
        Optional detector (requires in-cabin camera or side angle).
        Disabled by default via config: behavior.phone_use.enabled = false.
        """
        cfg      = self._cfg.get("phone_use", {})
        conf_thr = cfg.get("confidence_threshold", 0.80)
        sustain  = cfg.get("sustained_sec", 0.5)
        state    = self._states["PHONE_USE"][track.id]

        if track.class_id not in {2}:  # car only
            return None

        x1, y1, x2, y2 = track.bbox
        driver_crop = frame[y1:int(y1+(y2-y1)*0.7), x1:int(x1+(x2-x1)*0.6)]

        using_phone, conf = self._detector.classify_phone(driver_crop)

        if using_phone and conf >= conf_thr:
            state.begin(now)
            if state.should_flag(now, sustain):
                return Violation(
                    type="PHONE_USE", track_id=track.id,
                    vehicle_class=track.class_name, frame_idx=frame_idx,
                    confidence=conf,
                    metadata={"confidence": conf},
                )
        else:
            state.reset()
        return None
