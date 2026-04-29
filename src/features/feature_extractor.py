"""
src/features/feature_extractor.py
Per-track feature computation:
  - Speed (km/h): world-coord displacement / Δt via homography
  - Speed fallback: optical flow on the vehicle ROI
  - Lateral speed (m/s) and lateral acceleration (Δx/Δt²)
  - Acceleration (m/s²): Δspeed / Δt, smoothed
  - Direction vector (unit)
  - Lateral direction history (for zigzag sign-change counting)
  - Lane index
  - Following distance (m) and time headway (s) for tailgating
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tracking.tracker import Track
from src.utils.helpers import (
    smooth_values, count_sign_changes, compute_lateral_accel,
    vector_normalize, euclidean_distance,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Features Dataclass
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VehicleFeatures:
    """All computed features for one vehicle in one frame."""
    track_id:          int
    speed_kmh:         float = 0.0
    lateral_speed_mps: float = 0.0    # Δx_world / Δt
    accel_mps2:        float = 0.0    # |Δspeed / Δt|
    lateral_accel_mps2: float = 0.0  # Δx / Δt² proxy
    direction_vec:     Tuple[float, float] = (0.0, 0.0)  # unit vector
    # Rolling lateral delta history (m per frame) — for sign-change counter
    lateral_dx_history: Deque[float] = field(default_factory=lambda: deque(maxlen=150))
    lane_index:        int = 0
    following_dist_m:  float = 999.0   # distance to vehicle ahead (same lane)
    time_headway_s:    float = 999.0   # following_dist / speed
    # Raw pixel centroid history kept by tracker — referenced here
    speed_history:     Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    moving_towards_camera: bool = False  # True if moving down the image (y increasing)


# ══════════════════════════════════════════════════════════════════════
# Feature Extractor
# ══════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Stateful feature extractor — maintains rolling history per track ID.
    Call update() once per frame after the tracker.
    """

    def __init__(
        self,
        cfg: dict,
        homography=None,         # HomographyCalibrator | None
        fps: float = 15.0,
    ) -> None:
        """
        Args:
            cfg:         'features' sub-dict from settings.yaml.
            homography:  Calibrated HomographyCalibrator (or None for px fallback).
            fps:         Camera/pipeline FPS (used when timestamps unavailable).
        """
        self._speed_win  = cfg.get("speed_smoothing_window", 5)
        self._accel_win  = cfg.get("accel_smoothing_window", 3)
        self._min_speed  = cfg.get("min_speed_for_tailgate", 5.0)
        self._use_flow   = cfg.get("use_optical_flow_fallback", True)
        # px_per_meter_fallback: used when homography is not calibrated.
        # Tune this to match your camera: count pixels for a known real-world length.
        self._px_per_m   = cfg.get("px_per_meter_fallback", 20.0)
        self._homography = homography
        self._fps        = fps
        self._dt         = 1.0 / fps
        self._uncalib_warn_counter = 0   # throttle calibration warnings

        # Per-track rolling state
        self._world_prev:   Dict[int, Tuple[float, float]] = {}  # (wx, wy) previous
        self._pixel_prev:   Dict[int, Tuple[int, int]]     = {}  # (cx, cy) previous
        self._pixel_y_hist: Dict[int, Deque[int]]          = defaultdict(lambda: deque(maxlen=15))
        self._ts_prev:      Dict[int, float]               = {}  # epoch seconds
        self._speed_hist:   Dict[int, Deque[float]]        = defaultdict(lambda: deque(maxlen=30))
        self._lateral_hist: Dict[int, Deque[float]]        = defaultdict(lambda: deque(maxlen=150))
        self._features:     Dict[int, VehicleFeatures]     = {}

        # Optical flow (dense Farneback) — computed once per frame, shared
        self._prev_gray: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────

    def update(
        self,
        tracks: List[Track],
        frame: np.ndarray,
        lane_result=None,        # LaneResult | None
        world_history: Optional[Dict] = None,  # from tracker.world_history
        ts_history:    Optional[Dict] = None,  # from tracker.ts_history
    ) -> Dict[int, VehicleFeatures]:
        """
        Compute features for all active tracks this frame.

        Args:
            tracks:        Confirmed tracks from VehicleTracker.update().
            frame:         Current BGR frame.
            lane_result:   LaneResult from LaneDetector (for lane index).
            world_history: tracker.world_history dict (world centroids).
            ts_history:    tracker.ts_history dict (timestamps per track).

        Returns:
            Dict mapping track_id → VehicleFeatures.
        """
        # Compute dense optical flow for this frame (shared across all tracks)
        flow = self._compute_flow(frame)

        results: Dict[int, VehicleFeatures] = {}

        for track in tracks:
            tid = track.id
            f   = self._features.get(tid, VehicleFeatures(track_id=tid))

            # ── Speed via world coordinates ───────────────────────────
            speed_kmh = 0.0
            lat_speed = 0.0
            if world_history and tid in world_history and len(world_history[tid]) >= 2:
                wh = list(world_history[tid])
                th = list(ts_history[tid]) if ts_history else None
                
                # Real-world fix: Use a longer baseline (up to 15 frames / 1 sec)
                # to eliminate bounding box jitter noise. Frame-to-frame jitter
                # artificially inflates speed.
                N = min(len(wh), 15)
                wx_start, wy_start = wh[-N]
                wx2, wy2 = wh[-1]
                
                if th and len(th) >= N:
                    dt = th[-1] - th[-N]
                else:
                    dt = self._dt * (N - 1)
                dt = max(dt, 1e-3)

                dist_m = math.sqrt((wx2 - wx_start)**2 + (wy2 - wy_start)**2)
                
                # Ignore sub-meter jitter if it's virtually stationary
                if dist_m < 0.5:
                    speed_mps = 0.0
                else:
                    speed_mps = dist_m / dt
                    
                speed_kmh = speed_mps * 3.6
                
                # Frame-to-frame delta for lateral acceleration and direction
                wx1, wy1 = wh[-2]
                dt_frame = (th[-1] - th[-2]) if (th and len(th) >= 2) else self._dt
                dt_frame = max(dt_frame, 1e-3)
                
                lat_speed = abs(wx2 - wx1) / dt_frame   # lateral world speed (m/s)

                # Direction vector
                f.direction_vec = vector_normalize((wx2 - wx1, wy2 - wy1))

                # Lateral delta for zigzag
                dx_world = wx2 - wx1
                self._lateral_hist[tid].append(dx_world)
                f.lateral_dx_history = self._lateral_hist[tid]

            elif self._use_flow and flow is not None:
                # Fallback: optical flow inside bbox
                speed_kmh = self._flow_speed(flow, track.bbox, frame.shape)
                # Warn if homography is not calibrated
                self._uncalib_warn_counter += 1
                if self._uncalib_warn_counter % 100 == 1:
                    if self._homography is None or not getattr(self._homography, 'is_calibrated', True):
                        logger.warning(
                            "[FeatureExtractor] Homography NOT calibrated — speed values "
                            "are APPROXIMATE (optical flow fallback). "
                            "Set src_points/dst_points in config/settings.yaml for accurate readings."
                        )

            # ── Speed smoothing ───────────────────────────────────────
            self._speed_hist[tid].append(speed_kmh)
            smoothed_speed = smooth_values(self._speed_hist[tid], self._speed_win)
            f.speed_kmh    = max(0.0, smoothed_speed)
            f.speed_history = self._speed_hist[tid]

            # ── Acceleration ──────────────────────────────────────────
            spd_list = list(self._speed_hist[tid])
            if len(spd_list) >= 2:
                dt    = self._dt
                if ts_history and tid in ts_history and len(ts_history[tid]) >= 2:
                    th = list(ts_history[tid])
                    dt = max(th[-1] - th[-2], 1e-3)
                dv_mps       = (spd_list[-1] - spd_list[-2]) / 3.6  # km/h → m/s
                f.accel_mps2 = abs(dv_mps / dt)
            else:
                f.accel_mps2 = 0.0

            # ── Lateral acceleration (zigzag proxy) ───────────────────
            if len(self._lateral_hist[tid]) >= 2:
                f.lateral_speed_mps  = lat_speed
                f.lateral_accel_mps2 = compute_lateral_accel(
                    self._lateral_hist[tid], self._dt
                )
            else:
                f.lateral_speed_mps  = 0.0
                f.lateral_accel_mps2 = 0.0

            # ── Lane index ────────────────────────────────────────────
            if lane_result is not None:
                from src.lane.lane_detection import LaneDetector  # local to avoid circular
                # We only need assign_lane which is stateless — use a helper
                f.lane_index = _assign_lane(track.centroid[0], lane_result)
            else:
                f.lane_index = 0

            # ── Directionality (towards/away) ──────────────────────────
            self._pixel_y_hist[tid].append(track.centroid[1])
            if len(self._pixel_y_hist[tid]) >= 5:
                # If y is increasing over the window, it's moving towards camera
                f.moving_towards_camera = (self._pixel_y_hist[tid][-1] > self._pixel_y_hist[tid][0] + 2)
            else:
                f.moving_towards_camera = False

            # Cache updated features
            self._features[tid] = f
            results[tid] = f

        # Prune stale entries not in current tracks
        active_ids = {t.id for t in tracks}
        for tid in list(self._features.keys()):
            if tid not in active_ids:
                # Keep for 30 frames then evict
                pass  # tracker handles eviction
                
        for tid in list(self._pixel_y_hist.keys()):
            if tid not in active_ids:
                del self._pixel_y_hist[tid]

        return results

    def compute_following_distances(
        self,
        tracks: List[Track],
        features: Dict[int, VehicleFeatures],
        world_history: Dict,
    ) -> None:
        """
        For each track, find the closest vehicle ahead in the same lane
        and compute following distance (m) + time headway (s).

        Modifies features dict in-place.

        Args:
            tracks:        Current tracks.
            features:      Feature dict from update().
            world_history: tracker.world_history.
        """
        for track in tracks:
            tid = track.id
            f   = features.get(tid)
            if f is None:
                continue

            best_dist = 999.0
            wy_self   = list(world_history.get(tid, []))[-1][1] if world_history.get(tid) else 0.0

            for other in tracks:
                if other.id == tid:
                    continue
                # Same lane check
                if features.get(other.id) and features[other.id].lane_index != f.lane_index:
                    continue
                # Ahead check: lower y in image = further along road (camera looks forward)
                # In world coords: smaller y_world = ahead (depends on calibration direction)
                wy_other = list(world_history.get(other.id, []))[-1][1] if world_history.get(other.id) else 0.0
                if wy_other < wy_self:   # other vehicle is ahead
                    dist = abs(wy_self - wy_other)
                    if dist < best_dist:
                        best_dist = dist

            f.following_dist_m = best_dist
            speed_mps = f.speed_kmh / 3.6
            if speed_mps > 0.5:
                f.time_headway_s = best_dist / speed_mps
            else:
                f.time_headway_s = 999.0

    # ── Optical Flow ──────────────────────────────────────────────────

    def _compute_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute dense Farneback optical flow between prev and current frame.
        Returns 2-channel flow array, or None if unavailable.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = None
        if self._prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
            except Exception as e:
                logger.debug(f"Optical flow failed: {e}")
        self._prev_gray = gray
        return flow

    def _flow_speed(
        self,
        flow: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple,
    ) -> float:
        """
        Estimate vehicle speed in km/h from mean optical flow magnitude
        inside its bounding box.  Approximate — returns 0 if flow is None.

        Args:
            flow:        2-channel Farneback flow array (h, w, 2).
            bbox:        Vehicle bbox (x1, y1, x2, y2).
            frame_shape: (h, w, c).

        Returns:
            Estimated speed in km/h.
        """
        if flow is None:
            return 0.0
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi_flow = flow[y1:y2, x1:x2]
        mag = np.sqrt(roi_flow[:, :, 0]**2 + roi_flow[:, :, 1]**2)
        mean_px_per_frame = float(mag.mean())
        # Conversion: pixels/frame → m/s → km/h
        # px_per_meter_fallback is configurable in settings.yaml (features section).
        # Default 20.0 px/m is a rough estimate; calibrate per-camera for accuracy.
        speed_mps = mean_px_per_frame * self._fps / max(self._px_per_m, 1.0)
        return speed_mps * 3.6


# ══════════════════════════════════════════════════════════════════════
# Standalone helper (avoids circular import from LaneDetector)
# ══════════════════════════════════════════════════════════════════════

def _assign_lane(centroid_x: int, lane_result) -> int:
    """Return 0-based lane index for a given pixel x and LaneResult."""
    bounds = sorted(lane_result.boundaries_px)
    for i, bx in enumerate(bounds):
        if centroid_x < bx:
            return i
    return len(bounds)
