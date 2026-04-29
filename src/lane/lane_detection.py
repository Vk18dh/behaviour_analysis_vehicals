"""
src/lane/lane_detection.py
Lane detection via Canny + Hough Lines with perspective homography.
Provides:
  - LaneDetector: detects left/right lane lines per frame
  - HomographyCalibrator: pixel ↔ world-coordinate transforms
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LaneLine:
    """A single lane line described by its two endpoints."""
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def slope(self) -> Optional[float]:
        dx = self.x2 - self.x1
        if abs(dx) < 1e-6:
            return None  # vertical
        return (self.y2 - self.y1) / dx

    @property
    def x_at_bottom(self) -> float:
        """X position at the bottom of the frame (useful for lane boundary)."""
        slope = self.slope
        if slope is None or abs(slope) < 1e-6:
            return float(self.x1)
        return self.x1 + (self.y1 - self.y2) / slope * 0  # extrapolated


@dataclass
class LaneResult:
    """Result from one frame's lane detection."""
    left:             Optional[LaneLine]
    right:            Optional[LaneLine]
    # Pixel x-coordinates of each lane boundary at the frame bottom
    boundaries_px:    List[int] = field(default_factory=list)
    # Which lane index (0=leftmost) each vehicle is in (filled by LaneDetector.assign_lane)
    frame_width:      int = 1280


# ══════════════════════════════════════════════════════════════════════
# Homography Calibrator
# ══════════════════════════════════════════════════════════════════════

class HomographyCalibrator:
    """
    Computes and applies a perspective homography that maps
    camera-plane pixels → bird's-eye world coordinates in metres.

    Call compute_homography() once at startup with calibration points.
    Then use pixel_to_world() on every centroid.
    """

    def __init__(self, pixels_per_meter: float = 12.0) -> None:
        """
        Args:
            pixels_per_meter: Calibration constant: how many bird's-eye
                              pixels correspond to one real-world metre.
        """
        self._H:    Optional[np.ndarray] = None   # pixel → world (bird's-eye px)
        self._H_inv: Optional[np.ndarray] = None  # world → pixel
        self._ppm   = pixels_per_meter
        self._calibrated = False

    def compute_homography(
        self,
        src_points: List[List[int]],
        dst_points: List[List[int]],
    ) -> None:
        """
        Compute the perspective warp matrix from 4 matching point pairs.

        Args:
            src_points: 4 camera-frame pixel coords [[x,y], ...].
            dst_points: 4 corresponding bird's-eye pixel coords [[x,y], ...].
        """
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        self._H     = cv2.getPerspectiveTransform(src, dst)
        self._H_inv = cv2.getPerspectiveTransform(dst, src)
        self._calibrated = True
        logger.info("Homography calibrated: pixel ↔ world mapping ready.")

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """
        Convert a camera-pixel coordinate to world metres.

        Args:
            px, py: Camera pixel (x, y).

        Returns:
            (x_m, y_m) in real-world metres.
        """
        if not self._calibrated:
            return (px / self._ppm, py / self._ppm)  # fallback: scale only

        pt = np.array([[[px, py]]], dtype=np.float32)
        bird = cv2.perspectiveTransform(pt, self._H)
        bx, by = float(bird[0][0][0]), float(bird[0][0][1])
        return (bx / self._ppm, by / self._ppm)

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Convert world metres back to camera pixel coords.

        Args:
            x_m, y_m: World coordinates in metres.

        Returns:
            (px, py) camera pixel.
        """
        if not self._calibrated:
            return (x_m * self._ppm, y_m * self._ppm)

        bx = x_m * self._ppm
        by = y_m * self._ppm
        pt = np.array([[[bx, by]]], dtype=np.float32)
        cam = cv2.perspectiveTransform(pt, self._H_inv)
        return (float(cam[0][0][0]), float(cam[0][0][1]))

    def compute_lane_width_m(
        self,
        left_x_px: float,
        right_x_px: float,
        y_px: float,
    ) -> float:
        """
        Estimate real-world lane width in metres given two pixel x-positions.
        """
        lx_m, _ = self.pixel_to_world(left_x_px, y_px)
        rx_m, _ = self.pixel_to_world(right_x_px, y_px)
        return abs(rx_m - lx_m)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


# ══════════════════════════════════════════════════════════════════════
# Lane Detector
# ══════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    Detects left and right lane lines in a preprocessed frame using
    Canny edge detection + Probabilistic Hough Lines.

    Maintains the last valid result so downstream modules always
    have a non-None lane estimate even on missed frames.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 'lane' section from settings.yaml.
        """
        self._canny_low  = cfg.get("canny_low", 50)
        self._canny_high = cfg.get("canny_high", 150)
        self._hough_rho  = cfg.get("hough_rho", 1)
        self._hough_theta = math.radians(cfg.get("hough_theta_deg", 1.0))
        self._hough_thresh = cfg.get("hough_threshold", 50)
        self._min_len     = cfg.get("hough_min_line_len", 40)
        self._max_gap     = cfg.get("hough_max_line_gap", 150)
        self._roi_top     = cfg.get("roi_top_fraction", 0.55)
        self._last_result: Optional[LaneResult] = None

        # Build calibrator from config
        self.calibrator = HomographyCalibrator(
            pixels_per_meter=cfg.get("pixels_per_meter", 12.0)
        )
        self._cfg_src = cfg.get("src_points")
        self._cfg_dst = cfg.get("dst_points")

        logger.info("LaneDetector initialised.")

    # ── Internals ─────────────────────────────────────────────────────

    def _roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Apply trapezoidal ROI mask — keep bottom portion of frame."""
        h, w = frame.shape[:2]
        mask = np.zeros_like(frame)
        top_y = int(h * self._roi_top)
        poly  = np.array([[
            (0, h), (0, top_y), (w, top_y), (w, h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255 if frame.ndim == 2 else (255, 255, 255))
        return cv2.bitwise_and(frame, mask)

    def _fit_line(
        self,
        segments: List[Tuple[int, int, int, int]],
        frame_h: int,
        frame_w: int,
    ) -> Optional[LaneLine]:
        """
        Average a list of line segments into a single full-frame line.

        Args:
            segments: List of (x1, y1, x2, y2) raw Hough segments.
            frame_h:  Frame height (for extrapolation).
            frame_w:  Frame width (for clamping).

        Returns:
            Averaged LaneLine or None if no valid segments.
        """
        if not segments:
            return None

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in segments:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        # Fit a polynomial of degree 1 (straight line): x = m*y + b
        try:
            coeffs = np.polyfit(y_coords, x_coords, 1)
        except np.linalg.LinAlgError:
            return None

        poly  = np.poly1d(coeffs)
        y_bot = frame_h
        y_top = int(frame_h * self._roi_top)
        x_bot = int(np.clip(poly(y_bot), 0, frame_w - 1))
        x_top = int(np.clip(poly(y_top), 0, frame_w - 1))
        return LaneLine(x1=x_bot, y1=y_bot, x2=x_top, y2=y_top)

    # ── Public API ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        Detect lane lines in the given frame.

        Args:
            frame: Preprocessed BGR frame.

        Returns:
            LaneResult with left/right lines and boundary pixels.
            On failure, returns last known good result.
        """
        h, w = frame.shape[:2]

        # Dynamically scale calibration points on first frame
        if not self.calibrator.is_calibrated and self._cfg_src and self._cfg_dst:
            scaled_src = []
            for px, py in self._cfg_src:
                scaled_src.append([int(px * w) if px <= 1.5 else px, int(py * h) if py <= 1.5 else py])
            scaled_dst = []
            for px, py in self._cfg_dst:
                scaled_dst.append([int(px * w) if px <= 1.5 else px, int(py * h) if py <= 1.5 else py])
            self.calibrator.compute_homography(scaled_src, scaled_dst)

        try:
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges   = cv2.Canny(blurred, self._canny_low, self._canny_high)
            masked  = self._roi_mask(edges)

            raw_lines = cv2.HoughLinesP(
                masked,
                rho=self._hough_rho,
                theta=self._hough_theta,
                threshold=self._hough_thresh,
                minLineLength=self._min_len,
                maxLineGap=self._max_gap,
            )

            left_segs: List[Tuple] = []
            right_segs: List[Tuple] = []

            if raw_lines is not None:
                for line in raw_lines:
                    x1, y1, x2, y2 = line[0]
                    dx = x2 - x1
                    if abs(dx) < 1e-3:
                        continue
                    slope = (y2 - y1) / dx
                    # Filter near-horizontal and near-vertical noise
                    if abs(slope) < 0.3 or abs(slope) > 10.0:
                        continue
                    if slope < 0:
                        left_segs.append((x1, y1, x2, y2))
                    else:
                        right_segs.append((x1, y1, x2, y2))

            left_line  = self._fit_line(left_segs,  h, w)
            right_line = self._fit_line(right_segs, h, w)

            # Collect boundary x-coords at bottom of frame
            boundaries = []
            if left_line:
                boundaries.append(left_line.x1)
            if right_line:
                boundaries.append(right_line.x1)
            boundaries.sort()

            result = LaneResult(
                left=left_line,
                right=right_line,
                boundaries_px=boundaries,
                frame_width=w,
            )
            self._last_result = result
            return result

        except Exception as e:
            logger.warning(f"LaneDetector: detection failed — {e}. Using last result.")
            if self._last_result is not None:
                return self._last_result
            return LaneResult(left=None, right=None, frame_width=w)

    def assign_lane(self, centroid_x: int, result: LaneResult) -> int:
        """
        Determine which lane index (0 = leftmost) a vehicle is in.

        Args:
            centroid_x: Vehicle centroid x in pixels.
            result:     LaneResult from detect().

        Returns:
            Lane index (integer, 0-based).
        """
        bounds = sorted(result.boundaries_px)
        for i, bx in enumerate(bounds):
            if centroid_x < bx:
                return i
        return len(bounds)  # rightmost lane

    def count_lane_line_crossings(
        self,
        centroids: List[Tuple[int, int]],
        result: LaneResult,
    ) -> int:
        """
        Count how many times a trajectory crosses a lane boundary.
        Used in zigzag detection for the lane-crossing counter.

        Args:
            centroids: Ordered list of (cx, cy) pixel centroids.
            result:    Current LaneResult.

        Returns:
            Number of lane boundary crossings.
        """
        if len(centroids) < 2 or not result.boundaries_px:
            return 0

        crossings = 0
        prev_lane = self.assign_lane(centroids[0][0], result)
        for cx, cy in centroids[1:]:
            curr_lane = self.assign_lane(cx, result)
            if curr_lane != prev_lane:
                crossings += 1
            prev_lane = curr_lane
        return crossings

    def draw_lanes(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """
        Draw lane lines and a filled polygon on the frame.

        Args:
            frame:  BGR image.
            result: LaneResult.

        Returns:
            Annotated frame.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        left  = result.left
        right = result.right

        # Draw individual lines
        if left:
            cv2.line(frame, (left.x1, left.y1), (left.x2, left.y2), (255, 0, 0), 3)
        if right:
            cv2.line(frame, (right.x1, right.y1), (right.x2, right.y2), (0, 0, 255), 3)

        # Fill lane polygon
        if left and right:
            pts = np.array([
                [left.x1, left.y1],
                [left.x2, left.y2],
                [right.x2, right.y2],
                [right.x1, right.y1],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        return frame
