"""
src/utils/helpers.py
Shared utility functions: geometry, drawing, config loading, smoothing.
Used by every other module — import this, not individual modules.
"""

from __future__ import annotations

import math
import os
import threading
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


# ══════════════════════════════════════════════════════════════════════
# Config Loader (singleton via thread-safe lock)
# ══════════════════════════════════════════════════════════════════════

_CONFIG_LOCK = threading.Lock()
_CONFIG_CACHE: Optional[dict] = None


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Load YAML config exactly once (singleton).  Thread-safe.
    Resolves path relative to project root, CWD, or CONFIG_PATH env var.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    with _CONFIG_LOCK:
        if _CONFIG_CACHE is None:
            # Allow env var override (useful for tests)
            env_path = os.environ.get("CONFIG_PATH")
            candidates = []
            if env_path:
                candidates.append(Path(env_path))
            candidates.append(Path(config_path))
            # Try relative to this file's location (src/utils/helpers.py → ../../config/)
            candidates.append(Path(__file__).parent.parent.parent / config_path)

            path = None
            for c in candidates:
                if c.exists():
                    path = c
                    break

            if path is None:
                raise FileNotFoundError(
                    f"Config not found in any of: {[str(c) for c in candidates]}. "
                    "Run from project root (behaviorpbl/) or set CONFIG_PATH env var."
                )
            with path.open("r", encoding="utf-8") as f:
                _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


# ══════════════════════════════════════════════════════════════════════
# Geometry Utilities
# ══════════════════════════════════════════════════════════════════════

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def vector_dot(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Dot product of two 2-D vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1]


def vector_normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    """Return unit vector. Returns (0, 0) for zero-length input."""
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2)
    if mag < 1e-9:
        return (0.0, 0.0)
    return (v[0] / mag, v[1] / mag)


def vector_angle_deg(v: Tuple[float, float]) -> float:
    """Angle of vector from positive X-axis, degrees (−180 to 180)."""
    return math.degrees(math.atan2(v[1], v[0]))


def angle_between_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Signed angle from v1 to v2 in degrees."""
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    diff = math.degrees(a2 - a1)
    # Normalise to [-180, 180]
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def poly_contains_point(polygon: List[Tuple[int, int]],
                        point: Tuple[float, float]) -> bool:
    """
    Test if a point is inside a polygon using OpenCV's pointPolygonTest.

    Args:
        polygon: List of (x, y) integer vertices.
        point: (x, y) query point.

    Returns:
        True if inside or on boundary.
    """
    pts = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return result >= 0


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Return the centre of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2)

    Returns:
        (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """Area of bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(b1: Tuple[int, int, int, int],
             b2: Tuple[int, int, int, int]) -> float:
    """Intersection over Union for two bounding boxes."""
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    union = bbox_area(b1) + bbox_area(b2) - inter
    return inter / union if union > 0 else 0.0


def pixel_to_world(H: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    """
    Apply homography matrix H to convert pixel → world (metres) coordinates.

    Args:
        H: 3×3 homography matrix (from cv2.getPerspectiveTransform).
        px, py: Pixel coordinates.

    Returns:
        (x_m, y_m) in metres.
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0][0][0]), float(world[0][0][1])


# ══════════════════════════════════════════════════════════════════════
# Signal / Smoothing Utilities
# ══════════════════════════════════════════════════════════════════════

def smooth_values(history: Deque[float], window: int) -> float:
    """Moving average of the last `window` values in history."""
    if not history:
        return 0.0
    values = list(history)[-window:]
    return sum(values) / len(values)


def count_sign_changes(sequence: Sequence[float]) -> int:
    """
    Count the number of times the sign of the sequence flips
    (ignoring zero values).  Used for zigzag lateral direction changes.

    Args:
        sequence: Series of signed float values (e.g. Δx per frame).

    Returns:
        Number of sign change events.
    """
    changes = 0
    prev_sign: Optional[int] = None
    for val in sequence:
        if abs(val) < 1e-9:
            continue  # skip zero / near-zero
        curr_sign = 1 if val > 0 else -1
        if prev_sign is not None and curr_sign != prev_sign:
            changes += 1
        prev_sign = curr_sign
    return changes


def compute_lateral_accel(dx_history: Deque[float], dt: float) -> float:
    """
    Proxy lateral acceleration: a_y ≈ Δx / Δt²
    Using two consecutive lateral displacement values.

    Args:
        dx_history: Recent lateral displacements (metres per frame).
        dt: Time per frame in seconds.

    Returns:
        Estimated lateral acceleration in m/s².
    """
    if len(dx_history) < 2:
        return 0.0
    vals = list(dx_history)
    dx1 = vals[-2]
    dx2 = vals[-1]
    # acceleration = change in velocity / time = (Δx2 - Δx1) / dt^2
    if dt < 1e-9:
        return 0.0
    return abs((dx2 - dx1) / (dt ** 2))


# ══════════════════════════════════════════════════════════════════════
# Drawing Utilities
# ══════════════════════════════════════════════════════════════════════

# Violation-type colour palette
VIOLATION_COLORS: dict[str, Tuple[int, int, int]] = {
    "ZIGZAG":               (0, 0, 255),     # red
    "TAILGATING":           (0, 128, 255),   # orange
    "RED_LIGHT":            (0, 0, 200),     # dark red
    "OVERSPEED":            (255, 0, 255),   # magenta
    "WRONG_DIRECTION":      (0, 0, 180),     # crimson
    "HIGHWAY_RESTRICTION":  (0, 165, 255),   # orange
    "LANE_VIOLATION":       (255, 0, 0),     # blue (BGR)
    "RASH_DRIVING":         (0, 0, 255),
    "NO_HELMET":            (0, 255, 255),   # yellow
    "NO_SEATBELT":          (0, 215, 255),   # gold
    "TRIPLE_RIDING":        (180, 0, 100),
    "ILLEGAL_TURN":         (255, 100, 0),
    "PHONE_USE":            (100, 0, 200),
}
DEFAULT_COLOR = (0, 0, 255)


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw a bounding box with an optional label on `frame` (in-place).

    Args:
        frame: BGR image array.
        bbox: (x1, y1, x2, y2).
        label: Text to display above the box.
        color: BGR colour.
        thickness: Line thickness.
        font_scale: Font size.

    Returns:
        Annotated frame (same object, modified in-place).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        bg_y1 = max(y1 - th - 6, 0)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def draw_lane_overlay(
    frame: np.ndarray,
    left_line: Optional[Tuple[int, int, int, int]],
    right_line: Optional[Tuple[int, int, int, int]],
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Draw semi-transparent lane fill between left and right lines.

    Args:
        frame: BGR image.
        left_line: (x1, y1, x2, y2) or None.
        right_line: (x1, y1, x2, y2) or None.
        alpha: Transparency (0=invisible, 1=solid).

    Returns:
        Annotated frame.
    """
    if left_line is None and right_line is None:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]

    def line_pts(line):
        return [(line[0], line[1]), (line[2], line[3])]

    if left_line and right_line:
        pts = np.array([
            (left_line[0], left_line[1]),
            (right_line[0], right_line[1]),
            (right_line[2], right_line[3]),
            (left_line[2], left_line[3]),
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    elif left_line:
        cv2.line(overlay, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), (255, 0, 0), 3)
    elif right_line:
        cv2.line(overlay, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), (0, 0, 255), 3)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def put_text_block(
    frame: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int],
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    line_height: int = 20,
) -> np.ndarray:
    """
    Render multiple lines of text with a dark background box.

    Args:
        frame: BGR image.
        lines: Text lines to render.
        origin: Top-left (x, y) for the text block.
        font_scale: Font size.
        color: Text colour (BGR).
        bg_color: Background rectangle colour.
        line_height: Pixels between lines.

    Returns:
        Annotated frame.
    """
    x, y = origin
    pad = 4
    max_w = max(
        cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for l in lines
    ) if lines else 0
    total_h = line_height * len(lines)
    cv2.rectangle(
        frame,
        (x - pad, y - line_height),
        (x + max_w + pad, y + total_h - line_height + pad),
        bg_color, -1,
    )
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )
    return frame
