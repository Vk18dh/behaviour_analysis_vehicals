"""
src/preprocessing/dip.py
Classical Digital Image Processing pipeline.
Stages (in order):
  1. White Balance (grey-world)
  2. Defogging (Dark Channel Prior) — optional
  3. Gaussian Blur
  4. Median Filter
  5. CLAHE on LAB L-channel
  6. Gamma Correction
  7. Frame Stabilization — optional
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Individual DIP Operations
# ══════════════════════════════════════════════════════════════════════

def white_balance(frame: np.ndarray) -> np.ndarray:
    """
    Grey-world white balance: scale each channel so their means are equal.

    Args:
        frame: BGR uint8 image.

    Returns:
        White-balanced BGR image.
    """
    result = frame.copy().astype(np.float32)
    b_mean = result[:, :, 0].mean()
    g_mean = result[:, :, 1].mean()
    r_mean = result[:, :, 2].mean()
    overall_mean = (b_mean + g_mean + r_mean) / 3.0
    if b_mean > 0:
        result[:, :, 0] *= overall_mean / b_mean
    if g_mean > 0:
        result[:, :, 1] *= overall_mean / g_mean
    if r_mean > 0:
        result[:, :, 2] *= overall_mean / r_mean
    return np.clip(result, 0, 255).astype(np.uint8)


def dark_channel_prior_defog(
    frame: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
) -> np.ndarray:
    """
    Defog (dehaze) using the Dark Channel Prior (He et al. 2011).

    Args:
        frame:      BGR uint8 image.
        patch_size: Local patch radius for dark channel estimate.
        omega:      Haze removal strength (0–1). Higher = more aggressive.
        t0:         Minimum transmission to avoid division by near-zero.

    Returns:
        Dehazed BGR image.
    """
    I = frame.astype(np.float64) / 255.0

    # Dark channel: min over patch and channels
    dark = cv2.erode(
        np.min(I, axis=2),
        np.ones((patch_size, patch_size), np.uint8),
    )

    # Atmospheric light: top 0.1% brightest pixels in dark channel
    flat_dark = dark.flatten()
    flat_I    = I.reshape(-1, 3)
    n_top     = max(1, int(len(flat_dark) * 0.001))
    indices   = np.argsort(flat_dark)[-n_top:]
    A         = flat_I[indices].max(axis=0)  # per-channel atmospheric light
    A         = np.clip(A, 0.001, 1.0)

    # Transmission estimate
    dark_norm = cv2.erode(
        np.min(I / A, axis=2),
        np.ones((patch_size, patch_size), np.uint8),
    )
    transmission = np.clip(1.0 - omega * dark_norm, t0, 1.0)
    t3 = transmission[:, :, np.newaxis]

    # Scene radiance recovery
    J = (I - A) / t3 + A
    J = np.clip(J * 255, 0, 255).astype(np.uint8)
    return J


def gaussian_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Gaussian blur for noise smoothing. kernel_size must be odd."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def median_filter(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter for salt-and-pepper noise removal. kernel_size must be odd."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(frame, k)


def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation on the L-channel of LAB.

    Args:
        frame:      BGR uint8 image.
        clip_limit: CLAHE contrast limit.
        tile_grid:  Grid size for contextual regions.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correction(frame: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Gamma correction using a lookup table (fast).

    Args:
        frame: BGR uint8 image.
        gamma: Correction factor.
               > 1 → brightens (useful for low-light footage).
               < 1 → darkens.

    Returns:
        Gamma-corrected image.
    """
    if abs(gamma - 1.0) < 1e-4:
        return frame
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, lut)


# ══════════════════════════════════════════════════════════════════════
# Video Stabilizer
# ══════════════════════════════════════════════════════════════════════

class FrameStabilizer:
    """
    Simple feature-point based inter-frame stabilization.
    Estimates affine transform between consecutive frames via
    FAST corner detection + optical flow + RANSAC.
    """

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts:  Optional[np.ndarray] = None

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize `frame` relative to the previous frame.

        Args:
            frame: Current BGR frame.

        Returns:
            Stabilized BGR frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return frame

        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        if prev_pts is None or len(prev_pts) < 10:
            self._prev_gray = gray
            return frame

        # Track points into current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None
        )
        valid_prev = prev_pts[status.ravel() == 1]
        valid_curr = curr_pts[status.ravel() == 1]

        if len(valid_prev) < 6:
            self._prev_gray = gray
            return frame

        # Estimate affine with RANSAC
        M, _ = cv2.estimateAffinePartial2D(
            valid_prev, valid_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if M is None:
            self._prev_gray = gray
            return frame

        # Invert transform to cancel motion
        M_inv = np.eye(2, 3, dtype=np.float64)
        M_inv[:2, :2] = M[:2, :2].T
        M_inv[:, 2]   = -M[:2, :2].T @ M[:, 2]

        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, M_inv, (w, h))
        self._prev_gray = gray
        return stabilized


# ══════════════════════════════════════════════════════════════════════
# Main Preprocessor
# ══════════════════════════════════════════════════════════════════════

class DIPPreprocessor:
    """
    Executes the full DIP preprocessing pipeline in the correct order.

    Instantiate once; call preprocess(frame) on every frame.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: The 'preprocessing' section of settings.yaml.
        """
        self._cfg       = cfg
        self._stabilizer = FrameStabilizer() if cfg.get("stabilize", False) else None
        logger.info(
            f"DIPPreprocessor init — "
            f"defog={cfg.get('defog', False)}, "
            f"stabilize={cfg.get('stabilize', False)}, "
            f"gamma={cfg.get('gamma', 1.0)}"
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Run full pipeline on a single BGR frame.

        Pipeline order:
          1. White Balance
          2. Defogging (optional)
          3. Gaussian Blur
          4. Median Filter
          5. CLAHE
          6. Gamma Correction
          7. Stabilization (optional)

        Args:
            frame: Raw BGR uint8 frame from VideoSource.

        Returns:
            Cleaned BGR frame.
        """
        if frame is None or frame.size == 0:
            logger.warning("DIPPreprocessor received empty frame — skipping.")
            return frame

        out = frame

        # 1. White balance
        if self._cfg.get("white_balance", True):
            out = white_balance(out)

        # 2. Defogging (slow — only for heavy fog/haze conditions)
        if self._cfg.get("defog", False):
            try:
                out = dark_channel_prior_defog(out)
            except Exception as e:
                logger.warning(f"Defogging failed: {e}")

        # 3. Gaussian blur
        k_gauss = self._cfg.get("gaussian_blur_kernel", 5)
        if k_gauss > 1:
            out = gaussian_blur(out, k_gauss)

        # 4. Median filter
        k_med = self._cfg.get("median_blur_kernel", 3)
        if k_med > 1:
            out = median_filter(out, k_med)

        # 5. CLAHE
        clip  = self._cfg.get("clahe_clip_limit", 2.0)
        grid  = tuple(self._cfg.get("clahe_tile_grid", [8, 8]))
        out   = apply_clahe(out, clip_limit=clip, tile_grid=grid)

        # 6. Gamma correction
        gamma = self._cfg.get("gamma", 1.2)
        out   = gamma_correction(out, gamma=gamma)

        # 7. Stabilization
        if self._stabilizer is not None:
            out = self._stabilizer.stabilize(out)

        return out

    @staticmethod
    def mean_brightness(frame: np.ndarray) -> float:
        """Return mean pixel brightness (0–255) of a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
