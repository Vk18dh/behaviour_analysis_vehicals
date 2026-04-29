"""
src/preprocessing/dip.py
Adaptive Digital Image Processing pipeline.

Per-frame problem detection + targeted fix application:
  Problem          │ Metric                      │ Fix
  ─────────────────┼─────────────────────────────┼────────────────────────────
  LOW_LIGHT        │ mean brightness < 80        │ aggressive CLAHE + gamma 1.8
  OVEREXPOSED      │ mean brightness > 200       │ gamma 0.7 (darken)
  BLUR             │ Laplacian variance < 100    │ unsharp-mask sharpening
  NOISE            │ local pixel std dev > 28    │ median filter k=5
  HAZE_FOG         │ contrast (std) < 35         │ Dark Channel Prior dehaze
  COLOR_CAST       │ channel mean deviation > 15 │ grey-world white balance

`preprocess(frame)` returns `(processed_frame, DIPReport)`.
A rolling buffer of the last 500 reports is kept on the instance
and exposed via `.recent_reports` for the dashboard / API.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Module-level active-instance registry
# ══════════════════════════════════════════════════════════════════════
# Any code that creates a DIPPreprocessor for a live pipeline should
# call register_dip_instance().  The /dip/stats API reads from here.
# Thread-safe for single-writer / multiple-reader use.

_ACTIVE_DIP_INSTANCE: Optional["DIPPreprocessor"] = None


def register_dip_instance(instance: "DIPPreprocessor") -> None:
    """Register the currently active DIPPreprocessor for API access."""
    global _ACTIVE_DIP_INSTANCE
    _ACTIVE_DIP_INSTANCE = instance


def get_active_dip_instance() -> Optional["DIPPreprocessor"]:
    """Return the currently active DIPPreprocessor, or None."""
    return _ACTIVE_DIP_INSTANCE


# ══════════════════════════════════════════════════════════════════════
# DIPReport — per-frame processing record
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DIPReport:
    """Records what problems were detected and which fixes were applied."""
    frame_idx:         int
    timestamp:         float = field(default_factory=time.time)

    # Problems detected BEFORE processing
    problems_detected: List[str] = field(default_factory=list)
    # Fixes applied (subset matching the problems)
    fixes_applied:     List[str] = field(default_factory=list)

    # Quality metrics BEFORE processing
    brightness_before: float = 0.0
    contrast_before:   float = 0.0   # pixel std-dev
    blur_score_before: float = 0.0   # Laplacian variance
    noise_level:       float = 0.0   # mean local std-dev

    # Quality metrics AFTER processing
    brightness_after:  float = 0.0
    contrast_after:    float = 0.0
    blur_score_after:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_idx":         self.frame_idx,
            "timestamp":         self.timestamp,
            "problems_detected": self.problems_detected,
            "fixes_applied":     self.fixes_applied,
            "brightness_before": round(self.brightness_before, 1),
            "brightness_after":  round(self.brightness_after,  1),
            "contrast_before":   round(self.contrast_before,   1),
            "contrast_after":    round(self.contrast_after,    1),
            "blur_score_before": round(self.blur_score_before, 1),
            "blur_score_after":  round(self.blur_score_after,  1),
            "noise_level":       round(self.noise_level,       1),
        }


# ══════════════════════════════════════════════════════════════════════
# Frame Quality Analyser
# ══════════════════════════════════════════════════════════════════════

def _measure_frame(gray: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute quality metrics from a grayscale image.

    Returns:
        (brightness, contrast, blur_score, noise_level)
        brightness  — mean pixel value (0–255)
        contrast    — standard deviation of pixel values
        blur_score  — Laplacian variance (higher = sharper)
        noise_level — mean of local 3×3 std-dev patches
    """
    brightness  = float(gray.mean())
    contrast    = float(gray.std())
    blur_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Noise: compute local std in 3x3 patches via erosion trick
    gray_f = gray.astype(np.float32)
    mean_sq = cv2.blur(gray_f ** 2, (3, 3))
    sq_mean = cv2.blur(gray_f,      (3, 3)) ** 2
    local_var = np.maximum(mean_sq - sq_mean, 0)
    noise_level = float(np.sqrt(local_var).mean())

    return brightness, contrast, blur_score, noise_level


def detect_frame_problems(
    frame: np.ndarray,
    brightness_low:  float = 80.0,
    brightness_high: float = 200.0,
    blur_threshold:  float = 100.0,
    noise_threshold: float = 28.0,
    haze_contrast:   float = 35.0,
    color_cast_thr:  float = 15.0,
) -> Tuple[List[str], float, float, float, float]:
    """
    Analyse a BGR frame for quality problems.

    Returns:
        (problems, brightness, contrast, blur_score, noise_level)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness, contrast, blur_score, noise_level = _measure_frame(gray)

    problems: List[str] = []

    if brightness < brightness_low:
        problems.append("LOW_LIGHT")
    elif brightness > brightness_high:
        problems.append("OVEREXPOSED")

    if blur_score < blur_threshold:
        problems.append("BLUR")

    if noise_level > noise_threshold:
        problems.append("NOISE")

    # Haze/Fog: low contrast + medium brightness + low noise.
    # The noise_level guard is critical: pure random/noisy frames also have
    # low contrast but are NOT hazy — they have high local std-dev (noise).
    # Real haze has smooth, low-frequency structure (noise_level < 12).
    if contrast < haze_contrast and brightness > 100 and noise_level < 12.0:
        problems.append("HAZE_FOG")

    # Color cast: check per-channel mean deviation
    b_mean = float(frame[:, :, 0].mean())
    g_mean = float(frame[:, :, 1].mean())
    r_mean = float(frame[:, :, 2].mean())
    overall = (b_mean + g_mean + r_mean) / 3.0
    if max(abs(b_mean - overall),
           abs(g_mean - overall),
           abs(r_mean - overall)) > color_cast_thr:
        problems.append("COLOR_CAST")

    return problems, brightness, contrast, blur_score, noise_level


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
    max_side: int = 480,
) -> np.ndarray:
    """
    Defog (dehaze) using the Dark Channel Prior (He et al. 2011).

    Performance optimisation: the expensive morphological operations are
    performed on a downscaled copy (max_side pixels on the longest edge),
    and the resulting transmission map is upscaled back to the original
    resolution before applying scene radiance recovery.  This cuts
    computation time from ~300 ms to ~40 ms on 1080p frames with
    negligible visual quality loss.

    Args:
        frame:      BGR uint8 image.
        patch_size: Local patch radius for dark channel estimate.
        omega:      Haze removal strength (0–1). Higher = more aggressive.
        t0:         Minimum transmission to avoid division by near-zero.
        max_side:   Longest edge of the downscaled processing frame (px).
                    Set to 0 to disable downscaling.

    Returns:
        Dehazed BGR image (same size as input).
    """
    orig_h, orig_w = frame.shape[:2]

    # ── Adaptive downscale for speed ─────────────────────────────────
    if max_side > 0 and max(orig_h, orig_w) > max_side:
        scale   = max_side / max(orig_h, orig_w)
        small_w = max(1, int(orig_w * scale))
        small_h = max(1, int(orig_h * scale))
        small   = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    else:
        small = frame

    I = small.astype(np.float64) / 255.0

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
    A         = flat_I[indices].max(axis=0)
    A         = np.clip(A, 0.001, 1.0)

    # Transmission estimate (on small frame)
    dark_norm    = cv2.erode(
        np.min(I / A, axis=2),
        np.ones((patch_size, patch_size), np.uint8),
    )
    transmission_small = np.clip(1.0 - omega * dark_norm, t0, 1.0)

    # ── Upsample transmission map to original resolution ─────────────
    if small.shape[:2] != (orig_h, orig_w):
        transmission = cv2.resize(
            transmission_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        transmission = transmission_small

    # Scene radiance recovery on full-resolution frame
    I_full = frame.astype(np.float64) / 255.0
    t3 = transmission[:, :, np.newaxis]
    J  = (I_full - A) / t3 + A
    J  = np.clip(J * 255, 0, 255).astype(np.uint8)
    return J


def gaussian_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Gaussian blur for noise smoothing. kernel_size must be odd."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def median_filter(frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter for salt-and-pepper noise removal. kernel_size must be odd."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(frame, k)


def unsharp_mask(frame: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Unsharp mask sharpening to restore detail in blurry frames.

    Args:
        frame:    BGR uint8 image.
        strength: Sharpening amount (1.0 = mild, 2.0 = strong).

    Returns:
        Sharpened BGR image.
    """
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    sharp   = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


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
# Main Preprocessor — Adaptive
# ══════════════════════════════════════════════════════════════════════

class DIPPreprocessor:
    """
    Adaptive DIP preprocessing pipeline.

    Each frame is analysed for quality problems FIRST, then only the
    necessary fixes are applied. Returns (processed_frame, DIPReport).

    Maintains a rolling buffer of the last `report_buffer_size` reports
    accessible via `.recent_reports` for dashboard / API use.

    Instantiate once; call preprocess(frame, frame_idx) on every frame.
    """

    # Thresholds for problem detection
    _BRIGHTNESS_LOW   = 80.0
    _BRIGHTNESS_HIGH  = 200.0
    _BLUR_THRESHOLD   = 100.0
    _NOISE_THRESHOLD  = 28.0
    _HAZE_CONTRAST    = 35.0
    _COLOR_CAST_THR   = 15.0

    def __init__(self, cfg: dict, report_buffer_size: int = 500) -> None:
        """
        Args:
            cfg: The 'preprocessing' section of settings.yaml.
            report_buffer_size: Max DIPReports kept in memory.
        """
        self._cfg        = cfg
        self._stabilizer = FrameStabilizer() if cfg.get("stabilize", False) else None

        # Configurable overrides from settings.yaml
        self._brightness_low  = cfg.get("brightness_low_threshold",  self._BRIGHTNESS_LOW)
        self._brightness_high = cfg.get("brightness_high_threshold", self._BRIGHTNESS_HIGH)
        self._blur_threshold  = cfg.get("blur_threshold",            self._BLUR_THRESHOLD)
        self._noise_threshold = cfg.get("noise_threshold",           self._NOISE_THRESHOLD)
        self._haze_contrast   = cfg.get("haze_contrast_threshold",   self._HAZE_CONTRAST)
        self._color_cast_thr  = cfg.get("color_cast_threshold",      self._COLOR_CAST_THR)

        # ── Pre-create CLAHE objects (avoids per-frame allocation cost) ──
        # Standard CLAHE for normal frames
        clip  = cfg.get("clahe_clip_limit", 2.0)
        grid  = tuple(cfg.get("clahe_tile_grid", [8, 8]))
        self._clahe_standard = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        # Aggressive CLAHE for low-light frames
        self._clahe_llie     = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

        # Rolling report buffer (thread-safe reads, single writer)
        self.recent_reports: Deque[DIPReport] = deque(maxlen=report_buffer_size)

        logger.info(
            f"DIPPreprocessor (adaptive) init — "
            f"stabilize={cfg.get('stabilize', False)}, "
            f"defog={cfg.get('defog', False)}, "
            f"buffer={report_buffer_size}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def preprocess(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[np.ndarray, DIPReport]:
        """
        Adaptively process a single BGR frame.

        Steps:
          1. Measure frame quality (brightness, contrast, blur, noise).
          2. Detect which problems are present.
          3. Apply targeted fixes for each detected problem.
          4. Re-measure after processing.
          5. Store and return DIPReport.

        Args:
            frame:     Raw BGR uint8 frame from VideoSource.
            frame_idx: Frame number (for report logging).

        Returns:
            (processed_frame, DIPReport)
        """
        if frame is None or frame.size == 0:
            logger.warning("DIPPreprocessor received empty frame — skipping.")
            return frame, DIPReport(frame_idx=frame_idx)

        # ── 1. Measure before ─────────────────────────────────────────
        problems, brightness_b, contrast_b, blur_b, noise_lv = detect_frame_problems(
            frame,
            brightness_low  = self._brightness_low,
            brightness_high = self._brightness_high,
            blur_threshold  = self._blur_threshold,
            noise_threshold = self._noise_threshold,
            haze_contrast   = self._haze_contrast,
            color_cast_thr  = self._color_cast_thr,
        )

        report = DIPReport(
            frame_idx         = frame_idx,
            problems_detected = problems,
            brightness_before = brightness_b,
            contrast_before   = contrast_b,
            blur_score_before = blur_b,
            noise_level       = noise_lv,
        )

        out = frame.copy()

        # ── 2. Apply targeted fixes (ORDER MATTERS) ───────────────────

        # Fix: Color cast → grey-world white balance (fast, run first)
        if "COLOR_CAST" in problems:
            out = white_balance(out)
            report.fixes_applied.append("WHITE_BALANCE")

        # Fix: Fog/Haze → Dark Channel Prior.
        # ONLY runs when HAZE_FOG is actually detected AND defog is enabled.
        # This is the most expensive operation (~40-300 ms); do NOT apply
        # unconditionally. Uses adaptive downscale for 1080p+ performance.
        if "HAZE_FOG" in problems and self._cfg.get("defog", False):
            try:
                out = dark_channel_prior_defog(out)
                report.fixes_applied.append("DEFOG")
            except Exception as e:
                logger.warning(f"Defogging failed: {e}")

        # Fix: Noise → median filter (stronger kernel for heavy noise)
        if "NOISE" in problems:
            k = 5 if noise_lv > 40 else 3
            out = median_filter(out, kernel_size=k)
            report.fixes_applied.append(f"MEDIAN_FILTER_K{k}")
        else:
            # Light baseline median (configurable; 0 or 1 = skip)
            k_med = self._cfg.get("median_blur_kernel", 3)
            if k_med > 1:
                out = median_filter(out, k_med)

        # Fix: Blur → unsharp mask sharpening
        if "BLUR" in problems:
            strength = 2.0 if blur_b < 30 else 1.2
            out = unsharp_mask(out, strength=strength)
            report.fixes_applied.append(f"UNSHARP_MASK_S{strength:.1f}")
        else:
            # Baseline Gaussian (optional; 0 = skip)
            k_gauss = self._cfg.get("gaussian_blur_kernel", 0)
            if k_gauss > 1:
                out = gaussian_blur(out, k_gauss)

        # Fix: Low-light → aggressive CLAHE + bright gamma
        # Uses pre-created CLAHE object (faster than creating per frame)
        if "LOW_LIGHT" in problems:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_eq = self._clahe_llie.apply(l_ch)
            out  = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
            out  = gamma_correction(out, gamma=1.8)
            report.fixes_applied.append("LLIE_CLAHE_GAMMA1.8")

        # Fix: Overexposure → darkening gamma
        elif "OVEREXPOSED" in problems:
            out = gamma_correction(out, gamma=0.65)
            report.fixes_applied.append("GAMMA_DARKEN0.65")

        else:
            # Normal exposure → standard CLAHE (cached) + mild gamma
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_eq = self._clahe_standard.apply(l_ch)
            out  = cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
            gamma_val = self._cfg.get("gamma", 1.1)
            out = gamma_correction(out, gamma=gamma_val)
            report.fixes_applied.append("CLAHE_STANDARD")

        # Fix: Camera shake → optical-flow stabilization (always if enabled)
        if self._stabilizer is not None:
            out = self._stabilizer.stabilize(out)
            if "STABILIZE" not in report.fixes_applied:
                report.fixes_applied.append("STABILIZE")

        # ── 3. Measure after ──────────────────────────────────────────
        gray_after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        brightness_a, contrast_a, blur_a, _ = _measure_frame(gray_after)
        report.brightness_after = brightness_a
        report.contrast_after   = contrast_a
        report.blur_score_after = blur_a

        # ── 4. Store report ───────────────────────────────────────────
        self.recent_reports.append(report)

        if problems:
            logger.debug(
                f"[frame {frame_idx}] DIP: {problems} → {report.fixes_applied}"
            )

        return out, report

    def preprocess_for_display(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, DIPReport]:
        """
        Like preprocess() but also returns the original frame for side-by-side display.

        Returns:
            (original_frame, processed_frame, DIPReport)
        """
        original = frame.copy()
        processed, report = self.preprocess(frame, frame_idx)
        return original, processed, report

    @staticmethod
    def mean_brightness(frame: np.ndarray) -> float:
        """Return mean pixel brightness (0–255) of a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
