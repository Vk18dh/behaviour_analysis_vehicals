"""
src/preprocessing/llie.py
Low-Light Image Enhancement (LLIE) module.
- Classic path: extra CLAHE + aggressive gamma (always available)
- Deep path: placeholder for DRM/KinD PyTorch model (swap in weights here)
Triggers automatically when frame mean brightness < threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.preprocessing.dip import apply_clahe, gamma_correction
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLIEProcessor:
    """
    Low-Light Image Enhancement processor.

    Usage:
        llie = LLIEProcessor(cfg=config["llie"])
        if llie.is_dark(frame):
            frame = llie.enhance(frame)
    """

    def __init__(self, cfg: dict, dark_threshold: float = 80.0) -> None:
        """
        Args:
            cfg: 'llie' section from settings.yaml.
            dark_threshold: Mean brightness (0–255) below which LLIE fires.
        """
        self._enabled        = cfg.get("enabled", True)
        self._use_deep       = cfg.get("use_deep_model", False)
        self._model_path     = cfg.get("deep_model_path", "")
        self._dark_threshold = dark_threshold
        self._model          = None  # loaded lazily

        if self._use_deep and self._model_path:
            self._load_deep_model()
        elif self._use_deep:
            logger.warning(
                "LLIE deep model requested but no model path set — "
                "falling back to classical."
            )
            self._use_deep = False

        logger.info(
            f"LLIEProcessor init — enabled={self._enabled}, "
            f"deep={self._use_deep}, threshold={self._dark_threshold}"
        )

    # ── Deep Model Loader ─────────────────────────────────────────────

    def _load_deep_model(self) -> None:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │  SWAP POINT — Deep LLIE Model                               │
        │  Replace the body of this method to load your PyTorch       │
        │  DRM / KinD / Zero-DCE model.                               │
        │                                                             │
        │  Example:                                                   │
        │    import torch                                             │
        │    self._model = torch.load(self._model_path)               │
        │    self._model.eval()                                       │
        └─────────────────────────────────────────────────────────────┘
        """
        mp = Path(self._model_path)
        if not mp.exists():
            logger.warning(
                f"LLIE model not found at {mp} — "
                "falling back to classical enhancement."
            )
            self._use_deep = False
            return
        # ── TODO: replace with actual model loading ──────────────────
        logger.info(f"LLIE deep model placeholder loaded from {mp}")
        self._model = "PLACEHOLDER"  # remove when real model loaded

    # ── Enhancement Methods ───────────────────────────────────────────

    def _enhance_classical(self, frame: np.ndarray) -> np.ndarray:
        """
        Classical two-step enhancement: aggressive CLAHE + bright gamma.
        Fast, no GPU required.

        Args:
            frame: BGR uint8 image (dark).

        Returns:
            Brightened BGR image.
        """
        # Step 1: Aggressive CLAHE (higher clip limit for dark frames)
        enhanced = apply_clahe(frame, clip_limit=4.0, tile_grid=(4, 4))
        # Step 2: Strong gamma brightening (γ = 1.8)
        enhanced = gamma_correction(enhanced, gamma=1.8)
        return enhanced

    def _enhance_deep(self, frame: np.ndarray) -> np.ndarray:
        """
        Deep model inference path.

        ┌─────────────────────────────────────────────────────────────┐
        │  SWAP POINT — Deep Inference                                │
        │  Replace the body of this method with:                     │
        │    1. Pre-process frame to model input tensor               │
        │    2. Run model forward pass                                │
        │    3. Post-process output tensor back to BGR uint8          │
        │                                                             │
        │  Example (PyTorch + DRM):                                   │
        │    import torch                                             │
        │    inp = frame_to_tensor(frame)                             │
        │    with torch.no_grad():                                    │
        │        out = self._model(inp)                               │
        │    return tensor_to_frame(out)                              │
        └─────────────────────────────────────────────────────────────┘
        """
        logger.debug("LLIE deep model: using classical fallback (placeholder active).")
        return self._enhance_classical(frame)  # fallback until real model swapped in

    # ── Public API ────────────────────────────────────────────────────

    def is_dark(self, frame: np.ndarray) -> bool:
        """
        Return True if the frame mean brightness is below the threshold.

        Args:
            frame: BGR uint8 image.
        """
        if not self._enabled:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < self._dark_threshold

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a dark frame.  Automatically chooses deep or classical.

        Args:
            frame: BGR uint8 image confirmed to be dark.

        Returns:
            Enhanced BGR image.
        """
        if not self._enabled:
            return frame
        try:
            if self._use_deep and self._model is not None:
                return self._enhance_deep(frame)
            return self._enhance_classical(frame)
        except Exception as e:
            logger.error(f"LLIE enhancement failed: {e} — returning original frame.")
            return frame

    def enhance_if_dark(self, frame: np.ndarray) -> np.ndarray:
        """
        Combined check-and-enhance in one call.

        Args:
            frame: Raw BGR frame.

        Returns:
            Enhanced frame if dark, original frame otherwise.
        """
        if self.is_dark(frame):
            logger.debug("Dark frame detected — applying LLIE.")
            return self.enhance(frame)
        return frame
