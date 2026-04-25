"""
src/input/video_input.py
Video source abstraction: webcam, RTSP stream, or uploaded file.
Supports multi-camera management and auto-retry on stream drop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SourceType(Enum):
    WEBCAM = auto()
    RTSP   = auto()
    FILE   = auto()


@dataclass
class FramePacket:
    """A single decoded frame plus metadata."""
    frame:      np.ndarray
    camera_id:  str
    frame_idx:  int
    timestamp:  float          # epoch seconds
    source_fps: float


class VideoSource:
    """
    Unified video source for webcam, RTSP, or file.

    Example:
        src = VideoSource.from_webcam(0, camera_id="cam_00")
        for packet in src.stream():
            process(packet.frame)
    """

    def __init__(
        self,
        uri: str | int,
        source_type: SourceType,
        camera_id: str = "cam_00",
        max_retry: int = 5,
        retry_delay: float = 2.0,
    ) -> None:
        self.uri        = uri
        self.source_type = source_type
        self.camera_id  = camera_id
        self.max_retry  = max_retry
        self.retry_delay = retry_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._source_fps: float = 25.0
        self._open()

    # ── Factories ────────────────────────────────────────────────────

    @classmethod
    def from_webcam(cls, index: int = 0, **kwargs) -> "VideoSource":
        return cls(uri=index, source_type=SourceType.WEBCAM, **kwargs)

    @classmethod
    def from_rtsp(cls, url: str, **kwargs) -> "VideoSource":
        return cls(uri=url, source_type=SourceType.RTSP, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "VideoSource":
        if not Path(path).exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        return cls(uri=path, source_type=SourceType.FILE, **kwargs)

    # ── Internal ─────────────────────────────────────────────────────

    def _open(self) -> None:
        """Open (or re-open) the capture device."""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.uri)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.uri}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps > 0 else 25.0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"[{self.camera_id}] Opened {self.source_type.name} "
            f"resolution={w}×{h} fps={self._source_fps:.1f}"
        )

    def _read_with_retry(self) -> Optional[np.ndarray]:
        """Read one frame; retry up to max_retry on failure."""
        for attempt in range(self.max_retry + 1):
            if self._cap is None or not self._cap.isOpened():
                if attempt < self.max_retry:
                    logger.warning(
                        f"[{self.camera_id}] Stream closed — retry {attempt+1}/{self.max_retry}"
                    )
                    time.sleep(self.retry_delay)
                    try:
                        self._open()
                    except IOError as e:
                        logger.error(f"[{self.camera_id}] Reconnect failed: {e}")
                    continue

            success, frame = self._cap.read()
            if success and frame is not None:
                return frame

            # Read failed
            if self.source_type == SourceType.FILE:
                return None  # end of file — normal termination

            logger.warning(
                f"[{self.camera_id}] Frame read failed — retry {attempt+1}/{self.max_retry}"
            )
            time.sleep(self.retry_delay)
            try:
                self._open()
            except IOError as e:
                logger.error(f"[{self.camera_id}] Reconnect failed: {e}")

        logger.error(f"[{self.camera_id}] Max retries exceeded. Stopping stream.")
        return None

    # ── Public API ───────────────────────────────────────────────────

    def read_frame(self) -> Tuple[bool, Optional[FramePacket]]:
        """
        Read one frame.

        Returns:
            (success, FramePacket or None)
        """
        frame = self._read_with_retry()
        if frame is None:
            return False, None
        packet = FramePacket(
            frame=frame,
            camera_id=self.camera_id,
            frame_idx=self._frame_idx,
            timestamp=time.time(),
            source_fps=self._source_fps,
        )
        self._frame_idx += 1
        return True, packet

    def stream(self) -> Generator[FramePacket, None, None]:
        """
        Generator yielding FramePackets until source ends or fails.

        Usage:
            for packet in src.stream():
                ...
        """
        while True:
            success, packet = self.read_frame()
            if not success or packet is None:
                break
            yield packet

    @property
    def total_frames(self) -> int:
        """Total frames (files only; −1 for live streams)."""
        if self._cap and self.source_type == SourceType.FILE:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    @property
    def source_fps(self) -> float:
        return self._source_fps

    def release(self) -> None:
        """Release the capture device."""
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(f"[{self.camera_id}] Source released.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# ══════════════════════════════════════════════════════════════════════
# Multi-Camera Manager
# ══════════════════════════════════════════════════════════════════════

class MultiCameraManager:
    """
    Manages multiple VideoSource instances and iterates round-robin.

    Example:
        mgr = MultiCameraManager()
        mgr.add(VideoSource.from_rtsp("rtsp://...", camera_id="cam_01"))
        mgr.add(VideoSource.from_webcam(0, camera_id="cam_00"))
        for packet in mgr.stream():
            ...
    """

    def __init__(self) -> None:
        self._sources: List[VideoSource] = []

    def add(self, source: VideoSource) -> None:
        self._sources.append(source)
        logger.info(f"MultiCameraManager: added {source.camera_id}")

    def add_from_config(self, cam_configs: list) -> None:
        """
        Build sources from the 'camera.rtsp_streams' config list.

        Args:
            cam_configs: List of dicts with 'id' and 'url' keys.
        """
        for cfg in cam_configs:
            src = VideoSource.from_rtsp(
                url=cfg["url"],
                camera_id=cfg["id"],
            )
            self.add(src)

    def stream(self) -> Generator[FramePacket, None, None]:
        """
        Round-robin generator over all sources.
        Stops only when ALL sources are exhausted.
        """
        active = list(range(len(self._sources)))
        while active:
            still_active = []
            for idx in active:
                success, packet = self._sources[idx].read_frame()
                if success and packet:
                    yield packet
                    still_active.append(idx)
                else:
                    logger.info(
                        f"Source {self._sources[idx].camera_id} exhausted — removing."
                    )
            active = still_active

    def release_all(self) -> None:
        for src in self._sources:
            src.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release_all()
