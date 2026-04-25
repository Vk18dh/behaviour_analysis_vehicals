"""
src/input/frame_buffer.py
Thread-safe frame buffer with FPS governor.
Producer reads from VideoSource at native FPS.
Consumer reads at target pipeline FPS.
Stale frames are dropped when the queue is full.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from src.input.video_input import FramePacket, VideoSource
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameTimeoutError(Exception):
    """Raised when get_frame() times out."""


class FrameBuffer:
    """
    Thread-safe bounded queue used as a ring buffer between the
    video producer thread and the processing pipeline.

    Also maintains a rolling `ring_buffer` (deque) of the last
    N raw frames — used by EvidenceGenerator for ±2 s clip extraction.

    Args:
        maxsize: Maximum frames in the queue before oldest is dropped.
        ring_size: Size of the rolling ring buffer (default = fps * 4).
    """

    def __init__(self, maxsize: int = 128, ring_size: int = 100) -> None:
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=maxsize)
        self.ring_buffer: Deque[FramePacket] = deque(maxlen=ring_size)
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._dropped_frames: int = 0
        self._total_produced: int = 0

    # ── Producer ─────────────────────────────────────────────────────

    def start_producer(
        self,
        source: VideoSource,
        target_fps: float = 15.0,
    ) -> None:
        """
        Launch the producer thread that reads from `source`.

        Args:
            source: An opened VideoSource instance.
            target_fps: Desired pipeline FPS (governs sleep between reads).
        """
        self._stop_event.clear()
        self._producer_thread = threading.Thread(
            target=self._produce,
            args=(source, target_fps),
            daemon=True,
            name=f"producer-{source.camera_id}",
        )
        self._producer_thread.start()
        logger.info(
            f"[{source.camera_id}] Frame producer started "
            f"(target_fps={target_fps}, buffer_size={self._queue.maxsize})"
        )

    def _produce(self, source: VideoSource, target_fps: float) -> None:
        """Internal producer loop (runs in background thread)."""
        interval = 1.0 / max(target_fps, 1.0)
        while not self._stop_event.is_set():
            t_start = time.monotonic()
            success, packet = source.read_frame()
            if not success or packet is None:
                logger.info(f"[{source.camera_id}] Producer: source exhausted.")
                self._stop_event.set()
                break

            self._total_produced += 1

            # Always update the ring buffer (for clip extraction)
            self.ring_buffer.append(packet)

            # Enqueue for pipeline; drop oldest if full (non-blocking)
            if self._queue.full():
                try:
                    self._queue.get_nowait()  # discard oldest
                    self._dropped_frames += 1
                    if self._dropped_frames % 100 == 0:
                        logger.warning(
                            f"[{source.camera_id}] Dropped {self._dropped_frames} stale frames."
                        )
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                self._dropped_frames += 1  # race condition guard

            # FPS governor
            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── Consumer ─────────────────────────────────────────────────────

    def get_frame(self, timeout: float = 2.0) -> FramePacket:
        """
        Block until a frame is available or timeout expires.

        Args:
            timeout: Seconds to wait.

        Returns:
            FramePacket.

        Raises:
            FrameTimeoutError: If no frame arrives within timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise FrameTimeoutError(
                f"No frame received within {timeout}s. "
                "Check if the video source is running."
            )

    def get_frame_nowait(self) -> Optional[FramePacket]:
        """Non-blocking get. Returns None if queue is empty."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    # ── Control ──────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the producer to stop and wait for thread exit."""
        self._stop_event.set()
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5.0)
        logger.info(
            f"FrameBuffer stopped. "
            f"Produced={self._total_produced}, Dropped={self._dropped_frames}"
        )

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames
