"""
src/pipeline/realtime_pipeline.py
End-to-end live camera pipeline.

Full chain per frame:
  VideoSource → FrameBuffer → DIP → LLIE → LaneDetect → VehicleDetect
  → Track → Features → FollowingDist → Behaviors → ANPR → Evidence
  → RuleEngine → DB → (post-approval: Notifier)

Targets ≥ 15 FPS, < 100 ms per frame (logged every 100 frames).
Supports graceful shutdown via threading.Event.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from src.anpr.anpr import ANPRSystem
from src.behavior.behavior_engine import BehaviorEngine
from src.database.db import init_db, save_violation, sync_score_to_db
from src.detection.vehicle_detector import VehicleDetector
from src.evidence.evidence_generator import EvidenceGenerator
from src.features.feature_extractor import FeatureExtractor
from src.input.frame_buffer import FrameBuffer
from src.input.video_input import VideoSource
from src.lane.lane_detection import LaneDetector
from src.preprocessing.dip import DIPPreprocessor
from src.preprocessing.llie import LLIEProcessor
from src.rules.rule_engine import RuleEngine
from src.tracking.tracker import VehicleTracker
from src.utils.helpers import load_config
from src.utils.logger import get_logger, log_violation, setup_logger

logger = get_logger(__name__)


class RealtimePipeline:
    """
    Live camera processing pipeline.

    Usage:
        pipeline = RealtimePipeline(cfg)
        pipeline.run(camera_id="cam_01", rtsp_url="rtsp://...")
        # or
        pipeline.run(camera_id="cam_00", webcam_index=0)
    """

    def __init__(self, cfg: dict, stop_event: Optional[threading.Event] = None) -> None:
        self._cfg        = cfg
        self._stop_event = stop_event or threading.Event()

        sys_cfg  = cfg.get("system", {})
        self._evidence_dir = sys_cfg.get("evidence_dir", "evidence")
        self._secret_key   = sys_cfg.get("secret_key", "CHANGEME_32bytes_key!!")
        self._db_url       = sys_cfg.get("db_url", "sqlite:///traffic.db")
        self._target_fps   = cfg.get("camera", {}).get("target_fps", 15)

        # Initialise DB
        init_db(self._db_url)

        # Instantiate all modules
        self._dip      = DIPPreprocessor(cfg.get("preprocessing", {}))
        self._llie     = LLIEProcessor(
            cfg.get("llie", {}),
            dark_threshold=cfg.get("preprocessing", {}).get("dark_frame_threshold", 80),
        )
        self._lane     = LaneDetector(cfg.get("lane", {}))
        self._detector = VehicleDetector(cfg.get("detection", {}))
        self._tracker  = VehicleTracker(cfg.get("tracking", {}))
        self._features = FeatureExtractor(
            cfg.get("features", {}),
            homography=self._lane.calibrator,
            fps=self._target_fps,
        )
        self._behavior = BehaviorEngine(cfg.get("behavior", {}), detector=self._detector)
        self._rules    = RuleEngine(cfg.get("rules", {}))
        self._anpr     = ANPRSystem(cfg.get("anpr", {}))
        self._evidence = EvidenceGenerator(self._evidence_dir)

        logger.info("RealtimePipeline: all modules initialised.")

    # ── Camera Config Lookup ──────────────────────────────────────────

    def _camera_cfg(self, camera_id: str) -> dict:
        for cam in self._cfg.get("camera", {}).get("rtsp_streams", []):
            if cam.get("id") == camera_id:
                return cam
        return {}

    # ── Main Run Loop ─────────────────────────────────────────────────

    def run(
        self,
        camera_id:    str = "cam_00",
        rtsp_url:     Optional[str] = None,
        webcam_index: Optional[int] = None,
    ) -> None:
        """
        Start the real-time pipeline loop.

        Args:
            camera_id:    Identifier for this camera stream.
            rtsp_url:     RTSP URL (overrides config if provided).
            webcam_index: Webcam index (overrides RTSP if provided).
        """
        cam_cfg   = self._camera_cfg(camera_id)
        gps_lat   = cam_cfg.get("gps_lat", 0.0)
        gps_lon   = cam_cfg.get("gps_lon", 0.0)

        # Build source
        if webcam_index is not None:
            source = VideoSource.from_webcam(webcam_index, camera_id=camera_id)
        elif rtsp_url:
            source = VideoSource.from_rtsp(rtsp_url, camera_id=camera_id)
        else:
            url = cam_cfg.get("url", "")
            if not url:
                raise ValueError(f"No URL or webcam_index for camera '{camera_id}'.")
            source = VideoSource.from_rtsp(url, camera_id=camera_id)

        buffer = FrameBuffer(
            maxsize=self._cfg.get("camera", {}).get("buffer_size", 128),
            ring_size=int(self._target_fps * 4),
        )
        buffer.start_producer(source, target_fps=self._target_fps)

        logger.info(f"[{camera_id}] Pipeline running at target {self._target_fps} FPS.")
        frame_count   = 0
        latency_total = 0.0

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                # ── 1. Grab frame ─────────────────────────────────────
                try:
                    packet = buffer.get_frame(timeout=2.0)
                except Exception:
                    if not buffer.is_running:
                        logger.info(f"[{camera_id}] Source exhausted.")
                        break
                    continue

                frame     = packet.frame
                frame_idx = packet.frame_idx
                frame_ts  = packet.timestamp

                process_every_n = self._cfg.get("camera", {}).get("process_every_n_frames", 3)
                if frame_idx % process_every_n != 0:
                    frame_count += 1
                    continue

                # ── 2. DIP Preprocessing ──────────────────────────────
                frame = self._dip.preprocess(frame)

                # ── 3. LLIE (if dark) ─────────────────────────────────
                frame = self._llie.enhance_if_dark(frame)

                # ── 4. Lane Detection ─────────────────────────────────
                lane_result = self._lane.detect(frame)
                frame       = self._lane.draw_lanes(frame, lane_result)

                # ── 5. Vehicle Detection ──────────────────────────────
                all_dets  = self._detector.detect(frame)
                veh_dets  = [d for d in all_dets if d.class_id != self._cfg.get("detection", {}).get("person_class_id", 0)]
                persons   = [d for d in all_dets if d.class_id == self._cfg.get("detection", {}).get("person_class_id", 0)]

                # ── 6. Tracking ───────────────────────────────────────
                tracks = self._tracker.update(
                    veh_dets, frame,
                    frame_ts=frame_ts,
                    homography=self._lane.calibrator,
                )

                # ── 7. Feature Extraction ─────────────────────────────
                feat_map = self._features.update(
                    tracks, frame,
                    lane_result=lane_result,
                    world_history=self._tracker.world_history,
                    ts_history=self._tracker.ts_history,
                )
                self._features.compute_following_distances(
                    tracks, feat_map, self._tracker.world_history
                )

                # ── 8. Behavior Detection ─────────────────────────────
                violations = self._behavior.run_all(
                    tracks, feat_map,
                    lane_result=lane_result,
                    frame_idx=frame_idx,
                    frame=frame,
                    persons=persons,
                )

                # ── 9. Per-violation: ANPR + Evidence + DB ────────────
                for violation in violations:
                    track = next((t for t in tracks if t.id == violation.track_id), None)
                    if track is None:
                        continue
                    f = feat_map.get(violation.track_id)

                    # ANPR
                    plate = self._anpr.recognize(frame, vehicle_bbox=track.bbox)
                    violation.plate_text = plate.text
                    violation.plate_conf = plate.confidence

                    # Rule engine
                    vid   = plate.text if plate.text else f"track_{violation.track_id}"
                    speed = violation.metadata.get("speed_kmh", 0.0)
                    limit = self._cfg.get("behavior", {}).get("overspeed", {}).get("speed_limit_kmh", 60.0)
                    rule_result = self._rules.apply_violation(
                        vid, violation.type,
                        speed_kmh=speed, limit_kmh=limit,
                        overspeed_cfg=self._cfg.get("behavior", {}).get("overspeed"),
                    )

                    # Build lane_lines tuple for evidence overlay
                    ll = (
                        lane_result.left.as_tuple()  if lane_result.left  else None,
                        lane_result.right.as_tuple() if lane_result.right else None,
                    )

                    # Evidence image + clip
                    img_path, clip_path = self._evidence.capture_all(
                        frame, track, violation, plate,
                        ring_buffer=buffer.ring_buffer,
                        camera_id=camera_id,
                        lane_lines=ll,
                        mv_act=rule_result.mv_act,
                        fps=self._target_fps,
                    )

                    # OCR status for DB
                    ocr_status = "pending"
                    if plate.status == "low_confidence":
                        ocr_status = "low_confidence"

                    # Persist to DB
                    save_violation(
                        plate_text=plate.text or "",
                        vehicle_class=track.class_name,
                        violation_type=violation.type,
                        speed_kmh=float(speed),
                        fine_inr=rule_result.fine_inr,
                        evidence_image=img_path,
                        evidence_clip=clip_path,
                        ocr_confidence=plate.confidence,
                        mv_act=rule_result.mv_act,
                        camera_id=camera_id,
                        gps_lat=gps_lat,
                        gps_lon=gps_lon,
                        metadata_dict=violation.metadata,
                        secret_key=self._secret_key,
                        ocr_status=ocr_status,
                    )

                    # Violation log
                    log_violation(
                        track_id=violation.track_id,
                        violation_type=violation.type,
                        plate=plate.text or "UNKNOWN",
                        speed=float(speed),
                        fine=rule_result.fine_inr,
                        camera_id=camera_id,
                    )

                # ── 10. Latency tracking ──────────────────────────────
                frame_count   += 1
                elapsed_ms     = (time.monotonic() - t0) * 1000
                latency_total += elapsed_ms
                if frame_count % 100 == 0:
                    avg_ms  = latency_total / frame_count
                    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                    logger.info(
                        f"[{camera_id}] Frame #{frame_count} | "
                        f"avg latency={avg_ms:.1f}ms | avg FPS={avg_fps:.1f}"
                    )
                    if avg_ms > 100:
                        logger.warning(
                            f"[{camera_id}] Latency {avg_ms:.1f}ms exceeds 100ms target!"
                        )

        finally:
            buffer.stop()
            source.release()
            logger.info(
                f"[{camera_id}] Pipeline stopped after {frame_count} frames."
            )
