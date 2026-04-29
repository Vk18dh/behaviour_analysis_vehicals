"""
src/pipeline/batch_pipeline.py
Batch pipeline for uploaded video files (MP4 / AVI).
Same module chain as the real-time pipeline but reads from a file,
processes every frame sequentially, and commits all DB records at end.
Shows a tqdm progress bar.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from src.anpr.anpr import ANPRSystem
from src.behavior.behavior_engine import BehaviorEngine
from src.database.db import init_db, save_violation
from src.detection.vehicle_detector import VehicleDetector
from src.evidence.evidence_generator import EvidenceGenerator
from src.features.feature_extractor import FeatureExtractor
from src.input.video_input import VideoSource
from src.lane.lane_detection import LaneDetector
from src.preprocessing.dip import DIPPreprocessor
from src.preprocessing.llie import LLIEProcessor
from src.rules.rule_engine import RuleEngine
from src.tracking.tracker import VehicleTracker
from src.utils.logger import get_logger, log_violation, setup_logger

logger = get_logger(__name__)


class BatchPipeline:
    """
    Processes a video file from start to finish.

    Usage:
        pipeline = BatchPipeline(cfg)
        pipeline.run("path/to/video.mp4")
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

        sys_cfg            = cfg.get("system", {})
        self._evidence_dir = sys_cfg.get("evidence_dir", "evidence")
        self._secret_key   = sys_cfg.get("secret_key", "CHANGEME_32bytes_key!!")
        self._db_url       = sys_cfg.get("db_url", "sqlite:///traffic.db")

        init_db(self._db_url)

        # All modules share the same config as the realtime pipeline
        self._dip      = DIPPreprocessor(cfg.get("preprocessing", {}))
        self._llie     = LLIEProcessor(
            cfg.get("llie", {}),
            dark_threshold=cfg.get("preprocessing", {}).get("dark_frame_threshold", 80),
        )
        self._lane     = LaneDetector(cfg.get("lane", {}))
        self._detector = VehicleDetector(cfg.get("detection", {}))
        self._tracker  = VehicleTracker(cfg.get("tracking", {}))

        src_fps = 25.0   # will be updated once file is opened
        self._features = FeatureExtractor(
            cfg.get("features", {}),
            homography=self._lane.calibrator,
            fps=src_fps,
        )
        self._behavior = BehaviorEngine(cfg.get("behavior", {}), detector=self._detector)
        self._rules    = RuleEngine(cfg.get("rules", {}))
        self._anpr     = ANPRSystem(cfg.get("anpr", {}))
        self._evidence = EvidenceGenerator(self._evidence_dir)

        logger.info("BatchPipeline: all modules initialised.")

    def run(self, video_path: str, camera_id: str = "batch") -> dict:
        """
        Process a video file end-to-end.

        Args:
            video_path: Absolute or relative path to MP4/AVI file.
            camera_id:  Label used in DB records.

        Returns:
            Summary dict: {total_frames, total_violations, elapsed_sec}.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        logger.info(f"BatchPipeline: processing '{path.name}'…")
        t_start = time.time()

        with VideoSource.from_file(str(path), camera_id=camera_id) as source:
            total_frames = source.total_frames
            fps          = source.source_fps
            self._features._fps = fps
            self._features._dt  = 1.0 / max(fps, 1.0)

            pbar = tqdm(
                total=total_frames if total_frames > 0 else None,
                desc=path.name,
                unit="frame",
            )

            frame_count = 0
            viol_count  = 0
            emitted_violations = set()

            for packet in source.stream():
                frame     = packet.frame
                frame_idx = packet.frame_idx
                frame_ts  = packet.timestamp

                # Batch uses a separate (higher) frame skip for speed
                process_every_n = self._cfg.get("camera", {}).get("batch_process_every_n_frames", 5)
                if frame_idx % process_every_n != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue

                # ── Resize for speed (batch only) ─────────────────────
                # Scale based on the longest edge to handle portrait gracefully
                resize_max = self._cfg.get("camera", {}).get("batch_resize_width", 960)
                h, w = frame.shape[:2]
                if resize_max and resize_max > 0 and max(h, w) > resize_max:
                    scale = resize_max / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


                # ── DIP + LLIE ────────────────────────────────────────
                frame, _dip_report = self._dip.preprocess(frame, frame_idx=frame_idx)
                frame = self._llie.enhance_if_dark(frame)

                # ── Lane ──────────────────────────────────────────────
                lane_result = self._lane.detect(frame)

                # ── Detect ───────────────────────────────────────────
                all_dets = self._detector.detect(frame)
                person_id = self._cfg.get("detection", {}).get("person_class_id", 0)
                veh_dets  = [d for d in all_dets if d.class_id != person_id]
                persons   = [d for d in all_dets if d.class_id == person_id]

                # ── Track ─────────────────────────────────────────────
                tracks = self._tracker.update(
                    veh_dets, frame,
                    frame_ts=frame_ts,
                    homography=self._lane.calibrator,
                )

                # ── Features ──────────────────────────────────────────
                feat_map = self._features.update(
                    tracks, frame,
                    lane_result=lane_result,
                    world_history=self._tracker.world_history,
                    ts_history=self._tracker.ts_history,
                )
                self._features.compute_following_distances(
                    tracks, feat_map, self._tracker.world_history
                )

                # ── Behavior ──────────────────────────────────────────
                violations = self._behavior.run_all(
                    tracks, feat_map,
                    lane_result=lane_result,
                    frame_idx=frame_idx,
                    frame=frame,
                    persons=persons,
                    frame_ts=frame_ts,
                )

                # ── Per-violation processing ──────────────────────────
                for violation in violations:
                    track = next(
                        (t for t in tracks if t.id == violation.track_id), None
                    )
                    if track is None:
                        continue

                    # Deduplication: 1 violation type per track per video
                    dedup_key = (track.id, violation.type)
                    if dedup_key in emitted_violations:
                        continue
                    emitted_violations.add(dedup_key)

                    # ANPR
                    plate = self._anpr.recognize(frame, vehicle_bbox=track.bbox)
                    violation.plate_text = plate.text
                    violation.plate_conf = plate.confidence

                    # Rule engine
                    vid   = plate.text if plate.text else f"track_{violation.track_id}"
                    speed = violation.metadata.get("speed_kmh", 0.0)
                    limit = (self._cfg.get("behavior", {})
                             .get("overspeed", {}).get("speed_limit_kmh", 60.0))
                    rule_result = self._rules.apply_violation(
                        vid, violation.type,
                        speed_kmh=speed, limit_kmh=limit,
                        overspeed_cfg=self._cfg.get("behavior", {}).get("overspeed"),
                    )

                    # Evidence image (no ring buffer in batch — just image)
                    ll = (
                        lane_result.left.as_tuple()  if lane_result.left  else None,
                        lane_result.right.as_tuple() if lane_result.right else None,
                    )
                    img_path, _ = self._evidence.capture_all(
                        frame, track, violation, plate,
                        ring_buffer=None,
                        camera_id=camera_id,
                        lane_lines=ll,
                        mv_act=rule_result.mv_act,
                        fps=fps,
                    )

                    ocr_status = "low_confidence" if plate.status == "low_confidence" else "pending"

                    save_violation(
                        plate_text=plate.text or "",
                        vehicle_class=track.class_name,
                        violation_type=violation.type,
                        speed_kmh=float(speed),
                        fine_inr=rule_result.fine_inr,
                        evidence_image=img_path,
                        evidence_clip="",
                        ocr_confidence=plate.confidence,
                        mv_act=rule_result.mv_act,
                        camera_id=camera_id,
                        gps_lat=0.0,
                        gps_lon=0.0,
                        metadata_dict=violation.metadata,
                        secret_key=self._secret_key,
                        ocr_status=ocr_status,
                        dedup_window_sec=self._cfg.get("system", {}).get("dedup_window_sec", 120),
                    )

                    log_violation(
                        track_id=violation.track_id,
                        violation_type=violation.type,
                        plate=plate.text or "UNKNOWN",
                        speed=float(speed),
                        fine=rule_result.fine_inr,
                        camera_id=camera_id,
                    )
                    viol_count += 1

                frame_count += 1
                pbar.update(1)

            pbar.close()

        elapsed = time.time() - t_start
        summary = {
            "total_frames":     frame_count,
            "total_violations": viol_count,
            "elapsed_sec":      round(elapsed, 2),
            "avg_fps":          round(frame_count / max(elapsed, 0.001), 1),
        }
        logger.info(
            f"BatchPipeline complete: {frame_count} frames, "
            f"{viol_count} violations in {elapsed:.1f}s "
            f"({summary['avg_fps']} FPS avg)"
        )
        return summary
