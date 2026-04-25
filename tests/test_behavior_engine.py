"""
tests/test_behavior_engine.py
Unit tests for src/behavior/behavior_engine.py

Uses minimal fake tracks + features to test each detector in isolation.
No CV model is loaded — sub-classifiers are mocked.
"""

import time
import pytest
from collections import deque
from unittest.mock import MagicMock

from src.behavior.behavior_engine import BehaviorEngine, Violation
from src.features.feature_extractor import VehicleFeatures
from src.tracking.tracker import Track


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

def _make_track(tid=1, class_id=1, class_name="motorcycle",
                cx=640, cy=360) -> Track:
    return Track(
        id=tid, bbox=(cx-30, cy-50, cx+30, cy+50),
        class_id=class_id, class_name=class_name,
        centroid=(cx, cy), confidence=0.9, age=10, hits=10,
    )


def _make_features(tid=1, **overrides) -> VehicleFeatures:
    f = VehicleFeatures(track_id=tid)
    for k, v in overrides.items():
        setattr(f, k, v)
    return f


def _engine_with_cfg(extra_cfg=None):
    base_cfg = {
        "zigzag": {
            "time_window_sec": 5.0, "min_direction_changes": 2,
            "lateral_displacement_m": 0.5,   # very low for test
            "noise_filter_m": 0.01,
            "lateral_accel_threshold": 0.1,  # very low for test
            "sustained_sec": 0.0,            # instant flag in tests
        },
        "tailgating": {
            "time_headway_sec": 1.5,
            "sustained_duration_sec": 0.0,
        },
        "overspeed": {
            "speed_limit_kmh": 60.0,
            "sustained_duration_sec": 0.0,
        },
        "red_light": {
            "signal_state": "RED",
            "roi_polygon": [[0,0],[1280,0],[1280,720],[0,720]],
            "min_speed_kmh": 1.0,
        },
        "wrong_direction": {
            "dot_product_threshold": -0.5,
            "sustained_sec": 0.0,
        },
        "rash_driving": {
            "accel_threshold":      1.0,
            "direction_change_rate": 0.5,
        },
        "highway_restriction": {
            "zone_polygon": [[0,0],[1280,0],[1280,720],[0,720]],
        },
        "lane_violation": {
            "fast_lane_x_fraction": 0.6,
            "sustained_sec": 0.0,
        },
        "helmet": {"confidence_threshold": 0.90, "sustained_sec": 0.0},
        "seatbelt": {"confidence_threshold": 0.85},
        "triple_riding": {"min_persons": 3},
        "illegal_turn": {
            "junction_roi": [[0,0],[1280,0],[1280,720],[0,720]],
            "angle_threshold_deg": 90.0,
        },
        "phone_use": {"enabled": False, "confidence_threshold": 0.80},
        "features": {"min_speed_for_tailgate": 5.0},
    }
    if extra_cfg:
        base_cfg.update(extra_cfg)
    mock_detector = MagicMock()
    mock_detector.is_two_wheeler.return_value  = True
    mock_detector.is_three_wheeler.return_value = False
    mock_detector.is_heavy.return_value        = False
    mock_detector.persons_inside_bbox.return_value = []
    mock_detector.classify_helmet.return_value = (False, 0.92)
    mock_detector.classify_seatbelt.return_value = (True, 0.95)
    mock_detector.classify_phone.return_value  = (False, 0.3)
    eng = BehaviorEngine(base_cfg, detector=mock_detector)
    return eng


# ══════════════════════════════════════════════════════════════════════
# Overspeed
# ══════════════════════════════════════════════════════════════════════

class TestOverspeed:
    def test_flags_above_limit(self):
        eng   = _engine_with_cfg()
        track = _make_track()
        f     = _make_features(speed_kmh=90.0)
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        types = [v.type for v in viols]
        assert "OVERSPEED" in types

    def test_no_flag_under_limit(self):
        eng   = _engine_with_cfg()
        track = _make_track()
        f     = _make_features(speed_kmh=55.0)
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert not any(v.type == "OVERSPEED" for v in viols)


# ══════════════════════════════════════════════════════════════════════
# Red Light
# ══════════════════════════════════════════════════════════════════════

class TestRedLight:
    def test_flags_inside_roi(self):
        eng              = _engine_with_cfg()
        eng.signal_state = "RED"
        track            = _make_track(cx=640, cy=360)   # inside full-frame ROI
        f                = _make_features(speed_kmh=30.0)
        viols            = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert any(v.type == "RED_LIGHT" for v in viols)

    def test_no_flag_green_signal(self):
        eng              = _engine_with_cfg()
        eng.signal_state = "GREEN"
        track            = _make_track()
        f                = _make_features(speed_kmh=30.0)
        viols            = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert not any(v.type == "RED_LIGHT" for v in viols)


# ══════════════════════════════════════════════════════════════════════
# Tailgating
# ══════════════════════════════════════════════════════════════════════

class TestTailgating:
    def test_flags_short_headway(self):
        eng   = _engine_with_cfg()
        track = _make_track()
        f     = _make_features(speed_kmh=80.0, time_headway_s=0.8,
                               following_dist_m=18.0)
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert any(v.type == "TAILGATING" for v in viols)

    def test_no_flag_safe_headway(self):
        eng   = _engine_with_cfg()
        track = _make_track()
        f     = _make_features(speed_kmh=80.0, time_headway_s=2.5,
                               following_dist_m=55.5)
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert not any(v.type == "TAILGATING" for v in viols)


# ══════════════════════════════════════════════════════════════════════
# Wrong Direction
# ══════════════════════════════════════════════════════════════════════

class TestWrongDirection:
    def test_flags_reverse_direction(self):
        eng   = _engine_with_cfg()
        track = _make_track()
        # Moving downward in world Y (road goes up Y): dot((0,1),(0,-1)) = -1
        f     = _make_features(speed_kmh=30.0, direction_vec=(0.0, 1.0))
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert any(v.type == "WRONG_DIRECTION" for v in viols)

    def test_no_flag_correct_direction(self):
        eng   = _engine_with_cfg()
        track = _make_track()
        f     = _make_features(speed_kmh=30.0, direction_vec=(0.0, -1.0))
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert not any(v.type == "WRONG_DIRECTION" for v in viols)


# ══════════════════════════════════════════════════════════════════════
# Zigzag
# ══════════════════════════════════════════════════════════════════════

class TestZigzag:
    def test_flags_with_lateral_motion(self):
        eng   = _engine_with_cfg()
        track = _make_track()
        f     = _make_features(lateral_accel_mps2=5.0)
        # Inject alternating lateral deltas directly into engine's window
        now = time.time()
        win = eng._zz_window[track.id]
        for dx in [1.0, -1.0, 1.5, -1.5, 1.0, -1.0]:
            win.append((now, dx))
            now += 0.1
        f.lateral_dx_history = deque(
            [1.0, -1.0, 1.5, -1.5, 1.0, -1.0], maxlen=150
        )
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert any(v.type == "ZIGZAG" for v in viols)


# ══════════════════════════════════════════════════════════════════════
# Lane Violation
# ══════════════════════════════════════════════════════════════════════

class TestLaneViolation:
    def test_flags_heavy_in_fast_lane(self):
        eng = _engine_with_cfg()
        eng._detector.is_heavy.return_value = True
        track = _make_track(class_id=5, class_name="truck")
        f     = _make_features(lane_index=1, speed_kmh=70)
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert any(v.type == "LANE_VIOLATION" for v in viols)

    def test_no_flag_heavy_in_slow_lane(self):
        eng = _engine_with_cfg()
        eng._detector.is_heavy.return_value = True
        track = _make_track(class_id=5, class_name="truck")
        f     = _make_features(lane_index=0, speed_kmh=40)
        viols = eng.run_all([track], {track.id: f}, frame_idx=1)
        assert not any(v.type == "LANE_VIOLATION" for v in viols)
