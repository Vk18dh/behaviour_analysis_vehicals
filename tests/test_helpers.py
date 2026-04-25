"""
tests/test_helpers.py
Unit tests for src/utils/helpers.py geometric utilities.
"""
import pytest
from collections import deque
from src.utils.helpers import (
    euclidean_distance, bbox_center, bbox_iou, bbox_area,
    count_sign_changes, compute_lateral_accel, smooth_values,
    poly_contains_point,
)


class TestGeometry:
    def test_euclidean_distance_zero(self):
        assert euclidean_distance((0, 0), (0, 0)) == 0.0

    def test_euclidean_distance_basic(self):
        result = euclidean_distance((0, 0), (3, 4))
        assert abs(result - 5.0) < 1e-6

    def test_bbox_center(self):
        assert bbox_center((0, 0, 100, 100)) == (50, 50)
        assert bbox_center((10, 20, 30, 40)) == (20, 30)

    def test_bbox_area(self):
        assert bbox_area((0, 0, 10, 10)) == 100
        assert bbox_area((5, 5, 5, 5)) == 0    # zero area

    def test_bbox_iou_identical(self):
        b = (0, 0, 100, 100)
        assert abs(bbox_iou(b, b) - 1.0) < 1e-6

    def test_bbox_iou_no_overlap(self):
        b1 = (0, 0, 50, 50)
        b2 = (100, 100, 200, 200)
        assert bbox_iou(b1, b2) == 0.0

    def test_bbox_iou_partial(self):
        b1 = (0, 0, 100, 100)
        b2 = (50, 50, 150, 150)
        iou = bbox_iou(b1, b2)
        assert 0 < iou < 1

    def test_poly_contains_inside(self):
        poly = [[0,0],[100,0],[100,100],[0,100]]
        assert poly_contains_point(poly, (50, 50))

    def test_poly_contains_outside(self):
        poly = [[0,0],[100,0],[100,100],[0,100]]
        assert not poly_contains_point(poly, (150, 50))


class TestSignalUtils:
    def test_count_sign_changes_none(self):
        assert count_sign_changes([1, 2, 3, 4]) == 0

    def test_count_sign_changes_basic(self):
        assert count_sign_changes([1, -1, 1, -1]) == 3

    def test_count_sign_changes_with_zeros(self):
        # zeros should be ignored
        assert count_sign_changes([1, 0, -1]) == 1

    def test_smooth_values_empty(self):
        assert smooth_values(deque(), 5) == 0.0

    def test_smooth_values_basic(self):
        d = deque([10, 20, 30])
        assert smooth_values(d, 3) == 20.0

    def test_smooth_values_window_smaller(self):
        d = deque([10, 20, 30, 40])
        assert smooth_values(d, 2) == 35.0    # last 2: (30+40)/2

    def test_lateral_accel_insufficient(self):
        d = deque([1.0])
        assert compute_lateral_accel(d, 0.1) == 0.0

    def test_lateral_accel_basic(self):
        d = deque([0.0, 2.0])     # dx1=0, dx2=2, dt=0.1
        result = compute_lateral_accel(d, 0.1)
        # (2-0) / 0.01 = 200 m/s²
        assert abs(result - 200.0) < 1e-3
