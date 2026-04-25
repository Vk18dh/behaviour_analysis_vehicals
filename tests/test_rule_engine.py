"""
tests/test_rule_engine.py
Unit tests for src/rules/rule_engine.py
"""
import pytest
from src.rules.rule_engine import RuleEngine

_CFG = {
    "initial_score": 100,
    "deductions": {
        "ZIGZAG": 10,
        "TAILGATING": 8,
        "OVERSPEED": 5,
        "RED_LIGHT": 15,
        "NO_HELMET": 6,
    },
    "fines_inr": {
        "ZIGZAG": 1000,
        "TAILGATING": 1000,
        "OVERSPEED": 400,
        "RED_LIGHT": 1000,
        "NO_HELMET": 1000,
    },
    "mv_act_sections": {
        "OVERSPEED": "MV Act §183",
        "ZIGZAG": "MV Act §184",
    },
    "categories": {
        "safe":     [80, 100],
        "moderate": [50, 79],
        "risky":    [0,  49],
    },
}

_OVERSPEED_CFG = {"per_5kmh": 100, "speed_limit_kmh": 60}


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine(_CFG)

    # ── Score management ──────────────────────────────────────────────

    def test_initial_score(self):
        assert self.engine.get_score("KA01AB1234") == 100

    def test_score_deduction_zigzag(self):
        self.engine.apply_violation("KA01AB1234", "ZIGZAG")
        assert self.engine.get_score("KA01AB1234") == 90

    def test_score_never_goes_below_zero(self):
        for _ in range(20):
            self.engine.apply_violation("MH01CD5678", "RED_LIGHT")
        assert self.engine.get_score("MH01CD5678") == 0

    def test_score_separate_vehicles(self):
        self.engine.apply_violation("V1", "ZIGZAG")
        self.engine.apply_violation("V2", "NO_HELMET")
        assert self.engine.get_score("V1") == 90
        assert self.engine.get_score("V2") == 94

    # ── Category ─────────────────────────────────────────────────────

    def test_category_safe(self):
        assert self.engine.get_category(100) == "Safe"
        assert self.engine.get_category(80)  == "Safe"

    def test_category_moderate(self):
        assert self.engine.get_category(79) == "Moderate"
        assert self.engine.get_category(50) == "Moderate"

    def test_category_risky(self):
        assert self.engine.get_category(49) == "Risky"
        assert self.engine.get_category(0)  == "Risky"

    # ── Fine calculation ──────────────────────────────────────────────

    def test_flat_fine_zigzag(self):
        r = self.engine.apply_violation("P1", "ZIGZAG")
        assert r.fine_inr == 1000.0

    def test_overspeed_fine_base(self):
        # speed=65, limit=60 → excess=5 → floor(5/5)=1 → 400 + 100*1 = 500
        r = self.engine.apply_violation("P2", "OVERSPEED",
                                        speed_kmh=65, limit_kmh=60,
                                        overspeed_cfg=_OVERSPEED_CFG)
        assert r.fine_inr == 500.0

    def test_overspeed_fine_larger(self):
        # speed=85, limit=60 → excess=25 → floor(25/5)=5 → 400+500=900
        r = self.engine.apply_violation("P3", "OVERSPEED",
                                        speed_kmh=85, limit_kmh=60,
                                        overspeed_cfg=_OVERSPEED_CFG)
        assert r.fine_inr == 900.0

    def test_overspeed_deduction(self):
        # excess=25 → 5 units per 5kmh → deduction=max(5,5)=5
        r = self.engine.apply_violation("P4", "OVERSPEED",
                                        speed_kmh=85, limit_kmh=60)
        assert r.deduction >= 5

    # ── MV Act section ────────────────────────────────────────────────

    def test_mv_act_section_mapped(self):
        r = self.engine.apply_violation("P5", "OVERSPEED")
        assert "§183" in r.mv_act

    def test_mv_act_section_default(self):
        r = self.engine.apply_violation("P6", "TRIPLE_RIDING")
        assert r.mv_act  # non-empty default
