"""
src/rules/rule_engine.py
Credit score system + fine calculation aligned with Indian Motor Vehicles Act.
- Starts at 100 per vehicle
- Deducts points per violation (configurable)
- Dynamic fines: OVERSPEED = base + 100×floor(excess/5)
- Categories: Safe (80-100), Moderate (50-79), Risky (0-49)
- All values loaded from settings.yaml — zero hardcoding
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Result Dataclass
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RuleResult:
    """Output of RuleEngine.apply_violation()."""
    vehicle_id:   str
    violation_type: str
    new_score:    int
    category:     str       # "Safe" | "Moderate" | "Risky"
    fine_inr:     float
    deduction:    int
    mv_act:       str       # MV Act section reference


# ══════════════════════════════════════════════════════════════════════
# Rule Engine
# ══════════════════════════════════════════════════════════════════════

class RuleEngine:
    """
    Applies violation rules to per-vehicle credit scores.

    Maintains an in-memory score cache; DB persistence is handled
    externally by the pipeline (via db.py).

    Usage:
        engine = RuleEngine(cfg=config["rules"])
        result = engine.apply_violation("KA01AB1234", "OVERSPEED",
                                        speed_kmh=85.0, limit_kmh=60.0)
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 'rules' section from settings.yaml.
        """
        self._initial      = cfg.get("initial_score", 100)
        self._deductions   = cfg.get("deductions", {})
        self._fines        = cfg.get("fines_inr", {})
        self._mv_act       = cfg.get("mv_act_sections", {})
        self._categories   = cfg.get("categories", {
            "safe":     [80, 100],
            "moderate": [50, 79],
            "risky":    [0, 49],
        })
        self._overspeed_cfg = {}   # filled from outer config if passed

        # In-memory score cache: {vehicle_id: score}
        self._scores: Dict[str, int] = {}

        logger.info(
            f"RuleEngine init — initial_score={self._initial}, "
            f"violations configured: {list(self._deductions.keys())}"
        )

    # ── Score Management ─────────────────────────────────────────────

    def get_score(self, vehicle_id: str) -> int:
        """Return current credit score for a vehicle (initialise if new)."""
        if vehicle_id not in self._scores:
            self._scores[vehicle_id] = self._initial
        return self._scores[vehicle_id]

    def set_score(self, vehicle_id: str, score: int) -> None:
        """Restore score from DB on startup."""
        self._scores[vehicle_id] = max(0, min(self._initial, score))

    # ── Category ────────────────────────────────────────────────────

    def get_category(self, score: int) -> str:
        """
        Map numeric score to human-readable risk category.

        Returns:
            "Safe" | "Moderate" | "Risky"
        """
        for label, (lo, hi) in self._categories.items():
            if lo <= score <= hi:
                return label.capitalize()
        return "Risky"

    # ── Fine Calculation ─────────────────────────────────────────────

    def calculate_fine(
        self,
        violation_type: str,
        speed_kmh: float = 0.0,
        limit_kmh: float = 60.0,
        overspeed_cfg: Optional[dict] = None,
    ) -> float:
        """
        Calculate the fine in INR for a violation.

        For OVERSPEED: fine = base_fine + per_5kmh × floor(excess / 5)
        All others: flat fine from config.

        Args:
            violation_type: e.g. "OVERSPEED", "ZIGZAG".
            speed_kmh:      Recorded vehicle speed (used for OVERSPEED).
            limit_kmh:      Speed limit at location.
            overspeed_cfg:  Optional overspeed sub-config for base/per values.

        Returns:
            Fine in INR as float.
        """
        base_fine = float(self._fines.get(violation_type, 500))

        if violation_type == "OVERSPEED" and speed_kmh > limit_kmh:
            cfg    = overspeed_cfg or {}
            per5   = float(cfg.get("per_5kmh", 100))
            excess = speed_kmh - limit_kmh
            extra  = per5 * math.floor(excess / 5.0)
            return base_fine + extra

        return base_fine

    def calculate_deduction(
        self,
        violation_type: str,
        speed_kmh: float = 0.0,
        limit_kmh: float = 60.0,
    ) -> int:
        """
        Calculate point deduction.
        For OVERSPEED: 1 point per 5 km/h over limit.
        All others: flat deduction from config.

        Returns:
            Integer point deduction (always non-negative).
        """
        base = int(self._deductions.get(violation_type, 3))

        if violation_type == "OVERSPEED" and speed_kmh > limit_kmh:
            excess = speed_kmh - limit_kmh
            return max(base, int(math.floor(excess / 5.0)))

        return base

    # ── Main Apply ───────────────────────────────────────────────────

    def apply_violation(
        self,
        vehicle_id: str,
        violation_type: str,
        speed_kmh: float = 0.0,
        limit_kmh: float = 60.0,
        overspeed_cfg: Optional[dict] = None,
    ) -> RuleResult:
        """
        Apply a violation to a vehicle's credit score.

        Args:
            vehicle_id:     Plate text or track-based ID.
            violation_type: e.g. "ZIGZAG", "OVERSPEED".
            speed_kmh:      Vehicle speed at time of violation.
            limit_kmh:      Posted speed limit.
            overspeed_cfg:  Overspeed fine config dict.

        Returns:
            RuleResult with updated score, category, and fine.
        """
        current_score = self.get_score(vehicle_id)
        deduction     = self.calculate_deduction(violation_type, speed_kmh, limit_kmh)
        fine          = self.calculate_fine(violation_type, speed_kmh, limit_kmh, overspeed_cfg)
        mv_act_ref    = self._mv_act.get(violation_type, "Motor Vehicles Act")

        new_score = max(0, current_score - deduction)
        self._scores[vehicle_id] = new_score
        category  = self.get_category(new_score)

        logger.info(
            f"[Rule] {vehicle_id} | {violation_type} | "
            f"score {current_score}→{new_score} ({category}) | "
            f"fine=INR{fine:.0f} | {mv_act_ref}"
        )

        return RuleResult(
            vehicle_id=vehicle_id,
            violation_type=violation_type,
            new_score=new_score,
            category=category,
            fine_inr=fine,
            deduction=deduction,
            mv_act=mv_act_ref,
        )

    def summary(self, vehicle_id: str) -> Dict:
        """Return a summary dict for API responses."""
        score    = self.get_score(vehicle_id)
        category = self.get_category(score)
        return {
            "vehicle_id": vehicle_id,
            "score":      score,
            "category":   category,
            "max_score":  self._initial,
        }
