"""Bound where the arm can be while Operator's telemetry is paused for motion.

The ER-V cannot report joint counts in ACL manual mode, so during a continuous
follow the newest reading predates the move. This bounds each joint's count from
the moves Pi-Vision itself commanded: every move is assumed to run at an
upper-bound rate for the current SPEED and then coast. Overestimating only stops
a follow early near a limit; it never lets one run past it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

AXES = ("base", "shoulder", "elbow", "pitch")


@dataclass(frozen=True)
class TravelModel:
    # Upper-bound counts/s per axis at SPEED 100. Speed is not linear in SPEED
    # (SPEED 20 runs ~25% faster than a linear scale-down), so these are fitted
    # to the worst SPEED seen and SPEED below min_speed_percent is not scaled.
    rate_at_100: Dict[str, float]
    # Sign of the count change for a +1 command on each axis (azimuth, joint jog
    # direction, or altitude). 0 means unknown: bound both directions.
    count_sign: Dict[str, int]
    # Travel after Pi-Vision stops a move, as seconds at the move's rate.
    coast_s: float = 0.2
    # How far ahead a check looks, so the next check is not already too late.
    lookahead_s: float = 0.05
    # A reading is timestamped on receipt; LISTPV may have sampled the joints up
    # to this long before (its reply takes ~390 ms).
    sample_latency_s: float = 0.45
    # Servo settle after a stop drifts a few counts either way (seen up to 11).
    settle_counts: float = 25.0
    # Slowest SPEED the rates were fitted at; slower is bounded as this fast.
    min_speed_percent: int = 20

    def rate(self, axis: str, speed_percent: int) -> float:
        return self.rate_at_100[axis] * max(self.min_speed_percent, min(100, speed_percent)) / 100.0


@dataclass
class _Move:
    axis: str
    direction: int
    start: float
    speed_percent: int
    end: Optional[float] = None


@dataclass
class TravelEstimator:
    model: TravelModel
    telemetry_timeout_s: float = 1.25
    _moves: List[_Move] = field(default_factory=list)

    def start(self, axis: str, direction: int, now: float, speed_percent: int) -> None:
        current = self._moves[-1] if self._moves and self._moves[-1].end is None else None
        if current and (current.axis, current.direction) == (axis, direction):
            return
        self.stop(now)
        self._moves.append(_Move(axis, direction, now, speed_percent))

    def stop(self, now: float) -> None:
        if self._moves and self._moves[-1].end is None:
            self._moves[-1].end = now

    def unanchored_since(self, telemetry, now: float) -> Optional[float]:
        """When untracked motion began, if the newest reading cannot anchor a bound.

        That is the case when the reading was already stale when the first
        retained move started: something (homing, a Commander MOVE) may have
        moved the arm since. None when the bound is anchored, or when there is no
        reading at all (an arm without telemetry).
        """
        if not telemetry.has_received:
            return None
        retained = self._retained(telemetry.last_updated_monotonic)
        anchor = retained[0].start if retained else now
        return anchor if anchor - telemetry.last_updated_monotonic > self.telemetry_timeout_s else None

    def _retained(self, read_at: float) -> List[_Move]:
        # Drop moves the reading fully reflects, coast included.
        sampled_after = read_at - self.model.sample_latency_s
        self._moves = [m for m in self._moves
                       if m.end is None or m.end + self.model.coast_s > sampled_after]
        return self._moves

    def count_range(
        self, axis: str, telemetry, now: float, speed_percent: int,
        planned: Optional[int] = None,
    ) -> Optional[Tuple[float, float]]:
        """Lowest and highest count `axis` could reach.

        ``planned`` is a direction about to be commanded (or continued) on this
        axis; if it is not already moving that way, a new move is assumed to
        start now. Returns None when no anchored reading exists, e.g. on an arm
        that reports no telemetry.
        """
        if not telemetry.has_received or self.unanchored_since(telemetry, now) is not None:
            return None
        sampled_after = telemetry.last_updated_monotonic - self.model.sample_latency_s

        count = float(getattr(telemetry, "wrist_pitch" if axis == "pitch" else axis))
        low, high = count - self.model.settle_counts, count + self.model.settle_counts
        moves = [m for m in self._moves if m.axis == axis]
        if planned and not any(m.end is None and m.direction == planned for m in moves):
            moves.append(_Move(axis, planned, now, speed_percent))
        sign = self.model.count_sign.get(axis, 0)
        for move in moves:
            if move.end is None:
                # Keep the highest speed an open move has run at.
                move.speed_percent = max(move.speed_percent, speed_percent)
            rate = self.model.rate(axis, move.speed_percent)
            if move.end is not None and move.end <= sampled_after:
                # Commanded travel is in the reading; only coast may be missing.
                travel = rate * self.model.coast_s
            else:
                end = move.end if move.end is not None else now + self.model.lookahead_s
                travel = rate * (max(0.0, end - move.start) + self.model.coast_s)
            if sign == 0:
                low, high = low - travel, high + travel
            elif sign * move.direction > 0:
                high += travel
            else:
                low -= travel
        return low, high


def fresh_count_range(model: TravelModel, axis: str, count: float,
                      speed_percent: int) -> Tuple[float, float]:
    """Bound for a move starting now from an up-to-date reading of `count`."""
    reach = model.settle_counts + model.rate(axis, speed_percent) * (model.lookahead_s + model.coast_s)
    return count - reach, count + reach


def reaches_limit(model: TravelModel, axis: str, direction: int,
                  count_range: Tuple[float, float], limits: Tuple[int, int]) -> bool:
    """True when moving `axis` in `direction` could reach either relevant limit."""
    low, high = count_range
    lo_limit, hi_limit = limits
    sign = model.count_sign.get(axis, 0)
    if sign == 0:
        return high >= hi_limit or low <= lo_limit
    if sign * direction > 0:
        return high >= hi_limit
    return low <= lo_limit


# Fitted on Bingo's 2026-11-16 traces (manual jogs at SPEED 20 and 100,
# auto-track at 10, 20, 30, 50 and 83): replaying every LISTPV reading through this
# estimator, the smallest rates that never let a reading fall outside its
# bound were base 1642, shoulder 2063, elbow 2330 and pitch 906 at coast 0.2 s.
# These add ~20% headroom. Pitch at SPEED 10 and 20 fitted lower (458, 788)
# than at 50, so its SPEED 50 figure bounds the capped speeds too.
#
# Signs measured on Bingo (2026-11-13 and 11-16 traces): azimuth +1 (key q)
# lowers the base count, elbow jog +1 (e) lowers the elbow count, shoulder jog
# +1 (w) raises the shoulder count, and altitude +1 (4) raises pitch. The
# shoulder breaks the digit-raises/letter-lowers pattern of the other three.
DEFAULT_TRAVEL_MODEL = TravelModel(
    rate_at_100={"base": 2000.0, "elbow": 2800.0, "shoulder": 2500.0, "pitch": 1100.0},
    count_sign={"base": -1, "elbow": -1, "shoulder": 1, "pitch": 1},
)
