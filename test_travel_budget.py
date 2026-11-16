"""Joint limits hold during a continuous follow, while Operator cannot poll."""

import pytest

from controller import UNANCHORED_MOTION_S, TrackingController, TrackingState
from operator_client import RobotTelemetry
from travel_budget import DEFAULT_TRAVEL_MODEL, TravelEstimator, TravelModel, reaches_limit

MODEL = TravelModel(
    rate_at_100={"base": 1800.0, "elbow": 2000.0, "shoulder": 2000.0, "pitch": 2000.0},
    count_sign={"base": -1, "elbow": -1, "shoulder": 0, "pitch": 1},
    coast_s=0.4,
    lookahead_s=0.05,
    settle_counts=0,  # Exact expected values below; settle is tested on its own.
)


def reading(at, **counts):
    return RobotTelemetry(**counts, last_updated_monotonic=at, has_received=True)


class FakeOperator:
    is_connected = True

    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.stops = 0
        self.accept_starts = True
        self.speeds = []

    def get_telemetry(self):
        return self.telemetry

    def get_fault(self):
        return None

    def tracking_jog_start(self, *_):
        return self.accept_starts

    def joint_jog_start(self, *_):
        return self.accept_starts

    def erv_set_speed_percent(self, percent):
        self.speeds.append(percent)
        return True

    def polar_pan_continuous_stop(self):
        self.stops += 1

    def joint_jog_stop(self):
        self.stops += 1


def controller(telemetry, **kwargs):
    operator = FakeOperator(telemetry)
    tracker = TrackingController(operator, continuous_tracking=True, speed_percent=50,
                                 travel_model=MODEL, **kwargs)
    tracker.last_observation_time = 1e9  # Camera freshness is not under test.
    tracker.last_motion_refresh = 1e9
    return tracker, operator


def test_range_grows_with_worst_case_travel_and_coast():
    estimator = TravelEstimator(MODEL)
    estimator.start("base", 1, now=10.0, speed_percent=50)
    estimator.stop(now=11.0)
    low, high = estimator.count_range("base", reading(9.9, base=0), now=11.0, speed_percent=50)
    # +azimuth lowers the base count on Bingo: 900/s for 1 s plus 0.4 s coast.
    assert high == 0
    assert low == pytest.approx(-900 * 1.4)


def test_reading_timestamps_allow_for_listpv_sampling_early():
    # Readings are stamped on receipt, up to 0.45 s after LISTPV sampled.
    def low_after(read_at):
        estimator = TravelEstimator(MODEL)
        estimator.start("base", 1, now=10.0, speed_percent=50)
        estimator.stop(now=11.0)
        return estimator.count_range("base", reading(read_at, base=-900), now=12.0, speed_percent=50)[0]
    # Sampled possibly mid-move: the whole move may still be ahead of it.
    assert low_after(11.3) == pytest.approx(-900 - 900 * 1.4)
    # Sampled after the move, possibly mid-coast: only the coast may be missing.
    assert low_after(11.6) == pytest.approx(-900 - 900 * 0.4)
    # Sampled after the coast: the reading is the estimate.
    assert low_after(11.9) == -900


def test_continuous_follow_stops_before_the_limit_without_new_telemetry():
    tracker, operator = controller(reading(0.0, base=-4000))
    assert tracker._start_pulse(1, 0, 0.1, now=0.1)
    stopped_at = None
    t = 0.1
    while t < 3.0 and stopped_at is None:
        t += 1 / 60
        tracker.service_motion(now=t)
        if not tracker.is_currently_moving:
            stopped_at = t
    assert stopped_at is not None and operator.stops == 1
    assert tracker.state == TrackingState.LIMIT_REACHED
    # 1000 counts of room at 900/s, minus 0.45 s of coast and lookahead.
    assert stopped_at - 0.1 == pytest.approx((1000 - 900 * 0.45) / 900, abs=0.03)
    # Even at the worst-case rate, the arm stops short of -5000.
    assert -4000 - 900 * (stopped_at - 0.1 + MODEL.coast_s) > -5000


def test_base_limit_uses_bingos_count_sign():
    # Near +5000, +azimuth lowers the count (safe) and -azimuth raises it (blocked).
    tracker, _ = controller(reading(0.0, base=4900))
    assert not tracker._limit_block("base", 1, now=0.1)
    assert tracker._limit_block("base", -1, now=0.1) == "base would reach its limit"


def test_arm_without_telemetry_is_not_limited():
    tracker, _ = controller(RobotTelemetry())
    assert not tracker._limit_block("base", 1, now=0.1)


def test_motion_on_a_stale_reading_stops_after_a_second_until_a_fresh_one():
    # A stale reading cannot anchor a bound (homing or a Commander MOVE may have
    # moved the arm), so motion is capped and the stop lets Operator poll.
    tracker, operator = controller(reading(0.0, base=-4999))
    assert not tracker._limit_block("base", 1, now=5.0)
    assert tracker._start_pulse(1, 0, 0.1, now=5.0)
    t = 5.0
    while tracker.is_currently_moving and t < 8.0:
        t += 1 / 60
        tracker.service_motion(now=t)
    assert t - 5.0 == pytest.approx(UNANCHORED_MOTION_S - MODEL.lookahead_s, abs=0.02)
    assert "fresh joint reading" in tracker.last_thought
    # Still blocked until a new reading lands, then anchored to it.
    assert tracker._limit_block("base", 1, now=t + 0.3)
    operator.telemetry = reading(t + 0.7, base=0)
    assert not tracker._limit_block("base", 1, now=t + 0.8)


def test_unknown_sign_is_bounded_both_ways():
    assert reaches_limit(MODEL, "shoulder", 1, (-90, 90), (-100, 80))
    assert reaches_limit(MODEL, "shoulder", -1, (-90, 90), (-100, 80))
    assert not reaches_limit(MODEL, "shoulder", 1, (-90, 70), (-100, 80))


def test_vertical_selection_skips_an_elbow_that_would_reach_its_limit():
    tracker, _ = controller(reading(0.0, elbow=-4900))
    tracker.elbow_limits = (-5000, 5000)
    tracker.track_elbow = True
    # Image-up commands elbow -1, which raises the elbow count: room to move.
    from vertical_axes import select_vertical_motion
    pick = lambda image_up: select_vertical_motion(
        elbow=True, shoulder=False, polar_vertical=True, image_up=image_up,
        blocked=lambda axis, direction: bool(tracker._limit_block(axis, direction, 0.1)))
    assert pick(True) == ("joint", 3, -1)
    # Image-down commands elbow +1, toward -5000, so it falls through to polar.
    assert pick(False) == ("polar", -1)


def test_default_model_covers_every_axis():
    for axis in ("base", "shoulder", "elbow", "pitch"):
        assert DEFAULT_TRAVEL_MODEL.rate_at_100[axis] > 0
        assert axis in DEFAULT_TRAVEL_MODEL.count_sign


def test_reversals_widen_both_sides():
    estimator = TravelEstimator(MODEL)
    estimator.start("base", 1, now=1.0, speed_percent=50)
    estimator.start("base", -1, now=2.0, speed_percent=50)
    estimator.stop(now=3.0)
    low, high = estimator.count_range("base", reading(0.9, base=0), now=3.0, speed_percent=50)
    assert (low, high) == (pytest.approx(-900 * 1.4), pytest.approx(900 * 1.4))


def test_open_move_keeps_its_fastest_speed():
    estimator = TravelEstimator(MODEL)
    estimator.start("base", 1, now=0.0, speed_percent=20)
    estimator.count_range("base", reading(0.0, base=0), now=1.0, speed_percent=100)
    low, _ = estimator.count_range("base", reading(0.0, base=0), now=2.0, speed_percent=20)
    assert low == pytest.approx(-1800 * (2.0 + MODEL.lookahead_s + MODEL.coast_s))


def test_rejected_axis_switch_closes_the_old_move():
    tracker, operator = controller(reading(0.0, elbow=0, base=0))
    tracker.elbow_limits = (-5000, 5000)
    assert tracker._start_joint(3, 1, 0.1, now=0.1)
    operator.accept_starts = False
    assert not tracker._start_pulse(1, 0, 0.1, now=0.5)
    # The elbow moved 0.4 s; its bound must stop growing after the switch.
    first = tracker.travel.count_range("elbow", operator.telemetry, now=1.0, speed_percent=50)
    later = tracker.travel.count_range("elbow", operator.telemetry, now=1.2, speed_percent=50)
    assert first == later == (pytest.approx(-1000 * 0.8), 0)


def test_joint_moves_run_at_the_capped_speed_and_base_moves_restore_it():
    tracker, operator = controller(reading(0.0, base=0, elbow=0), axis_max_speed_percent={"elbow": 30})
    tracker._applied_speed = 50  # As set when tracking was enabled.
    assert tracker._start_joint(3, -1, 0.1, now=0.1)
    assert tracker._start_joint(3, -1, 0.1, now=0.2)  # A refresh sends nothing.
    assert tracker._start_pulse(1, 0, 0.1, now=0.3)
    assert operator.speeds == [30, 50]
    # The estimate uses the speed each move actually ran at.
    elbow_move, base_move = tracker.travel._moves[-2:]
    assert (elbow_move.speed_percent, base_move.speed_percent) == (30, 50)


def test_settle_allowance_widens_every_bound():
    model = TravelModel(MODEL.rate_at_100, MODEL.count_sign, settle_counts=25)
    assert TravelEstimator(model).count_range("base", reading(0.0, base=100), now=0.1, speed_percent=50) == (75, 125)
