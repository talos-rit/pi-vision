"""Vertical axis selection stays on the Pi. Commander only sends which axes are allowed."""

from types import SimpleNamespace

from vertical_axes import select_vertical_motion
from web_server import apply_vertical_axes


def test_order_is_elbow_then_shoulder_then_polar():
    common = dict(
        image_up=True,
        elbow_count=0,
        shoulder_count=0,
        pitch_count=0,
        counts_fresh=True,
        elbow_limits=(-5, 5),
        shoulder_limits=(-5, 5),
        pitch_limits=(-5, 5),
    )
    assert select_vertical_motion(elbow=True, shoulder=True, polar_vertical=True, **common) == ("joint", 3, -1)
    at_elbow_stop = {**common, "elbow_count": -5}
    assert select_vertical_motion(elbow=True, shoulder=True, polar_vertical=True, **at_elbow_stop) == ("joint", 2, 1)
    assert select_vertical_motion(elbow=True, shoulder=False, polar_vertical=False, **at_elbow_stop) is None
    assert select_vertical_motion(
        elbow=True, shoulder=True, polar_vertical=True, **{**at_elbow_stop, "counts_fresh": False}
    ) == ("joint", 3, -1)


class Controller:
    def __init__(self):
        self.calls = []

    def set_vertical_axes(self, **kwargs):
        self.calls.append(kwargs)


def test_lease_body_sets_axes_without_commanding_a_disabled_one():
    controller = Controller()
    apply_vertical_axes(controller, {
        "vertical_axes": {"elbow": True, "shoulder": False, "polar_vertical": False},
        "elbow_limits": [-100, 100],
        "shoulder_limits": None,
    })
    assert controller.calls == [{
        "elbow": True,
        "shoulder": False,
        "polar_vertical": False,
        "elbow_limits": (-100, 100),
        "elbow_limits_set": True,
        "shoulder_limits": None,
        "shoulder_limits_set": True,
    }]


def test_controller_jogs_the_elbow_and_does_not_send_polar():
    import controller as controller_module

    class Operator:
        is_connected = True
        calls = []

        def get_fault(self):
            return None

        def get_telemetry(self):
            return SimpleNamespace(
                base=0, shoulder=0, elbow=0, wrist_pitch=0, wrist_roll=0,
                has_received=False, last_updated_monotonic=0,
                is_fresh=lambda *_args, **_kwargs: False,
            )

        def erv_enable_control(self):
            return True

        def erv_set_speed_percent(self, _percent):
            return True

        def joint_jog_start(self, axis, direction):
            self.calls.append(("joint", axis, direction))
            return True

        def joint_jog_stop(self):
            self.calls.append(("joint_stop",))
            return True

        def tracking_jog_start(self, *_args):
            self.calls.append(("tracking", _args))
            return True

        def polar_pan_continuous_stop(self):
            self.calls.append(("polar_stop",))
            return True

        def polar_pan_continuous_start(self, *_args):
            self.calls.append(("polar", _args))
            return True

    operator = Operator()
    tracker = controller_module.TrackingController(operator, continuous_tracking=True)
    assert tracker.set_enabled(True)
    tracker.set_vertical_axes(elbow=True, shoulder=False, polar_vertical=False, elbow_limits_set=True, elbow_limits=None)
    detection = SimpleNamespace(center=(100, 10), box=(80, 0, 120, 30), confidence=0.9, keypoints=None)
    status = tracker.update(200, 200, [detection], observation_time=tracker.last_observation_time or __import__("time").monotonic())

    assert ("joint", 3, -1) in operator.calls
    assert not any(name == "tracking" for name, *_rest in operator.calls)
    assert status.state == controller_module.TrackingState.TRACKING
