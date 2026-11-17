"""Commander can change tracking speed without releasing the lease."""

import pytest
from fastapi import HTTPException

from control_api import ControlLease
from controller import TrackingController

OWNER = "commander-test"


class Operator:
    is_connected = True

    def __init__(self, accept=True):
        self.accept = accept
        self.speeds = []

    def get_fault(self):
        return None

    def erv_enable_control(self):
        return True

    def erv_set_speed_percent(self, percent):
        self.speeds.append(percent)
        return self.accept

    def get_telemetry(self):
        return type("Telemetry", (), {"has_received": False})()


def lease(operator):
    return ControlLease(TrackingController(operator, speed_percent=20), operator, clock=lambda: 0.0)


def test_speed_applies_immediately_while_tracking():
    operator = Operator()
    control = lease(operator)
    control.update(OWNER, True)
    status = control.tune(OWNER, speed_percent=55)
    assert operator.speeds == [20, 55]
    assert status["speed_percent"] == 55 and status["owner"] == OWNER and status["enabled"]


def test_speed_is_stored_for_the_next_arm_while_idle():
    operator = Operator()
    control = lease(operator)
    control.tune(OWNER, speed_percent=40)
    assert operator.speeds == []
    control.update(OWNER, True)
    assert operator.speeds == [40]


def test_bad_or_rejected_speed_is_an_error():
    control = lease(Operator(accept=False))
    with pytest.raises(HTTPException) as bad:
        control.tune(OWNER, speed_percent=0)
    assert bad.value.status_code == 422
    control.controller.enabled = True
    with pytest.raises(HTTPException) as rejected:
        control.tune(OWNER, speed_percent=50)
    assert rejected.value.status_code == 503

