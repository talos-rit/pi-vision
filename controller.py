"""Fresh-observation visual servo controller for Bingo.

The ER-V cannot provide joint telemetry while its ACL manual-motion path is active.
Tracking therefore uses short, bounded pulses and requires both a newer camera frame
and post-stop joint telemetry before issuing the next pulse. A high-rate caller must
invoke :meth:`service_motion` so stopping never waits for inference to finish.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

from loguru import logger

try:
    from .detector import Detection
    from .operator_client import OperatorClient
except ImportError:
    from detector import Detection
    from operator_client import OperatorClient


class TrackingState(str, Enum):
    DISABLED = "DISABLED"
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    CENTERED = "CENTERED"
    LIMIT_REACHED = "LIMIT"
    WAITING_FEEDBACK = "WAITING_FEEDBACK"
    FAULT = "FAULT"


@dataclass
class ControllerStatus:
    enabled: bool = False
    state: TrackingState = TrackingState.DISABLED
    last_thought: str = "Controller initialized and idle."
    target_center: Optional[Tuple[int, int]] = None
    target_box: Optional[Tuple[int, int, int, int]] = None
    target_confidence: float = 0.0
    target_keypoints: Optional[Tuple[Tuple[int, int, float], ...]] = None
    error_x: int = 0
    error_y: int = 0
    acceptable_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    current_command: str = "NONE"
    moving_azimuth: int = 0
    moving_altitude: int = 0
    fps: float = 0.0
    observation_age_s: Optional[float] = None
    telemetry_age_s: Optional[float] = None
    pulse_remaining_s: float = 0.0
    fault: Optional[str] = None
    recent_logs: List[str] = field(default_factory=list)


class TrackingController:
    """Thread-safe, pulse-and-observe controller for visual tracking."""

    def __init__(
        self,
        operator_client: OperatorClient,
        acceptable_ratio: float = 0.35,
        target_lost_timeout: float = 0.35,
        enable_joint_limits: bool = True,
        base_limits: Tuple[int, int] = (-5000, 5000),
        pitch_limits: Optional[Tuple[int, int]] = None,
        telemetry_timeout: float = 1.25,
        max_observation_age: float = 0.35,
        settle_time: float = 0.12,
        min_pulse: float = 0.055,
        min_pitch_pulse: Optional[float] = None,
        max_base_pulse: float = 0.16,
        max_pitch_pulse: float = 0.09,
        speed_percent: int = 20,
    ):
        if not 0.0 < acceptable_ratio < 1.0:
            raise ValueError("acceptable_ratio must be between zero and one")
        min_pitch_pulse = min_pulse if min_pitch_pulse is None else min_pitch_pulse
        if min_pulse <= 0 or min_pitch_pulse <= 0 or max_base_pulse < min_pulse or max_pitch_pulse < min_pitch_pulse:
            raise ValueError("pulse durations are inconsistent")

        self.operator = operator_client
        self.acceptable_ratio = acceptable_ratio
        self.target_lost_timeout = target_lost_timeout
        self.enable_joint_limits = enable_joint_limits
        self.base_min, self.base_max = base_limits
        self.pitch_limits = pitch_limits
        self.telemetry_timeout = telemetry_timeout
        self.max_observation_age = max_observation_age
        self.settle_time = settle_time
        self.min_pulse = min_pulse
        self.min_pitch_pulse = min_pitch_pulse
        self.max_base_pulse = max_base_pulse
        self.max_pitch_pulse = max_pitch_pulse
        self.speed_percent = max(1, min(100, speed_percent))

        self.enabled = False
        self.state = TrackingState.DISABLED
        self.last_thought = "Idle. Autonomous tracking is OFF."
        self.last_target_time = 0.0
        self.is_currently_moving = False
        self.current_azimuth = 0
        self.current_altitude = 0
        self.pulse_deadline = 0.0
        self.last_motion_stop_time = 0.0
        self.last_observation_time = 0.0
        self.last_observation_seq = -1
        self._implicit_observation_seq = 0
        self._lock = threading.RLock()

        self.logs: Deque[str] = deque(maxlen=20)
        self.tracked_target_center: Optional[Tuple[int, int]] = None
        self._log("Controller initialized; motion requires a current camera frame and Operator connection.")

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        logger.info(f"[Controller] {message}")

    def set_enabled(self, enable: bool) -> bool:
        """Enable tracking when the Operator command connection is healthy."""
        with self._lock:
            if not enable:
                self.enabled = False
                self.state = TrackingState.DISABLED
                self.last_thought = "Tracking disabled by operator. Holding position."
                self.tracked_target_center = None
                self._stop_movement_locked("tracking disabled")
                return True

            if self.enabled:
                return True
            now = time.monotonic()
            fault = self.operator.get_fault()
            if fault:
                self.state = TrackingState.FAULT
                self.last_thought = f"Tracking blocked by latched controller fault: {fault}"
                return False
            if not self.operator.is_connected:
                self.state = TrackingState.WAITING_FEEDBACK
                self.last_thought = "Tracking blocked until the Operator command connection is available."
                return False
            if not self.operator.erv_enable_control():
                self.state = TrackingState.FAULT
                self.last_thought = "Operator rejected the one-time control-enable request."
                return False
            if not self.operator.erv_set_speed_percent(self.speed_percent):
                self.state = TrackingState.FAULT
                self.last_thought = "Operator rejected the tracking speed request."
                return False

            self.enabled = True
            self.state = TrackingState.SEARCHING
            self.last_thought = "Tracking enabled. Waiting for a fresh person detection."
            self.tracked_target_center = None
            self._log(f"Autonomous tracking ENABLED at {self.speed_percent}% speed.")
            return True

    def stop_movement(self, reason: str = "") -> None:
        with self._lock:
            self._stop_movement_locked(reason)

    def fail_closed(self, reason: str) -> None:
        """Disable autonomous motion and expose a latched local failure."""
        with self._lock:
            self.enabled = False
            self._stop_movement_locked(reason)
            self.state = TrackingState.FAULT
            self.last_thought = reason
            self._log(f"Tracking failed closed: {reason}")

    def _stop_movement_locked(self, reason: str = "", now: Optional[float] = None) -> None:
        if not self.is_currently_moving:
            return
        stopped_at = time.monotonic() if now is None else now
        self.operator.polar_pan_continuous_stop()
        self.is_currently_moving = False
        self.current_azimuth = 0
        self.current_altitude = 0
        self.pulse_deadline = 0.0
        self.last_motion_stop_time = stopped_at
        if reason:
            self._log(f"Motion stopped ({reason}).")

    def service_motion(self, now: Optional[float] = None) -> None:
        """Enforce pulse and stale-observation deadlines independently of inference."""
        current = time.monotonic() if now is None else now
        with self._lock:
            if not self.is_currently_moving:
                return
            if current >= self.pulse_deadline:
                self._stop_movement_locked("pulse complete", current)
            elif current - self.last_observation_time > self.max_observation_age:
                self._stop_movement_locked("camera observation became stale", current)

    def calculate_acceptable_box(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        box_w = int(frame_w * self.acceptable_ratio)
        box_h = int(frame_h * self.acceptable_ratio)
        cx, cy = frame_w // 2, frame_h // 2
        return cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2

    def _status(self, box: Tuple[int, int, int, int], now: float, observation_time: float) -> ControllerStatus:
        telemetry = self.operator.get_telemetry()
        telemetry_age = now - telemetry.last_updated_monotonic if telemetry.has_received else None
        return ControllerStatus(
            enabled=self.enabled,
            state=self.state,
            last_thought=self.last_thought,
            acceptable_box=box,
            current_command="PULSE" if self.is_currently_moving else "HOLDING",
            moving_azimuth=self.current_azimuth,
            moving_altitude=self.current_altitude,
            observation_age_s=max(0.0, now - observation_time),
            telemetry_age_s=telemetry_age,
            pulse_remaining_s=max(0.0, self.pulse_deadline - now),
            fault=self.operator.get_fault(),
            recent_logs=list(self.logs),
        )

    @staticmethod
    def _mark_status_stopped(status: ControllerStatus) -> None:
        status.moving_azimuth = 0
        status.moving_altitude = 0
        status.pulse_remaining_s = 0.0

    def update(
        self,
        frame_w: int,
        frame_h: int,
        detections: List[Detection],
        *,
        observation_time: Optional[float] = None,
        observation_seq: Optional[int] = None,
    ) -> ControllerStatus:
        """Process exactly one camera observation and possibly start one short pulse."""
        now = time.monotonic()
        observed_at = now if observation_time is None else observation_time
        with self._lock:
            if observation_seq is None:
                self._implicit_observation_seq += 1
                observation_seq = self._implicit_observation_seq

            box = self.calculate_acceptable_box(frame_w, frame_h)
            left, top, right, bottom = box
            frame_cx, frame_cy = frame_w // 2, frame_h // 2
            status = self._status(box, now, observed_at)

            if not self.enabled:
                self._stop_movement_locked("controller disabled", now)
                status.state = TrackingState.DISABLED
                status.last_thought = self.last_thought
                status.current_command = "STOPPED (DISABLED)"
                self._mark_status_stopped(status)
                return status

            if observation_seq <= self.last_observation_seq:
                return status
            self.last_observation_seq = observation_seq
            self.last_observation_time = observed_at

            observation_age = now - observed_at
            if observation_age > self.max_observation_age:
                self._stop_movement_locked("stale camera frame", now)
                self.state = TrackingState.SEARCHING
                self.last_thought = f"Camera frame is stale ({observation_age:.2f}s); motion inhibited."
                status.state = self.state
                status.last_thought = self.last_thought
                status.current_command = "BLOCKED (STALE VIDEO)"
                self._mark_status_stopped(status)
                return status

            fault = self.operator.get_fault()
            if fault:
                self._stop_movement_locked("controller fault", now)
                self.state = TrackingState.FAULT
                self.last_thought = f"Latched controller fault: {fault}"
                status.state = self.state
                status.last_thought = self.last_thought
                status.fault = fault
                status.current_command = "BLOCKED (FAULT)"
                self._mark_status_stopped(status)
                return status

            if not self.operator.is_connected:
                self._stop_movement_locked("Operator command connection lost", now)
                self.state = TrackingState.SEARCHING
                self.last_thought = "Operator command connection is unavailable."
                status.state = self.state
                status.last_thought = self.last_thought
                status.current_command = "BLOCKED (OPERATOR OFFLINE)"
                self._mark_status_stopped(status)
                return status

            # Telemetry is informational for pacing, but still supplies the
            # conservative base-limit check when it happens to be available.
            telemetry = self.operator.get_telemetry()

            if not detections:
                self._stop_movement_locked("no person in newest frame", now)
                if self.last_target_time and now - self.last_target_time <= self.target_lost_timeout:
                    self.last_thought = "Target absent in newest frame; holding immediately."
                else:
                    self.state = TrackingState.SEARCHING
                    self.tracked_target_center = None
                    self.last_thought = "No recent target. Searching while stationary."
                status.state = self.state
                status.last_thought = self.last_thought
                status.current_command = "HOLDING (NO TARGET)"
                self._mark_status_stopped(status)
                return status

            target = self._select_target(detections, now)
            self.tracked_target_center = target.center
            self.last_target_time = now
            tx, ty = target.center
            err_x, err_y = tx - frame_cx, ty - frame_cy
            status.target_center = (tx, ty)
            status.target_box = target.box
            status.target_confidence = target.confidence
            status.target_keypoints = target.keypoints
            status.error_x = err_x
            status.error_y = err_y

            inside_x = left <= tx <= right
            inside_y = top <= ty <= bottom
            if inside_x and inside_y:
                self._stop_movement_locked("target entered deadband", now)
                self.state = TrackingState.CENTERED
                self.last_thought = f"Target centered (dX={err_x}px, dY={err_y}px)."
                status.state = self.state
                status.last_thought = self.last_thought
                status.current_command = "HOLDING (ON TARGET)"
                self._mark_status_stopped(status)
                return status

            desired_azimuth = 0 if inside_x else (1 if err_x > 0 else -1)
            desired_altitude = 0 if inside_y else (-1 if err_y > 0 else 1)

            if self.is_currently_moving:
                reversed_direction = (
                    (self.current_azimuth and desired_azimuth != self.current_azimuth)
                    or (self.current_altitude and desired_altitude != self.current_altitude)
                )
                if reversed_direction:
                    self._stop_movement_locked("new observation requested reversal", now)
                    status.current_command = "HOLDING (REVERSAL)"
                    self._mark_status_stopped(status)
                self.state = TrackingState.TRACKING
                self.last_thought = "Newest observation accepted; current bounded pulse is finishing."
                status.state = self.state
                status.last_thought = self.last_thought
                return status

            # Camera observations, not slow controller telemetry, pace bounded tracking pulses.

            half_deadband_x = max(1, (right - left) // 2)
            half_deadband_y = max(1, (bottom - top) // 2)
            excess_x = max(0, abs(err_x) - half_deadband_x)
            excess_y = max(0, abs(err_y) - half_deadband_y)
            norm_x = excess_x / max(1, frame_cx - half_deadband_x)
            norm_y = excess_y / max(1, frame_cy - half_deadband_y)

            use_base = desired_azimuth != 0 and (
                desired_altitude == 0 or norm_x >= norm_y or self.pitch_limits is None
            )
            if use_base:
                if self.enable_joint_limits and (
                    (desired_azimuth > 0 and telemetry.base >= self.base_max)
                    or (desired_azimuth < 0 and telemetry.base <= self.base_min)
                ):
                    return self._limit_status(status, f"base count {telemetry.base}")
                pulse = self.min_pulse + (self.max_base_pulse - self.min_pulse) * min(1.0, norm_x)
                if not self._start_pulse(desired_azimuth, 0, pulse, now):
                    return self._fault_status(status, "Operator rejected tracking motion command")
            else:
                if self.pitch_limits is None:
                    self.state = TrackingState.CENTERED
                    self.last_thought = "Vertical correction unavailable; holding base position."
                    status.state = self.state
                    status.last_thought = self.last_thought
                    status.current_command = "HOLDING (PITCH UNCALIBRATED)"
                    self._mark_status_stopped(status)
                    return status
                pitch_min, pitch_max = self.pitch_limits
                if self.enable_joint_limits and (
                    (desired_altitude > 0 and telemetry.wrist_pitch >= pitch_max)
                    or (desired_altitude < 0 and telemetry.wrist_pitch <= pitch_min)
                ):
                    return self._limit_status(status, f"pitch count {telemetry.wrist_pitch}")
                pulse = self.min_pitch_pulse + (self.max_pitch_pulse - self.min_pitch_pulse) * min(1.0, norm_y)
                if not self._start_pulse(0, desired_altitude, pulse, now):
                    return self._fault_status(status, "Operator rejected tracking motion command")

            self.state = TrackingState.TRACKING
            self.last_thought = f"Bounded {pulse * 1000:.0f}ms correction from fresh frame (dX={err_x}, dY={err_y})."
            status.state = self.state
            status.last_thought = self.last_thought
            status.current_command = f"PULSE {pulse * 1000:.0f}ms"
            status.moving_azimuth = self.current_azimuth
            status.moving_altitude = self.current_altitude
            status.pulse_remaining_s = pulse
            return status

    def _select_target(self, detections: List[Detection], now: float) -> Detection:
        if self.tracked_target_center is not None and now - self.last_target_time < 1.5:
            close = sorted(
                detections,
                key=lambda det: (det.center[0] - self.tracked_target_center[0]) ** 2
                + (det.center[1] - self.tracked_target_center[1]) ** 2,
            )
            dx = close[0].center[0] - self.tracked_target_center[0]
            dy = close[0].center[1] - self.tracked_target_center[1]
            if dx * dx + dy * dy < 140 * 140:
                return close[0]
        return detections[0]

    def _start_pulse(self, azimuth: int, altitude: int, duration: float, now: float) -> bool:
        if not self.operator.polar_pan_continuous_start(azimuth, altitude):
            return False
        self.is_currently_moving = True
        self.current_azimuth = azimuth
        self.current_altitude = altitude
        self.pulse_deadline = now + duration
        axis = "base" if azimuth else "pitch"
        direction = azimuth if azimuth else altitude
        self._log(f"Started {duration * 1000:.0f}ms {axis} pulse, direction={direction}.")
        return True

    def _limit_status(self, status: ControllerStatus, reason: str) -> ControllerStatus:
        self.state = TrackingState.LIMIT_REACHED
        self.last_thought = f"Motion blocked by joint safety policy: {reason}."
        status.state = self.state
        status.last_thought = self.last_thought
        status.current_command = "BLOCKED (LIMIT)"
        return status

    def _fault_status(self, status: ControllerStatus, reason: str) -> ControllerStatus:
        self.state = TrackingState.FAULT
        self.last_thought = reason
        status.state = self.state
        status.last_thought = reason
        status.current_command = "BLOCKED (FAULT)"
        return status
