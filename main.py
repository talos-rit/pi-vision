"""
Main entry point for Pi Edge Tracker.
Initializes camera ingestion, object detection, closed-loop tracking,
and starts the diagnostic Web UI.
"""

import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
import uvicorn
from loguru import logger

try:
    from .config import CONFIG
    from .controller import ControllerStatus, TrackingController
    from .detector import Detection
    from .pose import PoseDetector
    from .operator_client import OperatorClient
    from .web_server import create_app
except ImportError:
    from config import CONFIG
    from controller import ControllerStatus, TrackingController
    from detector import Detection
    from pose import PoseDetector
    from operator_client import OperatorClient
    from web_server import create_app


@dataclass(frozen=True)
class FrameSample:
    frame: np.ndarray
    sequence: int
    captured_monotonic: float


class CameraStream:
    """Threaded camera capture to ensure zero frame-buffering lag with RTSP or V4L2."""

    def __init__(self, src: str):
        self.src = src
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_sequence = 0
        self.latest_captured_monotonic = 0.0
        self.is_running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a non-consuming frame copy for diagnostics/video streaming."""
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_latest_sample(self, after_sequence: int = -1) -> Optional[FrameSample]:
        """Return only the newest captured frame, never an inference backlog."""
        with self._lock:
            if self.latest_frame is None or self.latest_sequence <= after_sequence:
                return None
            return FrameSample(
                frame=self.latest_frame.copy(),
                sequence=self.latest_sequence,
                captured_monotonic=self.latest_captured_monotonic,
            )

    def _capture_loop(self):
        # Force FFmpeg RTSP to operate in zero-latency mode: discard all queued frames!
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0|probesize;32"
        logger.info(f"[Camera] Opening zero-latency video capture from: {self.src}")
        while self.is_running:
            source = int(self.src) if str(self.src).isdigit() else self.src
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                logger.warning(f"[Camera] Failed to open capture source {self.src}. Retrying in 2s...")
                time.sleep(2.0)
                continue

            logger.info(f"[Camera] Connected to video source {self.src} in zero-latency mode.")
            while self.is_running and self.cap.isOpened():
                # Rapid grab loop: dumps old frames to always stay on current frame
                grabbed = self.cap.grab()
                if not grabbed:
                    logger.warning("[Camera] Frame grab failed, reconnecting...")
                    break

                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    with self._lock:
                        self.latest_frame = frame
                        self.latest_sequence += 1
                        self.latest_captured_monotonic = time.monotonic()

            if self.cap:
                self.cap.release()
            time.sleep(1.0)


@dataclass
class PoseTuning:
    roi_x: float = 0.0
    roi_y: float = 0.0
    roi_w: float = 1.0
    roi_h: float = 1.0
    person_confidence: float = 0.35

    def validate(self) -> None:
        if not all(0.0 <= value <= 1.0 for value in self.__dict__.values()):
            raise ValueError("all perception values must be in [0, 1]")
        if self.roi_w <= 0 or self.roi_h <= 0 or self.roi_x + self.roi_w > 1 or self.roi_y + self.roi_h > 1:
            raise ValueError("ROI must be a non-empty rectangle inside the image")


def _update_pose_tuning(tuning: PoseTuning, lock: threading.Lock, values: dict) -> dict:
    with lock:
        candidate = PoseTuning(**{**tuning.__dict__, **values})
        candidate.validate()
        tuning.__dict__.update(candidate.__dict__)
        return tuning.__dict__.copy()


def main():
    parser = argparse.ArgumentParser(description="TALOS Pi Edge Tracker")
    parser.add_argument("--camera", default=CONFIG.camera_source, help="Camera RTSP URL or V4L2 device index")
    parser.add_argument("--operator-host", default=CONFIG.operator_host, help="Operator TCP IP")
    parser.add_argument("--operator-port", type=int, default=CONFIG.operator_port, help="Operator TCP Port")
    parser.add_argument("--model", default=CONFIG.model_weights, help="Path to YOLO .pt/.onnx model")
    parser.add_argument("--port", type=int, default=CONFIG.web_port, help="Web UI port")
    parser.add_argument("--host", default=CONFIG.web_host, help="Web UI host")
    parser.add_argument("--auto-start", action="store_true", help="Start tracking automatically on launch")
    args = parser.parse_args()

    logger.info("==========================================")
    logger.info("   TALOS Raspberry Pi Edge Tracker")
    logger.info("==========================================")

    # Initialize the production detector before starting any robot-facing service.
    detector = PoseDetector(args.model, conf_thresh=CONFIG.confidence_threshold)
    pose_tuning = PoseTuning()
    pose_lock = threading.Lock()

    camera = CameraStream(args.camera)
    camera.start()
    operator = OperatorClient(host=args.operator_host, port=args.operator_port, auto_connect=True)

    # 4. Initialize Closed Loop Controller
    controller = TrackingController(
        operator_client=operator,
        acceptable_ratio=CONFIG.acceptable_box_ratio,
        target_lost_timeout=CONFIG.target_lost_timeout_s,
        enable_joint_limits=CONFIG.enable_joint_limits,
        base_limits=(CONFIG.base_min_count, CONFIG.base_max_count),
        pitch_limits=(CONFIG.pitch_min_count, CONFIG.pitch_max_count)
        if CONFIG.pitch_min_count is not None and CONFIG.pitch_max_count is not None
        else None,
        telemetry_timeout=CONFIG.telemetry_timeout_s,
        max_observation_age=CONFIG.max_observation_age_s,
        settle_time=CONFIG.settle_time_s,
        min_pulse=CONFIG.min_pulse_s,
        min_pitch_pulse=CONFIG.min_pitch_pulse_s,
        max_base_pulse=CONFIG.max_base_pulse_s,
        max_pitch_pulse=CONFIG.max_pitch_pulse_s,
        speed_percent=CONFIG.default_speed_percent,
    )

    if args.auto_start:
        controller.set_enabled(True)

    # Shared state between tracking thread and web server
    latest_status = ControllerStatus()
    status_lock = threading.Lock()

    def get_latest_status_snapshot() -> ControllerStatus:
        with status_lock:
            return latest_status

    # Stop deadlines run independently of camera and inference latency.
    def motion_service_worker():
        interval = 1.0 / CONFIG.control_rate_hz
        while True:
            controller.service_motion()
            time.sleep(interval)

    motion_thread = threading.Thread(target=motion_service_worker, daemon=True, name="MotionSafetyThread")
    motion_thread.start()

    # Detect every newly available latest frame; never queue old observations.
    def tracking_worker():
        nonlocal latest_status
        fps_counter = 0
        fps_start = time.monotonic()
        current_fps = 0.0
        last_sequence = -1

        logger.info("[Tracker] Latest-frame inference loop started (no frame queue).")

        while True:
            sample = camera.get_latest_sample(after_sequence=last_sequence)
            if sample is not None:
                last_sequence = sample.sequence
                h, w = sample.frame.shape[:2]
                with pose_lock:
                    tuning = PoseTuning(**pose_tuning.__dict__)
                rx, ry, rw, rh = int(w * tuning.roi_x), int(h * tuning.roi_y), int(w * tuning.roi_w), int(h * tuning.roi_h)
                roi_frame = sample.frame[ry:ry + rh, rx:rx + rw]
                detector.conf_thresh = tuning.person_confidence

                try:
                    roi_detections = detector.detect(roi_frame)
                    detections = [Detection((item.box[0] + rx, item.box[1] + ry, item.box[2] + rx, item.box[3] + ry), item.confidence, item.class_id, item.class_name, (item.aim_point[0] + rx, item.aim_point[1] + ry) if item.aim_point else None, tuple((x + rx, y + ry, score) for x, y, score in item.keypoints) if item.keypoints else None) for item in roi_detections]
                except RuntimeError as error:
                    controller.fail_closed(str(error))
                    with status_lock:
                        latest_status = ControllerStatus(
                            enabled=False,
                            state=controller.state,
                            last_thought=controller.last_thought,
                            current_command="BLOCKED (DETECTOR FAILURE)",
                            recent_logs=list(controller.logs),
                        )
                    logger.exception("[Tracker] Detector failed; autonomous motion disabled.")
                    return

                status = controller.update(
                    w,
                    h,
                    detections,
                    observation_time=sample.captured_monotonic,
                    observation_seq=sample.sequence,
                )

                # Track FPS
                fps_counter += 1
                if time.monotonic() - fps_start >= 1.0:
                    current_fps = fps_counter / (time.monotonic() - fps_start)
                    fps_counter = 0
                    fps_start = time.monotonic()

                status.fps = current_fps

                with status_lock:
                    latest_status = status
            else:
                time.sleep(0.002)

    tracker_thread = threading.Thread(target=tracking_worker, daemon=True, name="TrackingWorkerThread")
    tracker_thread.start()

    # 6. Start Web UI & API
    app = create_app(
        controller=controller,
        detector=detector,
        operator=operator,
        get_latest_frame=camera.get_frame,
        get_latest_status=get_latest_status_snapshot,
        get_pose_tuning=lambda: pose_tuning.__dict__.copy(),
        update_pose_tuning=lambda values: _update_pose_tuning(pose_tuning, pose_lock, values),
    )

    logger.info(f"[WebUI] Starting Web UI on http://{args.host}:{args.port}")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        controller.set_enabled(False)
        camera.stop()
        operator.stop()


if __name__ == "__main__":
    main()
