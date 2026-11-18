import json
import threading
import time
from types import SimpleNamespace

import numpy as np

try:
    from .perception import PerceptionStore
    from .web_server import observation_events
except ImportError:
    from perception import PerceptionStore
    from web_server import observation_events


def frame(sequence):
    return SimpleNamespace(frame=np.zeros((480, 640, 3), np.uint8), sequence=sequence,
                           captured_monotonic=time.monotonic())


def person():
    return SimpleNamespace(box=(10, 20, 110, 220), confidence=0.9, center=(60, 70), keypoints=None)


def parse(event):
    assert event.startswith("data: ") and event.endswith("\n\n")
    return json.loads(event[len("data: "):])


def test_first_event_is_the_current_snapshot():
    store = PerceptionStore()
    store.publish(frame(4), [person()], 0.1)
    payload = parse(next(observation_events(store)))
    assert payload["session_id"] == store.session_id
    assert payload["sample"]["sequence"] == 4
    assert payload["sample"]["detections"][0]["box"] == [10, 20, 110, 220]


def test_publish_wakes_the_stream_before_the_heartbeat():
    store = PerceptionStore()
    events = observation_events(store, heartbeat_s=5)
    next(events)
    threading.Timer(0.05, store.publish, (frame(7), [], 0.1)).start()
    started = time.monotonic()
    payload = parse(next(events))
    assert payload["sample"]["sequence"] == 7
    assert time.monotonic() - started < 1


def test_heartbeat_resends_snapshot_without_new_samples():
    store = PerceptionStore()
    store.publish(frame(1), [], 0.1)
    events = observation_events(store, heartbeat_s=0.05)
    first, second = parse(next(events)), parse(next(events))
    assert first["sample"]["sequence"] == second["sample"]["sequence"] == 1
    assert second["sample"]["age_s"] >= first["sample"]["age_s"]


def test_pausing_perception_is_streamed():
    store = PerceptionStore()
    store.publish(frame(1), [], 0.1)
    events = observation_events(store, heartbeat_s=5)
    next(events)
    store.set_enabled(False)
    payload = parse(next(events))
    assert payload["enabled"] is False and payload["sample"] is None
