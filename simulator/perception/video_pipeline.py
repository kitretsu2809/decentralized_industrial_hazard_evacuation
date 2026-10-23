"""Recorded-video perception pipeline used by the all-software demonstration.

The pipeline deliberately keeps visual perception and temporal sensor anomaly
detection as separate branches. YOLO supplies object/person detections, the
fire-colour heuristic is an explicitly enabled and disclosed fallback, and the
1D CNN is used only when trained weights are configured.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.hazard.hazard_vector import combine_hazards, compute_detection_hazard
from core.hazard.threat_types import ThreatType, get_threat_mapping
from sim.services.perception.src.sensor_cnn import SensorAnomalyDetector
from sim.services.perception.src.yolo_detector import (
    HAS_CV2,
    Detection,
    ThreatDetector,
    cv2,
    detect_fire_heuristic,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def build_hazard_vector(
    *,
    visual_hazard: float,
    crowd_density: float,
    visual_confidence: float,
    sensor_anomaly: float,
) -> Dict[str, float]:
    """Create the four-component perception vector consumed by the simulator."""
    fused = combine_hazards([_clamp(visual_hazard), _clamp(sensor_anomaly)])
    return {
        "hazard": round(fused, 4),
        "crowd_density": round(max(0.0, min(2.0, float(crowd_density))), 4),
        "visual_confidence": round(_clamp(visual_confidence), 4),
        "sensor_anomaly": round(_clamp(sensor_anomaly), 4),
    }


@dataclass
class PerceptionFeed:
    feed_id: str
    node_id: str
    floor: int
    capacity: int
    video_path: str
    original_name: str
    status: str = "assigned"
    frame_index: int = 0
    detections: List[dict] = field(default_factory=list)
    person_count: int = 0
    frame_width: int = 0
    frame_height: int = 0
    hazard_type: Optional[str] = None
    vector: Dict[str, float] = field(default_factory=lambda: build_hazard_vector(
        visual_hazard=0.0,
        crowd_density=0.0,
        visual_confidence=0.0,
        sensor_anomaly=0.0,
    ))
    error: Optional[str] = None
    updated_at: float = 0.0
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: Optional[threading.Thread] = field(default=None, repr=False)

    def public_dict(self) -> dict:
        return {
            "feed_id": self.feed_id,
            "node_id": self.node_id,
            "floor": self.floor,
            "original_name": self.original_name,
            "status": self.status,
            "frame_index": self.frame_index,
            "detections": list(self.detections),
            "person_count": self.person_count,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "hazard_type": self.hazard_type,
            "vector": dict(self.vector),
            "error": self.error,
            "updated_at": self.updated_at,
            "media_url": f"/api/perception/feeds/{self.feed_id}/media",
        }


class SimulatorPerceptionPipeline:
    """Owns uploaded feeds and publishes node-level perception updates."""

    def __init__(
        self,
        on_update: Callable[[dict], None],
        sensor_provider: Callable[[str], Tuple[float, float, float]],
        detector: Optional[ThreatDetector] = None,
        sensor_detector: Optional[SensorAnomalyDetector] = None,
        inference_fps: float = 2.0,
    ):
        self.on_update = on_update
        self.sensor_provider = sensor_provider
        self.detector = detector or ThreatDetector()
        self.sensor_detector = sensor_detector or SensorAnomalyDetector()
        self.enable_fire_fallback = _env_flag("ENABLE_HSV_FIRE_FALLBACK", default=False)
        self.inference_fps = max(0.2, min(10.0, float(inference_fps)))
        self.feeds: Dict[str, PerceptionFeed] = {}
        self.node_feeds: Dict[str, str] = {}
        self._lock = threading.RLock()

    def capabilities(self) -> dict:
        return {
            "video_decoder": {
                "name": "OpenCV",
                "available": HAS_CV2,
            },
            "visual_detector": self.detector.backend_status,
            "sensor_detector": self.sensor_detector.backend_status,
            "fire_fallback": {
                "name": "HSV fire-colour heuristic (opt-in)",
                "available": HAS_CV2 and self.enable_fire_fallback,
                "configured": self.enable_fire_fallback,
                "disclosed_fallback": True,
            },
        }

    def assign_video(
        self,
        node_id: str,
        floor: int,
        capacity: int,
        video_path: str,
        original_name: str,
    ) -> dict:
        old_id = self.node_feeds.get(node_id)
        if old_id:
            self.stop_feed(old_id)

        feed = PerceptionFeed(
            feed_id=uuid.uuid4().hex,
            node_id=node_id,
            floor=int(floor),
            capacity=max(1, int(capacity)),
            video_path=str(video_path),
            original_name=original_name,
        )
        with self._lock:
            self.feeds[feed.feed_id] = feed
            self.node_feeds[node_id] = feed.feed_id

        if not HAS_CV2:
            feed.status = "unavailable"
            feed.error = "OpenCV is not installed; preview is available but inference is disabled"
            return feed.public_dict()

        feed.thread = threading.Thread(
            target=self._process_video,
            args=(feed.feed_id,),
            daemon=True,
            name=f"perception-{node_id}",
        )
        feed.thread.start()
        return feed.public_dict()

    def _process_video(self, feed_id: str) -> None:
        with self._lock:
            feed = self.feeds[feed_id]
            feed.status = "starting"

        cap = cv2.VideoCapture(feed.video_path)
        if not cap.isOpened():
            with self._lock:
                feed.status = "error"
                feed.error = "Video decoder could not open this file"
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_stride = max(1, int(round(source_fps / self.inference_fps)))
        frame_period = 1.0 / max(1.0, source_fps)
        next_frame_at = time.monotonic()
        feed.status = "processing"

        try:
            while not feed.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    next_frame_at = time.monotonic()
                    continue

                feed.frame_index += 1
                if feed.frame_index % frame_stride == 0:
                    detections = self.detector.detect_frame(frame)
                    update = self._frame_update(feed, frame, detections)
                    self.on_update(update)

                next_frame_at += frame_period
                time.sleep(max(0.0, next_frame_at - time.monotonic()))
        except Exception as exc:
            feed.status = "error"
            feed.error = str(exc)
        finally:
            cap.release()
            if feed.status != "error":
                feed.status = "stopped"

    def _frame_update(self, feed: PerceptionFeed, frame, detections: List[Detection]) -> dict:
        people = [d for d in detections if d.class_name.lower() == "person"]
        threats = [d for d in detections if d.threat_type is not None]
        visual_hazard = 0.0
        visual_confidence = 0.0
        hazard_type: Optional[ThreatType] = None

        for detection in threats:
            mapping = get_threat_mapping(detection.threat_type)
            score = compute_detection_hazard(
                detection.threat_type.value,
                detection.confidence,
                mapping.base_hazard_score,
            )
            if score > visual_hazard:
                visual_hazard = score
                visual_confidence = detection.confidence
                hazard_type = detection.threat_type

        if self.enable_fire_fallback:
            fire_detected, fire_confidence = detect_fire_heuristic(frame)
            if fire_detected:
                fire_score = compute_detection_hazard(
                    ThreatType.FIRE.value,
                    fire_confidence,
                    get_threat_mapping(ThreatType.FIRE).base_hazard_score,
                )
                if fire_score > visual_hazard:
                    visual_hazard = fire_score
                    visual_confidence = fire_confidence
                    hazard_type = ThreatType.FIRE

        crowd_density = len(people) / float(feed.capacity)
        gas, temperature, humidity = self.sensor_provider(feed.node_id)
        self.sensor_detector.add_reading(feed.node_id, gas, temperature, humidity)
        sensor_anomaly = self.sensor_detector.get_hazard_score(feed.node_id)
        vector = build_hazard_vector(
            visual_hazard=visual_hazard,
            crowd_density=crowd_density,
            visual_confidence=visual_confidence,
            sensor_anomaly=sensor_anomaly,
        )

        feed.person_count = len(people)
        feed.frame_height, feed.frame_width = frame.shape[:2]
        feed.hazard_type = hazard_type.name if hazard_type else None
        feed.vector = vector
        feed.detections = [
            {
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 3),
                "bbox": list(d.bbox),
                "threat_type": d.threat_type.name if d.threat_type else None,
            }
            for d in detections[:20]
        ]
        feed.updated_at = time.time()
        feed.status = "processing"
        feed.error = None

        return {
            "node_id": feed.node_id,
            "floor": feed.floor,
            "feed_id": feed.feed_id,
            "hazard_type": feed.hazard_type,
            "person_count": feed.person_count,
            "vector": vector,
            "detections": list(feed.detections),
            "source": "recorded_video",
            "updated_at": feed.updated_at,
        }

    def get_state(self) -> dict:
        with self._lock:
            return {
                "capabilities": self.capabilities(),
                "feeds": {feed_id: feed.public_dict() for feed_id, feed in self.feeds.items()},
                "node_feeds": dict(self.node_feeds),
            }

    def get_feed(self, feed_id: str) -> Optional[PerceptionFeed]:
        with self._lock:
            return self.feeds.get(feed_id)

    def stop_feed(self, feed_id: str) -> bool:
        with self._lock:
            feed = self.feeds.get(feed_id)
        if not feed:
            return False
        feed.stop_event.set()
        if feed.thread and feed.thread.is_alive():
            feed.thread.join(timeout=2.0)
        if self.node_feeds.get(feed.node_id) == feed_id:
            self.node_feeds.pop(feed.node_id, None)
        return True

    def stop_all(self) -> None:
        for feed_id in list(self.feeds):
            self.stop_feed(feed_id)

    def clear_feeds(self) -> None:
        """Stop and detach all feeds when the simulated physical site changes."""
        self.stop_all()
        with self._lock:
            self.feeds.clear()
            self.node_feeds.clear()

    def media_path(self, feed_id: str) -> Optional[Path]:
        feed = self.get_feed(feed_id)
        return Path(feed.video_path) if feed else None
