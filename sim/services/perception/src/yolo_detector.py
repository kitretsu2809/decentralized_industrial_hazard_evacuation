import numpy as np
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable

from core.hazard.threat_types import ThreatType, classify_yolo_detection

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    threat_type: Optional[ThreatType]

class ThreatDetector:
    def __init__(self, model_name: Optional[str] = None, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_name = model_name or os.environ.get("YOLO_MODEL_PATH")
        self.device = os.environ.get("YOLO_DEVICE", "cpu").strip() or "cpu"
        self.load_error = None
        if HAS_YOLO and self.model_name and Path(self.model_name).is_file():
            try:
                self.model = YOLO(self.model_name)
            except Exception as e:
                self.load_error = f"Failed to load YOLO model {self.model_name}: {e}"
        elif not HAS_YOLO:
            self.load_error = "Ultralytics is not installed"
        elif self.model_name:
            self.load_error = f"Local YOLO checkpoint not found: {self.model_name}"
        else:
            self.load_error = "No local YOLO_MODEL_PATH configured"

    @property
    def backend_status(self) -> dict:
        return {
            "name": "Ultralytics YOLO",
            "available": self.is_available(),
            "opencv_available": HAS_CV2,
            "model": self.model_name if self.is_available() else None,
            "device": self.device,
            "load_error": self.load_error,
        }

    def is_available(self) -> bool:
        return self.model is not None

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        if not self.is_available():
            return []
            
        results = self.model.predict(frame, verbose=False, device=self.device)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                threat_type = classify_yolo_detection(class_name, conf)
                
                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    threat_type=threat_type
                ))
                
        return detections
        
    def get_threat_detections(self, frame: np.ndarray) -> List[Detection]:
        detections = self.detect_frame(frame)
        return [d for d in detections if d.threat_type is not None]
        
    def detect_video(self, video_path: str, callback: Callable[[int, List[Detection]], None], frame_skip: int = 1):
        if not HAS_CV2:
            raise RuntimeError("OpenCV is required for video decoding")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video {video_path}")
            return
            
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip == 0:
                detections = self.get_threat_detections(frame)
                callback(frame_idx, detections)
                
            frame_idx += 1
            
        cap.release()

def detect_fire_heuristic(frame: np.ndarray) -> Tuple[bool, float]:
    """Uses color analysis (orange-red pixels in HSV space) to detect fire."""
    if not HAS_CV2:
        return False, 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for orange-red color
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    
    mask = mask1 | mask2 | mask3
    
    fire_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    ratio = fire_pixels / total_pixels
    
    # Heuristic: if more than 0.5% of pixels are fire-colored
    if ratio > 0.005:
        # Confidence scales with ratio, capped at 1.0
        conf = min(1.0, ratio * 50)
        return True, float(conf)
    return False, 0.0
