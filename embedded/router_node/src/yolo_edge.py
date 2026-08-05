"""
YOLOv8-nano Edge Inference

Optimized for Jetson Nano using TensorRT when available,
falling back to ONNX Runtime or Ultralytics.
"""
from typing import List, Dict
import time
import numpy as np

class YOLOEdge:
    def __init__(self, model_path: str = 'yolov8n.pt', use_tensorrt: bool = True):
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self._last_latency = 0.0
        
        # Skeleton implementation for model loading
        if self.use_tensorrt:
            self.engine = self._load_tensorrt_engine()
        else:
            self.engine = self._load_onnx_engine()

    def _load_tensorrt_engine(self):
        # Placeholder for TensorRT engine loading logic
        return "TRT_ENGINE_MOCK"

    def _load_onnx_engine(self):
        # Placeholder for ONNX engine loading logic
        return "ONNX_ENGINE_MOCK"

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run inference on a frame."""
        t0 = time.time()
        
        # Mock detection logic
        detections = [
            {'class_name': 'person', 'confidence': 0.85, 'bbox': [10, 10, 50, 100], 'threat_type': 'none'}
        ]
        
        self._last_latency = (time.time() - t0) * 1000.0
        return detections

    def get_latency_ms(self) -> float:
        """Returns the latency of the last inference in ms."""
        return self._last_latency
