"""
Jetson Camera HAL Implementation
"""
from typing import Optional
import numpy as np
import cv2
import logging
from core.hal.base import CameraHAL

logger = logging.getLogger(__name__)

class JetsonCamera(CameraHAL):
    def __init__(self, use_csi: bool = True, device: str = "/dev/video0", width: int = 640, height: int = 480):
        self.use_csi = use_csi
        self.device = device
        self.width = width
        self.height = height
        self.cap = None
        self._initialize_camera()

    def _gstreamer_pipeline(self) -> str:
        """GStreamer pipeline for Jetson CSI cameras."""
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, format=(string)NV12, framerate=(fraction)30/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink"
        )

    def _initialize_camera(self):
        if self.use_csi:
            logger.info("Initializing CSI camera via GStreamer...")
            self.cap = cv2.VideoCapture(self._gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            logger.info(f"Initializing USB camera on {self.device}...")
            self.cap = cv2.VideoCapture(self.device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
        if not self.cap.isOpened():
            logger.error("Failed to open camera.")

    def capture_frame(self, node_id: str) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"Failed to read frame on node {node_id}")
            return None
        return frame

    def is_available(self, node_id: str) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_resolution(self, node_id: str) -> tuple:
        return (self.width, self.height)

    def release(self):
        if self.cap is not None:
            self.cap.release()
