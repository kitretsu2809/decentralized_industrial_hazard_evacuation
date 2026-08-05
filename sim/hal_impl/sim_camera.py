"""
Simulated camera controller.

Reads video frames from files using OpenCV. In a real deployment,
this would capture frames from CSI/USB cameras via GStreamer.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from core.hal.base import CameraHAL

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. SimCamera will return None frames.")


class SimCamera(CameraHAL):
    """Simulated camera reading frames from video files.

    Maps node IDs to video file paths. Each call to capture_frame()
    reads the next frame from the assigned video.
    """

    def __init__(self):
        self._captures: Dict[str, "cv2.VideoCapture"] = {}
        self._video_paths: Dict[str, str] = {}
        self._resolutions: Dict[str, Tuple[int, int]] = {}

    def assign_video(self, node_id: str, video_path: str) -> bool:
        """Assign a video file to a specific node.

        Args:
            node_id: The router node ID.
            video_path: Path to the video file.

        Returns:
            True if the video was opened successfully.
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available. Cannot assign video.")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[CAMERA] Failed to open video: {video_path}")
            return False

        # Close previous capture if exists
        if node_id in self._captures:
            self._captures[node_id].release()

        self._captures[node_id] = cap
        self._video_paths[node_id] = video_path
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolutions[node_id] = (w, h)
        logger.info(f"[CAMERA] Node {node_id}: Assigned video {video_path} ({w}x{h})")
        return True

    def capture_frame(self, node_id: str) -> Optional[np.ndarray]:
        """Capture the next frame from the assigned video.

        Loops the video when it reaches the end.

        Returns:
            RGB numpy array (H, W, 3) or None if no video assigned.
        """
        if not CV2_AVAILABLE or node_id not in self._captures:
            return None

        cap = self._captures[node_id]
        ret, frame = cap.read()

        if not ret:
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None

        # Convert BGR (OpenCV default) to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def is_available(self, node_id: str) -> bool:
        """Check if a camera (video) is assigned and available."""
        return node_id in self._captures and self._captures[node_id].isOpened()

    def get_resolution(self, node_id: str) -> Tuple[int, int]:
        """Get video resolution for a node."""
        return self._resolutions.get(node_id, (0, 0))

    def release_all(self) -> None:
        """Release all video captures."""
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        self._video_paths.clear()
        self._resolutions.clear()

    def __del__(self):
        self.release_all()
