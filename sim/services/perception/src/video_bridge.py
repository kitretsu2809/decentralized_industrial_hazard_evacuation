import threading
import time
import cv2
from typing import Dict

from sim.services.perception.src.yolo_detector import ThreatDetector
from sim.services.perception.src.event_bus import PerceptionEventBus
from core.hazard.hazard_vector import compute_detection_hazard
from core.hazard.threat_types import get_threat_mapping
from core.messaging.schemas import HazardUpdateMsg
from core.messaging.constants import DEFAULT_REDIS_URL

class VideoBridge:
    def __init__(self, redis_url: str = DEFAULT_REDIS_URL):
        self.redis_url = redis_url
        self.event_bus = PerceptionEventBus(redis_url)
        self.detector = ThreatDetector()
        self.feeds: Dict[str, Dict] = {} # node_id -> info
        self.running = False
        self.threads: list = []
        
    def assign_video(self, node_id: str, floor: int, video_path: str):
        self.feeds[node_id] = {
            'floor': floor,
            'video_path': video_path,
            'status': 'assigned'
        }
        
    def _process_video(self, node_id: str):
        info = self.feeds[node_id]
        video_path = info['video_path']
        floor = info['floor']
        info['status'] = 'processing'
        
        cap = cv2.VideoCapture(video_path)
        
        while self.running:
            if not cap.isOpened():
                break
                
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            detections = self.detector.get_threat_detections(frame)
            
            if detections:
                for d in detections:
                    if d.threat_type:
                        mapping = get_threat_mapping(d.threat_type)
                        hazard_score = compute_detection_hazard(
                            threat_type_value=d.threat_type.value,
                            confidence=d.confidence,
                            base_hazard=mapping.base_hazard_score
                        )
                        
                        self.event_bus.publish_to_hazard_channel(
                            node_id=node_id,
                            floor=floor,
                            threat_type=d.threat_type.name,
                            hazard_score=hazard_score
                        )
            
            # Rate limit to ~10 FPS
            time.sleep(0.1)
            
        cap.release()
        info['status'] = 'stopped'

    def start_processing(self):
        self.running = True
        for node_id in self.feeds:
            t = threading.Thread(target=self._process_video, args=(node_id,), daemon=True)
            t.start()
            self.threads.append(t)
            
    def stop_processing(self):
        self.running = False
        for t in self.threads:
            t.join()
        self.threads.clear()
        
    def get_active_feeds(self) -> Dict[str, Dict]:
        return self.feeds
        
if __name__ == "__main__":
    import os
    print("Starting Video Bridge...")
    redis_url = os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    bridge = VideoBridge(redis_url=redis_url)
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop_processing()
