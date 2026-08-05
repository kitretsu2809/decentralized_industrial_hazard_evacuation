import time
import json
import redis
from core.messaging.schemas import HazardUpdateMsg
from core.messaging.constants import CHANNEL_HAZARD_UPDATE

class PerceptionEventBus:
    def __init__(self, redis_url: str):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        
    def publish_detection(self, update: HazardUpdateMsg):
        self.redis_client.publish(CHANNEL_HAZARD_UPDATE, update.model_dump_json())
        
    def publish_to_hazard_channel(self, node_id: str, floor: int, threat_type: str, hazard_score: float):
        msg = HazardUpdateMsg(
            node_id=node_id,
            floor=floor,
            threat_type=threat_type,
            hazard_score=hazard_score,
            timestamp=time.time()
        )
        self.publish_detection(msg)
        
    def close(self):
        self.redis_client.close()
