import json
import time
from typing import Dict, Optional, Any

from core.messaging.schemas import HazardUpdateMsg, DisasterInjectionMsg
from core.messaging.constants import CHANNEL_DISASTER_INJECT

class DisasterInjector:
    """Processes disaster injection requests and publishes to Redis."""
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def inject(self, threat_type: str, target_node: str, floor: int,
                     intensity: float = 0.8, spread_rate: float = 0.1,
                     metadata: dict = None) -> dict:
        """Inject a disaster into the simulation."""
        
        valid_threats = ['FIRE', 'COLLAPSE', 'GAS', 'WATER', 'CHEMICAL_SPILL', 'EXPLOSION', 'ANIMAL', 'WEAPON', 'SCENARIO', 'CLEAR_ALL', 'SET_WIND']
        if threat_type.upper() not in valid_threats:
            raise ValueError(f"Invalid threat type: {threat_type}")

        msg = DisasterInjectionMsg(
            threat_type=threat_type.upper(),
            target_node=target_node,
            floor=floor,
            intensity=intensity,
            spread_rate=spread_rate,
            metadata=metadata or {}
        )
        
        await self.redis.publish(
            CHANNEL_DISASTER_INJECT, 
            msg.model_dump_json()
        )
        
        return {
            "status": "success", 
            "message": f"Injected {threat_type} at {target_node}",
            "details": msg.model_dump()
        }
