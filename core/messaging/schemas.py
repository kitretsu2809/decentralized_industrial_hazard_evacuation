from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time

class NodeStateMsg(BaseModel):
    node_id: str
    floor: int
    hazard_score: float
    crowd_count: int
    capacity: int
    node_type: str
    position: List[float]
    edge_states: Dict[str, str]
    sign_directions: Dict[str, str]

class EvacueeMsg(BaseModel):
    id: str
    position: List[float]
    floor: int
    current_edge: Optional[str]
    status: str = 'moving'  # moving, evacuated, casualty

class ActiveThreatMsg(BaseModel):
    threat_id: str
    threat_type: str
    node_id: str
    floor: int
    hazard_score: float
    is_mobile: bool = False
    graph_action: str

class EnvironmentStateMsg(BaseModel):
    timestamp: float
    step: int
    nodes: Dict[str, NodeStateMsg]
    evacuees: List[EvacueeMsg]
    active_threats: List[ActiveThreatMsg]
    metrics: Dict[str, float]
    # Default metrics usually include: total_people, evacuated, casualties, max_congestion, elapsed_time

class HazardUpdateMsg(BaseModel):
    node_id: str
    floor: int
    threat_type: str
    hazard_score: float
    source: str = 'perception'  # perception, injection, propagation
    timestamp: float

class PolicyActionMsg(BaseModel):
    step: int
    actions: Dict[str, Dict]
    timestamp: float

class DisasterInjectionMsg(BaseModel):
    threat_type: str
    target_node: str
    floor: int
    intensity: float = 0.8
    spread_rate: float = 0.1
    metadata: Dict[str, str] = Field(default_factory=dict)

class SimControlMsg(BaseModel):
    command: str  # play, pause, step, reset, speed
    value: Optional[float] = None
