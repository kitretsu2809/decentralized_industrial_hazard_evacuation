from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

class ThreatType(Enum):
    FIRE = 1
    SMOKE = 2
    STRUCTURAL_COLLAPSE = 3
    WILDLIFE = 4
    WEAPON = 5
    UNKNOWN = 6

class GraphAction(Enum):
    DYNAMIC_REROUTE = 1
    EDGE_SEVER = 2
    MOBILE_TRACK = 3
    LOCKDOWN = 4
    NONE = 5

class ThreatSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ThreatMapping:
    threat_type: ThreatType
    default_action: GraphAction
    severity: ThreatSeverity
    base_hazard_score: float
    is_mobile: bool = False
    triggers_lockdown: bool = False
    auto_sever_edges: bool = False

THREAT_MAPPING_TABLE: Dict[ThreatType, ThreatMapping] = {
    ThreatType.FIRE: ThreatMapping(
        threat_type=ThreatType.FIRE,
        default_action=GraphAction.DYNAMIC_REROUTE,
        severity=ThreatSeverity.HIGH,
        base_hazard_score=0.7,
        is_mobile=False
    ),
    ThreatType.SMOKE: ThreatMapping(
        threat_type=ThreatType.SMOKE,
        default_action=GraphAction.DYNAMIC_REROUTE,
        severity=ThreatSeverity.MEDIUM,
        base_hazard_score=0.5,
        is_mobile=False
    ),
    ThreatType.STRUCTURAL_COLLAPSE: ThreatMapping(
        threat_type=ThreatType.STRUCTURAL_COLLAPSE,
        default_action=GraphAction.EDGE_SEVER,
        severity=ThreatSeverity.CRITICAL,
        base_hazard_score=1.0,
        auto_sever_edges=True
    ),
    ThreatType.WILDLIFE: ThreatMapping(
        threat_type=ThreatType.WILDLIFE,
        default_action=GraphAction.MOBILE_TRACK,
        severity=ThreatSeverity.MEDIUM,
        base_hazard_score=0.6,
        is_mobile=True
    ),
    ThreatType.WEAPON: ThreatMapping(
        threat_type=ThreatType.WEAPON,
        default_action=GraphAction.LOCKDOWN,
        severity=ThreatSeverity.CRITICAL,
        base_hazard_score=1.0,
        triggers_lockdown=True
    ),
    ThreatType.UNKNOWN: ThreatMapping(
        threat_type=ThreatType.UNKNOWN,
        default_action=GraphAction.NONE,
        severity=ThreatSeverity.LOW,
        base_hazard_score=0.1
    )
}

def get_threat_mapping(threat_type: ThreatType) -> ThreatMapping:
    """Get the threat mapping configuration for a given threat type."""
    return THREAT_MAPPING_TABLE.get(threat_type, THREAT_MAPPING_TABLE[ThreatType.UNKNOWN])

def classify_yolo_detection(class_name: str, confidence: float) -> Optional[ThreatType]:
    """Map YOLO class names to ThreatType based on predefined categories."""
    if confidence < 0.3:
        return None
        
    cls = class_name.lower()
    
    if cls in ('fire', 'flame'):
        return ThreatType.FIRE
    elif cls in ('smoke',):
        return ThreatType.SMOKE
    elif cls in ('knife', 'scissors', 'gun', 'weapon', 'rifle', 'pistol'):
        return ThreatType.WEAPON
    elif cls in ('cat', 'dog', 'bear', 'horse', 'bird', 'cow', 'sheep', 'elephant', 'wildlife', 'animal'):
        return ThreatType.WILDLIFE
    elif cls in ('rubble', 'collapse', 'debris'):
        return ThreatType.STRUCTURAL_COLLAPSE
        
    return None
