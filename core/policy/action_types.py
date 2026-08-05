from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

class EdgeAction(IntEnum):
    """Action for a standard edge/corridor."""
    ALLOW = 0
    REDIRECT = 1
    BLOCK = 2

class DoorCommand(IntEnum):
    """Command to actuate a door."""
    NO_CHANGE = 0
    OPEN = 1
    LOCK = 2
    EMERGENCY_SEAL = 3

class VerticalAction(IntEnum):
    """Action for vertical mobility nodes (stairs, elevators)."""
    ALLOW_BOTH = 0
    REDIRECT_UP = 1
    REDIRECT_DOWN = 2
    BLOCK_VERTICAL = 3

class ProtocolMode(Enum):
    """Overall protocol mode for the agent."""
    NORMAL = "NORMAL"
    EVACUATION = "EVACUATION"
    LOCKDOWN = "LOCKDOWN"
    SHELTER_IN_PLACE = "SHELTER_IN_PLACE"

@dataclass
class AgentAction:
    """Action output by the policy for a single agent (node)."""
    agent_id: str
    edge_actions: Dict[str, int]
    door_commands: Dict[str, int]
    vertical_action: Optional[int]
    protocol_mode: int

@dataclass
class GlobalAction:
    """Action output by the centralized or aggregated global policy."""
    actions: Dict[str, AgentAction]
    timestamp: float = 0.0
    step: int = 0
