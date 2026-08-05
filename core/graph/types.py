from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Optional, Dict

class NodeType(Enum):
    ROOM = "ROOM"
    CORRIDOR = "CORRIDOR"
    INTERSECTION = "INTERSECTION"
    STAIRWELL = "STAIRWELL"
    ELEVATOR = "ELEVATOR"
    EXIT = "EXIT"

class DoorType(Enum):
    NONE = "NONE"
    STANDARD = "STANDARD"
    FIRE_DOOR = "FIRE_DOOR"
    MAGNETIC = "MAGNETIC"

class EdgeState(Enum):
    OPEN = "OPEN"
    SEVERED = "SEVERED"
    LOCKED = "LOCKED"
    EMERGENCY_SEALED = "EMERGENCY_SEALED"

class SignDirection(Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    UP = "UP"
    DOWN = "DOWN"
    BLOCKED = "BLOCKED"
    LOCKDOWN = "LOCKDOWN"

@dataclass
class Node:
    id: str
    type: NodeType
    floor: int
    capacity: int
    position: Tuple[float, float]
    is_cross_floor: bool = False
    current_occupancy: int = 0
    hazard_score: float = 0.0

@dataclass
class Edge:
    id: str
    source: str
    target: str
    distance: float
    width: float
    max_throughput: int
    has_door: bool = False
    door_type: DoorType = DoorType.NONE
    state: EdgeState = EdgeState.OPEN
    is_vertical: bool = False
    floor_span: Optional[Tuple[int, int]] = None

@dataclass
class Floor:
    level: int
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    exits: List[str] = field(default_factory=list)

@dataclass
class Building:
    name: str
    floors: Dict[int, Floor] = field(default_factory=dict)
    cross_floor_edges: List[Edge] = field(default_factory=list)

    @property
    def total_capacity(self) -> int:
        return sum(node.capacity for floor in self.floors.values() for node in floor.nodes.values())

    @property
    def all_nodes(self) -> Dict[str, Node]:
        nodes = {}
        for floor in self.floors.values():
            nodes.update(floor.nodes)
        return nodes

    @property
    def all_edges(self) -> List[Edge]:
        edges = list(self.cross_floor_edges)
        for floor in self.floors.values():
            edges.extend(floor.edges)
        return edges

    @property
    def all_exits(self) -> List[str]:
        exits = []
        for floor in self.floors.values():
            exits.extend(floor.exits)
        return exits

    def get_node(self, node_id: str) -> Optional[Node]:
        for floor in self.floors.values():
            if node_id in floor.nodes:
                return floor.nodes[node_id]
        return None

    def get_floor_for_node(self, node_id: str) -> Optional[int]:
        for level, floor in self.floors.items():
            if node_id in floor.nodes:
                return level
        return None
