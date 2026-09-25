"""
Pedestrian agent for the Industrial Evacuation Simulator.
Each agent has a position in continuous 2D space, a velocity,
a state, and a current Dijkstra path (list of node IDs).
"""
from __future__ import annotations
import math
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class PedestrianState(str, Enum):
    NORMAL    = "normal"    # On duty / dwelling at workstation
    REACTING  = "reacting"  # Alarm recognized, hesitation / reaction delay (ISO 16738)
    MOVING    = "moving"    # Evacuating along safe Dijkstra path
    REROUTING = "rerouting" # Blocked path, recalculating
    DANGER    = "danger"    # In a hazardous zone
    EVACUATED = "evacuated" # Reached an exit
    CASUALTY  = "casualty"  # Trapped in lethal hazard


@dataclass
class Pedestrian:
    id: int
    x: float                     # metres, continuous space
    y: float
    floor: int = 1
    vx: float = 0.0              # velocity m/s
    vy: float = 0.0
    state: PedestrianState = PedestrianState.NORMAL
    path: List[str] = field(default_factory=list)   # remaining node waypoints
    current_node: str = ""       # node the agent is currently at / last passed
    target_node: str = ""        # immediate next node
    desired_speed: float = 1.34  # free-flow egress speed m/s (Weidmann 1992)
    reroute_cooldown: float = 0.0
    age: float = 0.0             # simulation time this agent has existed

    # Pre-evacuation & anchor tracking
    anchor_x: float = 0.0        # workstation anchor x
    anchor_y: float = 0.0        # workstation anchor y
    reaction_delay: float = 0.0  # pre-movement delay seconds (ISO/TR 16738)
    wander_target_x: float = 0.0 # micro-patrol target
    wander_target_y: float = 0.0
    wander_timer: float = 0.0

    # ── Rendering helpers ────────────────────────────────────────────────────
    def color(self) -> str:
        return {
            PedestrianState.NORMAL:    "#38bdf8",   # calm cyan/blue (working at station)
            PedestrianState.REACTING:  "#f59e0b",   # amber (alerted, hesitating)
            PedestrianState.MOVING:    "#22c55e",   # green (evacuating)
            PedestrianState.REROUTING: "#eab308",   # yellow (rerouting)
            PedestrianState.DANGER:    "#ef4444",   # red (in hazard)
            PedestrianState.EVACUATED: "#94a3b8",   # slate (fade out at exit)
            PedestrianState.CASUALTY:  "#1e293b",   # dark (overcome)
        }[self.state]

    def to_dict(self) -> dict:
        return {
            "id":    self.id,
            "x":     round(self.x, 2),
            "y":     round(self.y, 2),
            "floor": self.floor,
            "vx":    round(self.vx, 2),
            "vy":    round(self.vy, 2),
            "state": self.state.value,
            "color": self.color(),
            "path":  self.path[:3],   # only next 3 waypoints for bandwidth
        }


def spawn_pedestrians(n: int, node_positions: dict, node_floors: dict,
                      exclude_types: set = None) -> List[Pedestrian]:
    """
    Spawn n pedestrians distributed realistically across all floors in the facility
    (excluding muster exits and stairwells).
    """
    eligible = [
        nid for nid in node_positions
        if not any(t.lower() in nid.lower() for t in ("muster", "helipad", "slide_escape"))
        and not any(t.lower() in nid.lower() for t in ("stair", "hoist", "elevator"))
        and not any(t.lower() in nid.lower() for t in ("corridor", "perimeter", "catwalk", "walkway"))
    ]
    if not eligible:
        eligible = [
            nid for nid in node_positions
            if not any(t.lower() in nid.lower() for t in ("muster", "helipad", "slide_escape"))
        ]

    agents = []
    for i in range(n):
        node = random.choice(eligible)
        x, y = node_positions[node]
        fl = node_floors.get(node, 1)

        # Gentle scatter within workstation room (radius <= 1.2m)
        r = random.uniform(0.2, 1.2)
        ang = random.uniform(0, 2 * math.pi)
        px = x + r * math.cos(ang)
        py = y + r * math.sin(ang)

        agents.append(Pedestrian(
            id=i, x=px, y=py,
            floor=fl,
            current_node=node,
            state=PedestrianState.NORMAL,
            desired_speed=max(0.9, random.gauss(1.34, 0.20)),  # Weidmann distribution
            anchor_x=px,
            anchor_y=py,
            wander_target_x=px + random.uniform(-0.6, 0.6),
            wander_target_y=py + random.uniform(-0.6, 0.6),
            reaction_delay=random.uniform(1.2, 3.5),  # ISO/TR 16738 recognition delay
        ))
    return agents
