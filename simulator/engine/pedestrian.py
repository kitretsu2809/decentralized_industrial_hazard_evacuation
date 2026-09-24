"""
Pedestrian agent for the LBP Baseline Simulator.
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
    MOVING    = "moving"    # Following Dijkstra path safely
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
    state: PedestrianState = PedestrianState.MOVING
    path: List[str] = field(default_factory=list)   # remaining node waypoints
    current_node: str = ""       # node the agent is currently at / last passed
    target_node: str = ""        # immediate next node
    desired_speed: float = 1.4   # free-flow speed m/s  (Weidmann 1992)
    reroute_cooldown: float = 0.0
    age: float = 0.0             # simulation time this agent has existed

    # ── Rendering helpers ────────────────────────────────────────────────────
    def color(self) -> str:
        return {
            PedestrianState.MOVING:    "#22c55e",   # green
            PedestrianState.REROUTING: "#eab308",   # yellow
            PedestrianState.DANGER:    "#ef4444",   # red
            PedestrianState.EVACUATED: "#94a3b8",   # slate (fades out)
            PedestrianState.CASUALTY:  "#1e293b",   # dark (dead)
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
    Spawn n pedestrians randomly across nodes (skipping exits by default).
    node_positions: {node_id: (x, y)}
    node_floors:    {node_id: floor_int}
    """
    if exclude_types is None:
        exclude_types = {"EXIT", "STAIRWELL"}

    eligible = [
        nid for nid in node_positions
        if node_floors.get(nid) in (1,) and  # start on floor 1 for simplicity
        not any(t.lower() in nid.lower() for t in ("muster", "helipad", "slide_escape"))
    ]
    if not eligible:
        eligible = list(node_positions.keys())

    agents = []
    for i in range(n):
        node = random.choice(eligible)
        x, y = node_positions[node]
        # Scatter slightly within the node area (±2 m)
        x += random.uniform(-2.0, 2.0)
        y += random.uniform(-2.0, 2.0)
        agents.append(Pedestrian(
            id=i, x=x, y=y,
            floor=node_floors.get(node, 1),
            current_node=node,
            desired_speed=random.gauss(1.34, 0.26),  # Weidmann distribution
        ))
    return agents
