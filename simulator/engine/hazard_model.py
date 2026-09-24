"""
Hazard propagation — distance-weighted diffusion on the building graph.
Spread uses actual edge distances: closer neighbours receive more hazard.
"""
from __future__ import annotations
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional


class HazardType(str, Enum):
    GAS_RELEASE    = "GAS_RELEASE"
    FIRE           = "FIRE"
    EXPLOSION      = "EXPLOSION"
    CHEMICAL_SPILL = "CHEMICAL_SPILL"

# Spread rates are per-second to an immediate neighbour at 0 m (attenuated by distance)
_PARAMS = {
    HazardType.GAS_RELEASE:    {"rate": 0.06,  "decay": 0.003, "color": "#f97316", "rgb": (249,115,22)},
    HazardType.FIRE:           {"rate": 0.09,  "decay": 0.001, "color": "#ef4444", "rgb": (239,68,68)},
    HazardType.EXPLOSION:      {"rate": 0.35,  "decay": 0.008, "color": "#dc2626", "rgb": (220,38,38)},
    HazardType.CHEMICAL_SPILL: {"rate": 0.04,  "decay": 0.002, "color": "#a855f7", "rgb": (168,85,247)},
}

# Characteristic distance for attenuation: exp(-d / CHAR_DIST)
# At 20m: factor = exp(-20/40) = 0.61  — still significant
# At 60m: factor = exp(-60/40) = 0.22  — minimal
CHAR_DIST        = 40.0   # metres
BLOCKED_THRESHOLD = 0.80
DANGER_THRESHOLD  = 0.30
LETHAL_THRESHOLD  = 0.88


@dataclass
class HazardSource:
    node_id: str
    htype: HazardType
    intensity: float
    sustained: bool = True


class HazardModel:
    """
    adjacency: {node_id: [(neighbour_id, distance_m), ...]}
    """
    def __init__(self, adjacency: Dict[str, List[Tuple[str, float]]]):
        self.adjacency = adjacency
        self.levels: Dict[str, float]             = {n: 0.0 for n in adjacency}
        self.types:  Dict[str, Optional[HazardType]] = {n: None for n in adjacency}
        self.sources: List[HazardSource] = []
        self.blocked: Set[str] = set()

    def inject(self, node_id: str, htype: HazardType, intensity: float,
               sustained: bool = True) -> None:
        if node_id not in self.levels: return
        self.levels[node_id] = min(1.0, intensity)
        self.types[node_id]  = htype
        # Replace existing source for same node
        self.sources = [s for s in self.sources if s.node_id != node_id]
        self.sources.append(HazardSource(node_id, htype, intensity, sustained))
        if intensity >= BLOCKED_THRESHOLD:
            self.blocked.add(node_id)

    def step(self, dt: float) -> None:
        new_levels = dict(self.levels)

        # Pin sustained sources
        for src in self.sources:
            if src.sustained:
                new_levels[src.node_id] = max(new_levels[src.node_id], src.intensity)
                if self.types[src.node_id] is None:
                    self.types[src.node_id] = src.htype

        # Diffuse to immediate neighbours only (multi-hop happens naturally over time)
        for node, level in self.levels.items():
            if level < 0.01: continue
            htype = self.types[node]
            if htype is None: continue
            rate = _PARAMS[htype]["rate"]

            for neighbour, edge_dist in self.adjacency.get(node, []):
                # Distance attenuation: exp(-d / CHAR_DIST)
                attenuation = math.exp(-edge_dist / CHAR_DIST)
                delta = rate * level * dt * attenuation * (1.0 - new_levels[neighbour])
                if delta > 0.0001:
                    new_levels[neighbour] = min(1.0, new_levels[neighbour] + delta)
                    if self.types.get(neighbour) is None:
                        self.types[neighbour] = htype

        # Natural decay (non-source nodes)
        source_nodes = {s.node_id for s in self.sources if s.sustained}
        for node in new_levels:
            if node not in source_nodes and new_levels[node] > 0:
                htype = self.types.get(node)
                if htype:
                    decay = _PARAMS[htype]["decay"] * dt
                    new_levels[node] = max(0.0, new_levels[node] - decay)

        self.levels = new_levels
        self.blocked = {n for n, v in self.levels.items() if v >= BLOCKED_THRESHOLD}

    def get_color(self, node_id):
        lvl  = self.levels.get(node_id, 0.0)
        htype = self.types.get(node_id)
        return _PARAMS[htype]["color"] if lvl >= 0.05 and htype else None

    def to_dict(self):
        out = {}
        for node, lvl in self.levels.items():
            if lvl > 0.02:
                htype = self.types[node]
                out[node] = {
                    "level":   round(lvl, 3),
                    "type":    htype.value if htype else None,
                    "color":   self.get_color(node),
                    "rgb":     list(_PARAMS[htype]["rgb"]) if htype else None,
                    "blocked": node in self.blocked,
                }
        return out

    def reset(self):
        self.levels  = {n: 0.0 for n in self.adjacency}
        self.types   = {n: None for n in self.adjacency}
        self.sources = []
        self.blocked = set()
