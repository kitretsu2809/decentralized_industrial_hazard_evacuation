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

HAZARD_ALIASES: Dict[str, HazardType] = {
    "THERMAL_FIRE": HazardType.FIRE,
    "FIRE": HazardType.FIRE,
    "GAS_RELEASE": HazardType.GAS_RELEASE,
    "TOXIC_PLUME": HazardType.GAS_RELEASE,
    "EXPLOSION": HazardType.EXPLOSION,
    "CHEMICAL_SPILL": HazardType.CHEMICAL_SPILL,
}

# Spread rates are per-second to an immediate neighbour at 0 m (attenuated by distance)
_PARAMS = {
    HazardType.GAS_RELEASE:    {"rate": 0.05,  "decay": 0.003, "color": "#f97316", "rgb": (249,115,22)},
    HazardType.FIRE:           {"rate": 0.06,  "decay": 0.001, "color": "#ef4444", "rgb": (239,68,68)},
    HazardType.EXPLOSION:      {"rate": 0.25,  "decay": 0.008, "color": "#dc2626", "rgb": (220,38,38)},
    HazardType.CHEMICAL_SPILL: {"rate": 0.04,  "decay": 0.002, "color": "#a855f7", "rgb": (168,85,247)},
}

# Characteristic distance for attenuation: exp(-d / CHAR_DIST)
CHAR_DIST        = 35.0   # metres
BLOCKED_THRESHOLD = 0.80
DANGER_THRESHOLD  = 0.30
LETHAL_THRESHOLD  = 0.80


@dataclass
class HazardSource:
    node_id: str
    htype: HazardType
    intensity: float
    sustained: bool = True


class HazardModel:
    """
    Graph Laplacian Advective-Diffusive Hazard Propagation with Vertical Buoyancy
    and Strict Conservation of Mass.
    """
    def __init__(
        self,
        adjacency: Dict[str, List[Tuple[str, float]]],
        exits: Optional[List[str]] = None,
        node_floors: Optional[Dict[str, int]] = None,
    ):
        self.adjacency = adjacency
        self.exits: Set[str] = set(exits or [])
        self.node_floors: Dict[str, int] = node_floors or {n: 1 for n in adjacency}
        self.levels: Dict[str, float] = {n: 0.0 for n in adjacency}
        self.types: Dict[str, Optional[HazardType]] = {n: None for n in adjacency}
        self.sources: List[HazardSource] = []
        self.blocked: Set[str] = set()

        # Cache unique undirected edges: (u, v, dist)
        seen_edges = set()
        self.unique_edges: List[Tuple[str, str, float]] = []
        for u, nbrs in self.adjacency.items():
            for v, dist in nbrs:
                edge_key = tuple(sorted([u, v]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    self.unique_edges.append((edge_key[0], edge_key[1], float(dist)))

    def inject(self, node_id: str, htype: Any, intensity: float,
               sustained: bool = True) -> None:
        if node_id not in self.levels: return
        if isinstance(htype, str):
            htype = HAZARD_ALIASES.get(htype.upper(), HazardType.GAS_RELEASE)
        self.levels[node_id] = min(1.0, intensity)
        self.types[node_id]  = htype
        # Replace existing source for same node
        self.sources = [s for s in self.sources if s.node_id != node_id]
        self.sources.append(HazardSource(node_id, htype, intensity, sustained))
        if intensity >= BLOCKED_THRESHOLD:
            self.blocked.add(node_id)

    def step(self, dt: float) -> None:
        delta_c: Dict[str, float] = {n: 0.0 for n in self.levels}

        # 1. Compute conservative Laplacian flux across all unique edges
        for u, v, edge_dist in self.unique_edges:
            c_u = self.levels[u]
            c_v = self.levels[v]

            if abs(c_u - c_v) < 1e-4:
                continue

            # Identify driving source hazard
            if c_u > c_v:
                src_node, dst_node = u, v
                c_high, c_low = c_u, c_v
            else:
                src_node, dst_node = v, u
                c_high, c_low = c_v, c_u

            htype = self.types.get(src_node) or HazardType.GAS_RELEASE
            rate = _PARAMS[htype]["rate"]
            attenuation = math.exp(-edge_dist / CHAR_DIST)

            # Vertical buoyancy stack effect
            floor_src = self.node_floors.get(src_node, 1)
            floor_dst = self.node_floors.get(dst_node, 1)
            delta_floor = floor_dst - floor_src

            buoyancy = 1.0
            if htype in (HazardType.FIRE, HazardType.GAS_RELEASE):
                if delta_floor > 0:
                    buoyancy = 2.5   # Hot smoke / fire rushes UP stairwells
                elif delta_floor < 0:
                    buoyancy = 0.40  # Downward smoke penetration is resisted
            elif htype == HazardType.CHEMICAL_SPILL:
                if delta_floor < 0:
                    buoyancy = 2.0   # Dense chemical vapors pool DOWNWARD
                elif delta_floor > 0:
                    buoyancy = 0.20  # Dense liquid/vapors resist rising

            conductivity = rate * attenuation * buoyancy
            flux = conductivity * (c_high - c_low) * dt

            # CFL condition: prevent gradient inversion
            max_flux = 0.40 * (c_high - c_low)
            flux = min(flux, max_flux)

            # Apply conservative mass transfer
            delta_c[src_node] -= flux
            delta_c[dst_node] += flux

            if c_low + flux > 0.02 and self.types.get(dst_node) is None:
                self.types[dst_node] = htype

        # 2. Update levels and enforce boundaries
        new_levels = {}
        sustained_nodes = {s.node_id: s for s in self.sources if s.sustained}

        for n, level in self.levels.items():
            if n in self.exits:
                # Exterior muster zones remain clean (positive-pressure atmospheric dispersion)
                new_levels[n] = 0.0
                continue

            if n in sustained_nodes:
                src = sustained_nodes[n]
                new_levels[n] = max(src.intensity, level + delta_c[n])
                if self.types[n] is None:
                    self.types[n] = src.htype
            else:
                htype = self.types.get(n)
                decay = (_PARAMS[htype]["decay"] * dt) if htype else 0.0
                updated = level + delta_c[n] - decay
                new_levels[n] = max(0.0, min(1.0, updated))

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
