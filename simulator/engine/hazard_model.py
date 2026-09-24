"""
Hazard propagation model for the LBP Baseline Simulator.
Models: GAS_RELEASE, FIRE, EXPLOSION, CHEMICAL_SPILL.
Uses a discrete diffusion equation over the building graph.
"""
from __future__ import annotations
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional


class HazardType(str, Enum):
    GAS_RELEASE    = "GAS_RELEASE"
    FIRE           = "FIRE"
    EXPLOSION      = "EXPLOSION"
    CHEMICAL_SPILL = "CHEMICAL_SPILL"


# Per-type parameters
_PARAMS: Dict[HazardType, dict] = {
    HazardType.GAS_RELEASE: {
        "spread_rate": 0.18,   # hazard units/s diffused to neighbour
        "max_hops":    3,
        "decay":       0.005,  # natural decay rate if source not sustained
        "color":       "#f97316",  # orange
        "glow":        "rgba(249,115,22,",
    },
    HazardType.FIRE: {
        "spread_rate": 0.28,
        "max_hops":    2,
        "decay":       0.0,    # fire doesn't naturally decay
        "color":       "#ef4444",
        "glow":        "rgba(239,68,68,",
    },
    HazardType.EXPLOSION: {
        "spread_rate": 0.80,   # very fast
        "max_hops":    3,
        "decay":       0.01,
        "color":       "#dc2626",
        "glow":        "rgba(220,38,38,",
    },
    HazardType.CHEMICAL_SPILL: {
        "spread_rate": 0.10,
        "max_hops":    2,
        "decay":       0.003,
        "color":       "#a855f7",
        "glow":        "rgba(168,85,247,",
    },
}

BLOCKED_THRESHOLD = 0.85   # node becomes impassable
DANGER_THRESHOLD  = 0.35   # pedestrian enters DANGER state
LETHAL_THRESHOLD  = 0.90   # pedestrian becomes CASUALTY


@dataclass
class HazardSource:
    node_id: str
    htype: HazardType
    intensity: float   # initial injection level [0,1]
    sustained: bool = True  # if True, source node is pinned at intensity


class HazardModel:
    """
    Tracks hazard levels [0,1] at every node and diffuses them
    across the graph with each timestep.
    """

    def __init__(self, graph_edges: Dict[str, List[str]]):
        """
        graph_edges: adjacency dict {node_id: [neighbour_ids]}
        """
        self.adjacency: Dict[str, List[str]] = graph_edges
        self.levels: Dict[str, float] = {n: 0.0 for n in graph_edges}
        self.types: Dict[str, Optional[HazardType]] = {n: None for n in graph_edges}
        self.sources: List[HazardSource] = []
        self.blocked: Set[str] = set()
        self._hop_cache: Dict[Tuple[str, int], Set[str]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def inject(self, node_id: str, htype: HazardType, intensity: float,
               sustained: bool = True) -> None:
        if node_id not in self.levels:
            return
        self.levels[node_id] = min(1.0, intensity)
        self.types[node_id] = htype
        self.sources.append(HazardSource(node_id, htype, intensity, sustained))
        if intensity >= BLOCKED_THRESHOLD:
            self.blocked.add(node_id)
        self._hop_cache.clear()

    def step(self, dt: float) -> None:
        """Advance hazard diffusion by dt seconds."""
        new_levels = dict(self.levels)
        new_types  = dict(self.types)

        # Pin sustained sources
        for src in self.sources:
            if src.sustained:
                new_levels[src.node_id] = max(new_levels[src.node_id], src.intensity)
                new_types[src.node_id]  = src.htype

        # Diffuse from every active node to its neighbours
        for node, level in self.levels.items():
            if level < 0.01:
                continue
            htype = self.types[node]
            if htype is None:
                continue
            params    = _PARAMS[htype]
            max_hops  = params["max_hops"]
            rate      = params["spread_rate"]

            neighbours_in_range = self._get_nodes_within_hops(node, max_hops)
            for neighbour in neighbours_in_range:
                if neighbour == node:
                    continue
                hop_dist = self._hop_distance(node, neighbour)
                attenuation = math.exp(-0.5 * (hop_dist - 1))   # decays with distance
                delta = rate * level * dt * attenuation * (1.0 - new_levels[neighbour])
                if delta > 0.001:
                    new_levels[neighbour] = min(1.0, new_levels[neighbour] + delta)
                    if new_types[neighbour] is None:
                        new_types[neighbour] = htype

        # Natural decay (non-source nodes)
        source_nodes = {s.node_id for s in self.sources if s.sustained}
        for node in new_levels:
            if node not in source_nodes and new_levels[node] > 0:
                htype = new_types[node]
                if htype:
                    decay = _PARAMS[htype]["decay"] * dt
                    new_levels[node] = max(0.0, new_levels[node] - decay)

        self.levels = new_levels
        self.types  = new_types

        # Update blocked set
        self.blocked = {n for n, v in self.levels.items() if v >= BLOCKED_THRESHOLD}

    def get_color(self, node_id: str) -> Optional[str]:
        """Returns hex color for a hazard, or None if safe."""
        lvl   = self.levels.get(node_id, 0.0)
        htype = self.types.get(node_id)
        if lvl < 0.05 or htype is None:
            return None
        return _PARAMS[htype]["color"]

    def get_glow(self, node_id: str) -> Optional[str]:
        """Returns rgba prefix for canvas glow effect."""
        htype = self.types.get(node_id)
        if htype is None:
            return None
        return _PARAMS[htype]["glow"]

    def to_dict(self) -> dict:
        return {
            node: {
                "level": round(lvl, 3),
                "type":  self.types[node].value if self.types[node] else None,
                "color": self.get_color(node),
                "blocked": node in self.blocked,
            }
            for node, lvl in self.levels.items()
            if lvl > 0.01
        }

    def reset(self) -> None:
        self.levels  = {n: 0.0 for n in self.adjacency}
        self.types   = {n: None for n in self.adjacency}
        self.sources = []
        self.blocked = set()
        self._hop_cache.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_nodes_within_hops(self, start: str, max_hops: int) -> Set[str]:
        key = (start, max_hops)
        if key in self._hop_cache:
            return self._hop_cache[key]
        visited, frontier = {start}, {start}
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for nb in self.adjacency.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        self._hop_cache[key] = visited
        return visited

    def _hop_distance(self, a: str, b: str) -> int:
        """BFS hop count between nodes (slow — only used for startup caching)."""
        if a == b:
            return 0
        visited, frontier, dist = {a}, {a}, 0
        while frontier:
            dist += 1
            next_frontier = set()
            for node in frontier:
                for nb in self.adjacency.get(node, []):
                    if nb == b:
                        return dist
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        return 999
