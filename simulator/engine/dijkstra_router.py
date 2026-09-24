"""
Dijkstra-based router for the LBP Baseline Simulator.
Uses hazard-weighted edge costs so routes avoid dangerous zones.
cost(edge) = distance * (1 + 10 * max_hazard_on_endpoints)
"""
from __future__ import annotations
import math
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple


class DijkstraRouter:
    """
    Wraps a NetworkX graph and computes shortest safe paths to exits.
    The graph is rebuilt whenever hazard levels change significantly.
    """

    def __init__(self):
        self.G: Optional[nx.Graph] = None
        self.exits: List[str]      = []
        self.node_floors: Dict[str, int] = {}
        self._last_hazard_snapshot: Dict[str, float] = {}

    # ── Graph construction ────────────────────────────────────────────────────

    def build(self, edges: List[Tuple], exits: List[str],
              node_floors: Dict[str, int]) -> None:
        """
        edges: list of (src, tgt, distance_m, width_m)
        """
        self.G = nx.Graph()
        for src, tgt, dist, *_ in edges:
            self.G.add_edge(src, tgt, distance=dist, weight=dist)
        self.exits      = exits
        self.node_floors = node_floors

    def update_weights(self, hazard_levels: Dict[str, float],
                       blocked: Set[str]) -> bool:
        """
        Recompute edge weights based on current hazard.
        Returns True if any weight changed significantly (triggers reroute).
        """
        if self.G is None:
            return False

        changed = False
        for u, v, data in self.G.edges(data=True):
            h = max(hazard_levels.get(u, 0.0), hazard_levels.get(v, 0.0))
            if u in blocked or v in blocked:
                new_w = 1e9   # effectively impassable
            else:
                new_w = data["distance"] * (1.0 + 10.0 * h)

            prev = self._last_hazard_snapshot.get(f"{u}_{v}", -1)
            if abs(new_w - prev) > 1.0:
                changed = True
                self._last_hazard_snapshot[f"{u}_{v}"] = new_w
            self.G[u][v]["weight"] = new_w

        return changed

    # ── Path computation ──────────────────────────────────────────────────────

    def get_path(self, source: str, floor: int) -> List[str]:
        """
        Returns the optimal path from source to the nearest reachable exit
        on the same floor (or via stairs). Returns [] if unreachable.
        """
        if self.G is None or source not in self.G:
            return []

        best_path: List[str] = []
        best_cost = math.inf

        candidate_exits = [e for e in self.exits]   # all exits across all floors
        for exit_node in candidate_exits:
            if exit_node not in self.G:
                continue
            try:
                path = nx.dijkstra_path(self.G, source, exit_node, weight="weight")
                cost = nx.dijkstra_path_length(self.G, source, exit_node, weight="weight")
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        return best_path

    def path_cost(self, path: List[str]) -> float:
        if self.G is None or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            if self.G.has_edge(path[i], path[i + 1]):
                total += self.G[path[i]][path[i + 1]]["weight"]
            else:
                total += 1e9
        return total

    def path_has_hazard(self, path: List[str],
                        hazard_levels: Dict[str, float],
                        threshold: float = 0.5) -> bool:
        return any(hazard_levels.get(n, 0.0) > threshold for n in path)

    def is_blocked_path(self, path: List[str], blocked: Set[str]) -> bool:
        return any(n in blocked for n in path)
