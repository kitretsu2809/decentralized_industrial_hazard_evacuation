"""
Dynamic D* Lite Hazard-Aware Re-routing Baseline.

Implements dynamic replanning with real-time hazard-weighted edge costs:
    cost(u, v) = L(u, v) * (1.0 + beta * max(C_u, C_v)^1.5)
If an edge has max(C_u, C_v) >= critical_cutoff, cost = inf.
Uses exact multi-source weighted Dijkstra backwards from all safe exits.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Set, Tuple


class DStarLiteRouter:
    def __init__(
        self,
        G: nx.Graph,
        exit_indices: Set[int],
        hazard_weight: float = 25.0,
        critical_cutoff: float = 0.60
    ):
        self.G = G
        self.exit_indices = exit_indices
        self.hazard_weight = hazard_weight
        self.critical_cutoff = critical_cutoff

        self.edge_base_len: Dict[Tuple[int, int], float] = {}
        for u, v, data in G.edges(data=True):
            L = float(data.get('length', 10.0))
            self.edge_base_len[(u, v)] = L
            self.edge_base_len[(v, u)] = L

    def update_and_route(
        self,
        hazard_conc: np.ndarray,
        agents: List[str],
        neighbors_map: Dict[int, List[int]]
    ) -> Dict[str, int]:
        G_dyn = nx.DiGraph()

        # Build dynamic cost graph
        for (u, v), L in self.edge_base_len.items():
            c_u = float(hazard_conc[u]) if u < len(hazard_conc) else 0.0
            c_v = float(hazard_conc[v]) if v < len(hazard_conc) else 0.0
            max_c = max(c_u, c_v)

            if max_c >= self.critical_cutoff:
                cost = 1e8
            else:
                cost = L * (1.0 + self.hazard_weight * (max_c ** 1.5))

            G_dyn.add_edge(u, v, weight=cost)

        # Exact multi-source weighted Dijkstra from all exits on the reversed graph
        try:
            G_rev = G_dyn.reverse()
            dist_to_exit = nx.multi_source_dijkstra_path_length(
                G_rev, sources=list(self.exit_indices), weight='weight'
            )
        except Exception:
            dist_to_exit = {}

        actions = {}
        for agent in agents:
            node_idx = int(agent.split('_')[1])
            nbrs = neighbors_map.get(node_idx, [])
            if not nbrs:
                actions[agent] = 0
                continue

            if node_idx in self.exit_indices:
                actions[agent] = 0
                continue

            best_nbr = nbrs[0]
            best_cost = float('inf')

            for nbr in nbrs:
                c_u = float(hazard_conc[node_idx]) if node_idx < len(hazard_conc) else 0.0
                c_nbr = float(hazard_conc[nbr]) if nbr < len(hazard_conc) else 0.0
                max_c = max(c_u, c_nbr)
                L = self.edge_base_len.get((node_idx, nbr), 10.0)
                step_cost = 1e8 if max_c >= self.critical_cutoff else L * (1.0 + self.hazard_weight * (max_c ** 1.5))

                total_cost = step_cost + dist_to_exit.get(nbr, 1e8)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_nbr = nbr

            actions[agent] = nbrs.index(best_nbr)

        return actions
