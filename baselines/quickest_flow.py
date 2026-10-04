"""
Dynamic Quickest Flow Evacuation Baseline.

Implements a dynamic network flow optimization baseline that accounts for:
1. Physical corridor transit time: t_transit = L / v(rho)
2. Bottleneck constriction delays: t_wait = excess_queue / Q_max
3. Real-time toxic gas safety penalties.
Calculates exact arrival time to exits using weighted reverse multi-source Dijkstra.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Set, Tuple


class QuickestFlowRouter:
    def __init__(
        self,
        G: nx.Graph,
        exit_indices: Set[int],
        edge_info: Dict[Tuple[int, int], Dict[str, float]],
        q_specific: float = 1.33
    ):
        self.G = G
        self.exit_indices = exit_indices
        self.edge_info = edge_info
        self.q_specific = q_specific

    def get_actions(
        self,
        agents: List[str],
        neighbors_map: Dict[int, List[int]],
        hazard_conc: np.ndarray,
        node_occupancies: np.ndarray
    ) -> Dict[str, int]:
        G_flow = nx.DiGraph()

        # Build dynamic cost graph
        for (u, v), info in self.edge_info.items():
            L = info.get('length', 10.0)
            W = info.get('width', 2.0)
            cap = info.get('throughput', 10.0)

            # Bottleneck flow cap
            q_max = max(0.1, min(cap, W * self.q_specific))

            # Queue delay estimation based on local node occupancy
            occ = float(node_occupancies[u]) if u < len(node_occupancies) else 0.0
            t_queue = occ / q_max

            # Free flow travel time
            t_transit = L / 1.34

            # Hazard penalty
            c_u = float(hazard_conc[u]) if u < len(hazard_conc) else 0.0
            c_v = float(hazard_conc[v]) if v < len(hazard_conc) else 0.0
            max_c = max(c_u, c_v)

            if max_c >= 0.60:
                cost = 1e8
            else:
                hazard_mult = 1.0 + 30.0 * (max_c ** 2)
                cost = (t_transit + t_queue) * hazard_mult

            G_flow.add_edge(u, v, weight=cost)

        # Exact multi-source weighted Dijkstra backwards from all exits
        try:
            G_rev = G_flow.reverse()
            arrival_times = nx.multi_source_dijkstra_path_length(
                G_rev, sources=list(self.exit_indices), weight='weight'
            )
        except Exception:
            arrival_times = {}

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
            best_time = float('inf')

            for nbr in nbrs:
                edge_cost = G_flow.get_edge_data(node_idx, nbr, {}).get('weight', 10.0)
                tot_time = edge_cost + arrival_times.get(nbr, 1e8)
                if tot_time < best_time:
                    best_time = tot_time
                    best_nbr = nbr

            actions[agent] = nbrs.index(best_nbr)

        return actions
