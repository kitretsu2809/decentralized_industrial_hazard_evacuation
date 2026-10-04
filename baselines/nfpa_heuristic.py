"""
NFPA 101 Life Safety Code Static Heuristic Baseline.

Standard static emergency signage protocol:
Directs evacuees strictly along the static shortest geometric path to the
nearest architectural emergency exit, oblivious to evolving downstream hazard
propagation and crowd congestion.
"""

import networkx as nx
from typing import Dict, List, Set, Tuple


class NFPAStaticRouter:
    """
    Precomputes static shortest paths to the nearest exit for all nodes.
    """
    def __init__(self, G: nx.Graph, exit_indices: Set[int]):
        self.G = G
        self.exit_indices = exit_indices
        self.static_routes: Dict[int, int] = {}
        self._compute_static_routes()

    def _compute_static_routes(self):
        for node in self.G.nodes():
            if node in self.exit_indices:
                self.static_routes[node] = node
                continue

            best_exit = None
            best_dist = float('inf')
            best_next_hop = None

            for ex in self.exit_indices:
                try:
                    dist = nx.shortest_path_length(self.G, source=node, target=ex, weight='length')
                    if dist < best_dist:
                        path = nx.shortest_path(self.G, source=node, target=ex, weight='length')
                        if len(path) > 1:
                            best_dist = dist
                            best_exit = ex
                            best_next_hop = path[1]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

            if best_next_hop is not None:
                self.static_routes[node] = best_next_hop
            else:
                nbrs = list(self.G.neighbors(node))
                self.static_routes[node] = nbrs[0] if nbrs else node

    def get_actions(self, agents: List[str], neighbors_map: Dict[int, List[int]]) -> Dict[str, int]:
        actions = {}
        for agent in agents:
            node_idx = int(agent.split('_')[1])
            target_nbr = self.static_routes.get(node_idx, node_idx)
            nbrs = neighbors_map.get(node_idx, [])
            if target_nbr in nbrs:
                actions[agent] = nbrs.index(target_nbr)
            else:
                actions[agent] = 0
        return actions
