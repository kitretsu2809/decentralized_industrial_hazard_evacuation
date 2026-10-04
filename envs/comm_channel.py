"""
Stochastic Degraded Wireless Communication Channel.

Simulates real-world industrial RF propagation degradation during disaster scenarios:
1. Packet Dropout: Random message loss p_loss in [0.0, 0.8]
2. Range Constraints: Transmission radius limit R_comm <= 18.0m
3. Thermal / Physical Cutoff: Link failure if local hazard C >= C_critical
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple


class DegradedCommChannel:
    """
    Applies transmission range attenuation, stochastic packet drop, and
    thermal damage severing to agent-to-agent and node-to-node communication links.
    """

    def __init__(
        self,
        comm_radius: float = 18.0,
        packet_loss_rate: float = 0.15,
        critical_hazard_cutoff: float = 0.60,
        seed: Optional[int] = None
    ):
        self.comm_radius = float(comm_radius)
        self.p_loss = float(packet_loss_rate)
        self.critical_hazard = float(critical_hazard_cutoff)
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def set_packet_loss_rate(self, p_loss: float):
        self.p_loss = float(np.clip(p_loss, 0.0, 1.0))

    def filter_active_links(
        self,
        positions: Dict[int, Tuple[float, float, float]],
        hazards: np.ndarray,
        base_edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        active_links = []
        for u, v in base_edges:
            # 1. Thermal cutoff check
            if u < len(hazards) and hazards[u] >= self.critical_hazard:
                continue
            if v < len(hazards) and hazards[v] >= self.critical_hazard:
                continue

            # 2. Geometric communication radius check
            if u in positions and v in positions:
                p_u = np.array(positions[u], dtype=np.float64)
                p_v = np.array(positions[v], dtype=np.float64)
                dist = np.linalg.norm(p_u - p_v)
                if dist > self.comm_radius:
                    continue

            # 3. Stochastic packet drop
            if self.rng.random() < self.p_loss:
                continue

            active_links.append((u, v))

        return active_links

    def mask_adjacency_matrix(
        self,
        positions: Dict[int, Tuple[float, float, float]],
        hazards: np.ndarray,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        n = adj_matrix.shape[0]
        mask = np.zeros_like(adj_matrix, dtype=np.float32)

        for u in range(n):
            if u < len(hazards) and hazards[u] >= self.critical_hazard:
                continue
            for v in range(n):
                if adj_matrix[u, v] == 0:
                    continue
                if v < len(hazards) and hazards[v] >= self.critical_hazard:
                    continue

                if u in positions and v in positions:
                    p_u = np.array(positions[u], dtype=np.float64)
                    p_v = np.array(positions[v], dtype=np.float64)
                    if np.linalg.norm(p_u - p_v) > self.comm_radius:
                        continue

                if self.rng.random() >= self.p_loss:
                    mask[u, v] = 1.0

        return mask
