"""
Hazard Mechanics: Graph Laplacian Advection-Diffusion & ISO 13571 FED Toxicity.

Implements:
1. Discrete Graph Laplacian toxic gas dispersion:
       C_{t+1} = C_t + D * L * C_t - k * C_t + S_t
   with exit node zero-accumulation boundary condition (positive-pressure outdoor havens).
2. ISO 13571:2012 Fractional Effective Dose (FED) Model:
       FED_i(t) = sum_{tau=0}^t (C_tox(tau) / FED_threshold) * Delta t
   Casualty condition:
       is_toxic_casualty = (FED_i >= 1.0)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Union


class HazardSimulator:
    """
    Simulates spatiotemporal gas/smoke dispersion over an arbitrary graph topology
    and evaluates physiological incapacitation via ISO 13571 FED.
    """

    def __init__(
        self,
        num_nodes: int,
        adj_matrix: Optional[np.ndarray] = None,
        diffusion_coeff: float = 0.15,
        decay_rate: float = 0.01,
        dt: float = 1.0,
        fed_threshold: float = 60.0,
        critical_hazard_cutoff: float = 0.60,
        exit_indices: Optional[Set[int]] = None
    ):
        self.num_nodes = num_nodes
        self.D = float(diffusion_coeff)
        self.k = float(decay_rate)
        self.dt = float(dt)
        self.fed_threshold = float(fed_threshold)
        self.critical_hazard_cutoff = float(critical_hazard_cutoff)
        self.exit_indices: Set[int] = set(exit_indices or [])

        # Node hazard concentrations C in [0.0, 1.0]
        self.concentration = np.zeros(num_nodes, dtype=np.float64)
        # Source injection rates S_t (added each step)
        self.sources: Dict[int, float] = {}

        if adj_matrix is not None:
            self.set_topology(adj_matrix)
        else:
            self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            self.laplacian_op = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    def set_topology(self, adj_matrix: np.ndarray):
        self.adj_matrix = np.asarray(adj_matrix, dtype=np.float64)
        n = self.num_nodes
        deg = np.sum(self.adj_matrix, axis=1)

        # Discrete diffusion operator L_diff = (A - diag(deg))
        max_deg = np.max(deg) if np.max(deg) > 0 else 1.0
        cfl_factor = self.D * self.dt * max_deg
        if cfl_factor > 0.8:
            effective_d = 0.8 / (self.dt * max_deg)
        else:
            effective_d = self.D

        L_comb = self.adj_matrix - np.diag(deg)
        self.laplacian_op = effective_d * L_comb

    def set_source(self, node_idx: int, injection_rate: float):
        self.sources[node_idx] = float(injection_rate)

    def clear_sources(self):
        self.sources.clear()

    def reset(self):
        self.concentration.fill(0.0)
        self.sources.clear()

    def step(self) -> np.ndarray:
        """
        Advances hazard diffusion by one timestep Delta t:
            C_{t+1} = C_t + dt * (D * L * C_t - k * C_t + S_t)
        """
        diffusion_flux = self.laplacian_op @ self.concentration
        decay_flux = -self.k * self.concentration

        source_flux = np.zeros_like(self.concentration)
        for node_idx, rate in self.sources.items():
            if 0 <= node_idx < self.num_nodes and node_idx not in self.exit_indices:
                source_flux[node_idx] += rate

        delta_c = (diffusion_flux + decay_flux + source_flux) * self.dt
        self.concentration = np.maximum(0.0, np.minimum(1.0, self.concentration + delta_c))

        # Exterior muster assembly points do not accumulate indoor plume (NFPA 92)
        for ex in self.exit_indices:
            self.concentration[ex] = 0.0

        return self.concentration.copy()

    def compute_fed_delta(self, exposure_concentration: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes ISO 13571 incremental Fractional Effective Dose:
            Delta FED = (C_tox / FED_threshold) * Delta t
        """
        delta_fed = (np.asarray(exposure_concentration, dtype=np.float64) / self.fed_threshold) * self.dt
        return delta_fed

    def is_severed(self, node_idx: int) -> bool:
        if 0 <= node_idx < self.num_nodes:
            return bool(self.concentration[node_idx] >= self.critical_hazard_cutoff)
        return False
