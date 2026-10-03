"""
Continuous Velocity-Density Fundamental Diagram & Non-Linear Bottleneck Arching.

Implements:
1. Weidmann empirical velocity-density continuous curve:
       v(rho) = v_0 * (1 - exp(-gamma * (1/rho - 1/rho_jam)))
   v_0 = 1.34 m/s, rho_jam = 5.4 persons/m^2, gamma = 1.913

2. Helbing Bottleneck Doorway Arching Breakdown (Phase 1 Mandate):
   At constrictions (doorways, stairwell entries), when density exceeds
   critical threshold rho_crit = 3.5 persons/m^2, physical arching forces
   drop outflow non-linearly:
       Q(t) = Q_nominal * max(0.10, 1.0 - beta * (rho_u(t) - rho_crit)^2)
   with beta = 0.25, capping outflow to as low as 10% nominal and reducing
   effective speed toward 0.05 m/s.

3. Compressive Crowd Asphyxia (Crush Stress Index - CSI):
       CSI_i(t) = integral max(0, rho_local(tau) - 4.5) dtau
   Fatal threshold: CSI_fatal = 15.0 s * ped/m^2.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple


@dataclass
class WeidmannParams:
    v0: float = 1.34           # Free flow speed (m/s)
    rho_jam: float = 5.4       # Jam density (persons/m^2)
    gamma: float = 1.913       # Calibration parameter
    q_specific: float = 1.33   # Specific flow rate (persons / (m * s))
    min_velocity: float = 0.05 # Minimum creep velocity (m/s)
    rho_crit: float = 3.5      # Critical doorway arching density (persons/m^2)
    beta_arching: float = 0.25 # Helbing arching non-linear degradation coefficient
    csi_threshold: float = 4.5 # Crowd density threshold for compressive stress (persons/m^2)
    csi_fatal: float = 15.0    # Cumulative fatal crush stress index (s * ped/m^2)


class FundamentalDiagram:
    """
    Computes density-dependent pedestrian velocity, flow rate,
    Helbing doorway bottleneck capacity breakdown, and crush asphyxia.
    """

    def __init__(self, params: WeidmannParams = WeidmannParams()):
        self.params = params

    def velocity(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculates pedestrian velocity v_e(rho_e) based on continuous Weidmann curve.
        """
        is_scalar = np.isscalar(density)
        rho = np.atleast_1d(np.asarray(density, dtype=np.float64))
        v = np.zeros_like(rho)

        free_mask = rho <= 0.0
        v[free_mask] = self.params.v0

        jam_mask = rho >= self.params.rho_jam
        v[jam_mask] = 0.0

        valid_mask = (~free_mask) & (~jam_mask)
        if np.any(valid_mask):
            rho_v = rho[valid_mask]
            exponent = -self.params.gamma * (1.0 / rho_v - 1.0 / self.params.rho_jam)
            exponent = np.clip(exponent, -50.0, 50.0)
            v_val = self.params.v0 * (1.0 - np.exp(exponent))
            v[valid_mask] = np.clip(v_val, 0.0, self.params.v0)

        if is_scalar:
            return float(v[0])
        return v

    def bottleneck_capacity(
        self,
        edge_capacity: float,
        doorway_width: float,
        specific_capacity: Union[float, None] = None
    ) -> float:
        """
        Computes junction bottleneck flow capping:
            Q_max = min(C_e, W_d * q_specific)
        """
        q_spec = specific_capacity if specific_capacity is not None else self.params.q_specific
        doorway_flow = doorway_width * q_spec
        return float(min(edge_capacity, doorway_flow))

    def flow_rate(self, density: Union[float, np.ndarray], width: float = 1.0) -> Union[float, np.ndarray]:
        """
        Calculates corridor flow rate Q = width * rho * v(rho) in persons / second.
        """
        v = self.velocity(density)
        return width * density * v

    def helbing_bottleneck_flow(
        self,
        nominal_capacity: float,
        density: float,
        doorway_width: float = 1.0
    ) -> Tuple[float, float]:
        """
        Evaluates Helbing Doorway Arching Capacity Breakdown:
            If rho <= rho_crit (3.5 ped/m^2):
                Q(t) = min(nominal_capacity, doorway_width * q_specific)
                effective_speed = velocity(density)
            If rho > rho_crit:
                arching_factor = max(0.10, 1.0 - beta * (rho - rho_crit)^2)
                Q(t) = nominal_flow * arching_factor
                effective_speed = max(min_velocity, velocity(density) * arching_factor)

        Returns:
            (Q_effective, effective_speed)
        """
        nominal_flow = min(nominal_capacity, doorway_width * self.params.q_specific)
        if density <= self.params.rho_crit:
            spd = max(self.params.min_velocity, float(self.velocity(density)))
            return float(nominal_flow), spd

        excess = density - self.params.rho_crit
        arching_factor = max(0.10, 1.0 - self.params.beta_arching * (excess ** 2))
        q_effective = nominal_flow * arching_factor
        spd = max(self.params.min_velocity, float(self.velocity(density)) * arching_factor)
        return float(q_effective), float(spd)

    def update_csi(self, current_csi: float, local_density: float, dt: float) -> Tuple[float, bool]:
        """
        Integrates Compressive Crowd Asphyxia (Crush Stress Index):
            Delta CSI = max(0, rho_local - 4.5) * dt
            Fatal condition: CSI >= 15.0

        Returns:
            (new_csi, is_crush_fatality)
        """
        stress = max(0.0, local_density - self.params.csi_threshold)
        new_csi = current_csi + stress * dt
        is_fatal = bool(new_csi >= self.params.csi_fatal)
        return float(new_csi), is_fatal
