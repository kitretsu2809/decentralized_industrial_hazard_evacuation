"""
Social Force Model — Helbing & Molnár (1995)
"Social force model for pedestrian dynamics", Physical Review E 51, 4282.

F_total = F_goal + F_agent_repulsion + F_hazard_flee

Each pedestrian is treated as a particle in continuous 2D space.
Integration: Euler forward with dt = 0.05–0.1 s.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .pedestrian import Pedestrian

# ── SFM Parameters (Helbing 1995) ─────────────────────────────────────────────
TAU         = 0.5     # relaxation time (s) — how fast agent reaches desired vel
A_repulse   = 2000.0  # agent repulsion strength (N) — Helbing 1995 Table I
B_repulse   = 0.08    # agent repulsion length scale (m)
R_agent     = 0.25    # physical radius of pedestrian (m)
K_FLEE      = 6.0     # hazard flee force multiplier
V_JAM       = 5.4     # jamming density (persons/m²) — Weidmann 1992

# ── Weidmann velocity-density correction ──────────────────────────────────────
def weidmann_speed(v_free: float, density: float) -> float:
    """
    v(ρ) = v_free * (1 - exp(-1.913 * (1/ρ - 1/ρ_jam)))
    Caps free-flow speed based on local crowd density ρ (persons/m²).
    """
    if density <= 0.01:
        return v_free
    rho = max(0.01, min(density, V_JAM - 0.1))
    factor = 1.0 - math.exp(-1.913 * (1.0 / rho - 1.0 / V_JAM))
    return v_free * max(0.0, factor)


class SocialForceModel:
    """
    Computes the total social force on each pedestrian and integrates motion.
    """

    def __init__(self, node_positions: Dict[str, Tuple[float, float]]):
        """
        node_positions: {node_id: (x, y)} in metres
        """
        self.node_positions = node_positions

    def step(self, agents: List["Pedestrian"], hazard_levels: Dict[str, float],
             blocked: set, dt: float) -> None:
        """
        Update all pedestrians' positions by dt seconds.
        Modifies agents in-place.
        """
        # Build position array for vectorised repulsion
        active = [a for a in agents if a.state.value in ("moving", "rerouting", "danger")]
        if not active:
            return

        positions = np.array([(a.x, a.y) for a in active], dtype=np.float64)

        for i, agent in enumerate(active):
            if not agent.path:
                agent.vx, agent.vy = 0.0, 0.0
                continue

            # ── Goal force ───────────────────────────────────────────────────
            target_id = agent.path[0]
            if target_id not in self.node_positions:
                agent.path = agent.path[1:]
                continue

            tx, ty   = self.node_positions[target_id]
            dx, dy   = tx - agent.x, ty - agent.y
            dist_to  = math.hypot(dx, dy)

            # Arrive at waypoint: advance to next node
            if dist_to < 2.5:   # within 2.5 m = "at node"
                agent.current_node = target_id
                agent.path = agent.path[1:]
                if not agent.path:
                    agent.vx, agent.vy = 0.0, 0.0
                    continue
                target_id   = agent.path[0]
                tx, ty      = self.node_positions.get(target_id, (agent.x, agent.y))
                dx, dy      = tx - agent.x, ty - agent.y
                dist_to     = math.hypot(dx, dy)

            if dist_to < 0.01:
                dist_to = 0.01

            # Local density estimate (count neighbours within 3m)
            near = np.linalg.norm(positions - positions[i], axis=1)
            local_density = float(np.sum(near < 3.0) - 1) / (math.pi * 3.0 ** 2)
            v_desired = weidmann_speed(agent.desired_speed, local_density)

            # Desired velocity vector
            e_x, e_y = dx / dist_to, dy / dist_to
            vd_x, vd_y = e_x * v_desired, e_y * v_desired

            # Goal force: drive current velocity toward desired
            f_goal_x = (vd_x - agent.vx) / TAU
            f_goal_y = (vd_y - agent.vy) / TAU

            # ── Agent repulsion ───────────────────────────────────────────────
            f_rep_x, f_rep_y = 0.0, 0.0
            for j, other in enumerate(active):
                if j == i:
                    continue
                if other.floor != agent.floor:
                    continue
                ddx, ddy = agent.x - other.x, agent.y - other.y
                d = math.hypot(ddx, ddy)
                if d < 0.01 or d > 3.0:  # ignore beyond 3 m
                    continue
                r_ij = R_agent * 2   # sum of radii
                mag  = A_repulse * math.exp((r_ij - d) / B_repulse)
                f_rep_x += mag * ddx / d
                f_rep_y += mag * ddy / d

            # Clamp repulsion to avoid numerical explosion
            rep_mag = math.hypot(f_rep_x, f_rep_y)
            if rep_mag > 50.0:
                f_rep_x *= 50.0 / rep_mag
                f_rep_y *= 50.0 / rep_mag

            # ── Hazard flee force ─────────────────────────────────────────────
            f_flee_x, f_flee_y = 0.0, 0.0
            h_here = hazard_levels.get(agent.current_node, 0.0)
            if h_here > 0.1:
                # Flee radially away from current hazard node
                hx, hy   = self.node_positions.get(agent.current_node, (agent.x, agent.y))
                ddx, ddy = agent.x - hx, agent.y - hy
                d        = math.hypot(ddx, ddy)
                if d < 0.01:
                    ddx, ddy = 1.0, 0.0
                    d = 1.0
                f_flee_x = K_FLEE * h_here * ddx / d
                f_flee_y = K_FLEE * h_here * ddy / d

            # ── Total force → acceleration (mass = 1 kg, simplified) ─────────
            ax = f_goal_x + f_rep_x + f_flee_x
            ay = f_goal_y + f_rep_y + f_flee_y

            # Clamp total acceleration
            a_mag = math.hypot(ax, ay)
            if a_mag > 20.0:
                ax *= 20.0 / a_mag
                ay *= 20.0 / a_mag

            # Euler integration
            agent.vx += ax * dt
            agent.vy += ay * dt

            # Hard speed cap: v_max = 3.5 m/s (sprint)
            v_mag = math.hypot(agent.vx, agent.vy)
            v_cap = min(agent.desired_speed * 2.0, 3.5)
            if v_mag > v_cap:
                agent.vx *= v_cap / v_mag
                agent.vy *= v_cap / v_mag

            # Update position
            agent.x += agent.vx * dt
            agent.y += agent.vy * dt
            agent.age += dt

            # Update positions array for subsequent agents this tick
            positions[i] = [agent.x, agent.y]
