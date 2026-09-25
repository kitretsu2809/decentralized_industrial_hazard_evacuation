"""
Social Force Model — Helbing & Molnár (1995), Physical Review E 51, 4282.
Fixed: forces computed simultaneously for all agents (correct physics).
Fixed: proper mass, capped forces, no mid-loop position updates.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .pedestrian import Pedestrian

# ── SFM parameters ───────────────────────────────────────────────────────────
TAU        = 0.5      # relaxation time (s)
A_REPULSE  = 300.0   # repulsion strength — tuned down from Helbing to avoid instability
B_REPULSE  = 0.10    # length scale (m)
R_AGENT    = 0.30    # pedestrian radius (m)
MASS       = 80.0    # kg (realistic human mass)
K_FLEE     = 4.0     # hazard flee multiplier
V_FREE     = 1.34    # mean free-flow speed (m/s) — Weidmann 1992
V_JAM      = 5.4     # jamming density (persons/m²)
V_MAX_ABS  = 4.0     # hard cap (sprint)


def weidmann_speed(v_free: float, density: float) -> float:
    """Velocity-density relation from Weidmann (1992)."""
    if density < 0.01:
        return v_free
    rho = min(density, V_JAM - 0.05)
    factor = 1.0 - math.exp(-1.913 * (1.0 / rho - 1.0 / V_JAM))
    return v_free * max(0.0, factor)


class SocialForceModel:
    def __init__(self, node_positions: Dict[str, Tuple[float, float]]):
        self.node_positions = node_positions

    def step(self, agents: List, hazard_levels: Dict[str, float],
             blocked: set, dt: float) -> None:
        active = [a for a in agents if a.state.value in ("normal", "reacting", "moving", "rerouting", "danger")]
        if not active:
            return

        n = len(active)
        # Snapshot positions BEFORE any updates (correct simultaneous physics)
        px = np.array([a.x  for a in active], dtype=np.float64)
        py = np.array([a.y  for a in active], dtype=np.float64)
        vx = np.array([a.vx for a in active], dtype=np.float64)
        vy = np.array([a.vy for a in active], dtype=np.float64)

        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)

        import random

        for i, agent in enumerate(active):
            # ── Normal State: Dwelling / Micro-movement at Workstation ────────
            if agent.state.value == "normal":
                agent.wander_timer -= dt
                dist_to_wander = math.hypot(agent.wander_target_x - px[i], agent.wander_target_y - py[i])
                if agent.wander_timer <= 0.0 or dist_to_wander < 0.25:
                    agent.wander_timer = random.uniform(3.0, 7.0)
                    # Gentle roaming around anchor workstation (radius ~1.5m)
                    ang = random.uniform(0, 2 * math.pi)
                    rad = random.uniform(0.2, 1.8)
                    agent.wander_target_x = agent.anchor_x + rad * math.cos(ang)
                    agent.wander_target_y = agent.anchor_y + rad * math.sin(ang)

                dx_ = agent.wander_target_x - px[i]
                dy_ = agent.wander_target_y - py[i]
                dist = math.hypot(dx_, dy_)
                if dist > 0.1:
                    v_slow = min(0.35, dist * 0.4)
                    e_x, e_y = dx_ / dist, dy_ / dist
                    fx[i] += MASS * (e_x * v_slow - vx[i]) / TAU
                    fy[i] += MASS * (e_y * v_slow - vy[i]) / TAU
                else:
                    fx[i] += MASS * (0.0 - vx[i]) / TAU
                    fy[i] += MASS * (0.0 - vy[i]) / TAU

            # ── Reacting State: Recognition Hesitation ────────────────────────
            elif agent.state.value == "reacting":
                # Decelerate to pause as the emergency alarm sounds
                fx[i] += MASS * (0.0 - vx[i]) / (TAU * 0.5)
                fy[i] += MASS * (0.0 - vy[i]) / (TAU * 0.5)

            # ── Evacuation States: MOVING / REROUTING / DANGER ─────────────────
            else:
                if not agent.path:
                    fx[i] += MASS * (0.0 - vx[i]) / TAU
                    fy[i] += MASS * (0.0 - vy[i]) / TAU
                else:
                    target_id = agent.path[0]
                    if target_id not in self.node_positions:
                        agent.path = agent.path[1:]
                        continue

                    tx_, ty_ = self.node_positions[target_id]

                    # Arrive at waypoint
                    if math.hypot(tx_ - px[i], ty_ - py[i]) < 2.5:
                        agent.current_node = target_id
                        agent.path = agent.path[1:]
                        if not agent.path:
                            continue
                        target_id = agent.path[0]
                        tx_, ty_ = self.node_positions.get(target_id, (px[i], py[i]))

                    dx_, dy_ = tx_ - px[i], ty_ - py[i]
                    dist = math.hypot(dx_, dy_)
                    if dist >= 0.01:
                        # Local density for Weidmann correction
                        diff_sq = (px - px[i])**2 + (py - py[i])**2
                        local_count = float(np.sum(diff_sq < 9.0) - 1)
                        local_density = local_count / (math.pi * 9.0)
                        v_des = weidmann_speed(agent.desired_speed, local_density)

                        e_x, e_y = dx_ / dist, dy_ / dist
                        g_x = MASS * (e_x * v_des - vx[i]) / TAU
                        g_y = MASS * (e_y * v_des - vy[i]) / TAU
                        fx[i] += g_x
                        fy[i] += g_y

            # ── Agent Mutual Repulsion ────────────────────────────────────────
            for j in range(n):
                if j == i or active[j].floor != agent.floor:
                    continue
                ddx, ddy = px[i] - px[j], py[i] - py[j]
                d = math.hypot(ddx, ddy)
                if d < 0.01 or d > 4.0:
                    continue
                r_ij = R_AGENT * 2
                mag = A_REPULSE * math.exp((r_ij - d) / B_REPULSE)
                fx[i] += mag * ddx / d
                fy[i] += mag * ddy / d

            # ── Hazard flee force (if in hazardous zone) ──────────────────────
            h = hazard_levels.get(agent.current_node, 0.0)
            if h > 0.05:
                hx_, hy_ = self.node_positions.get(agent.current_node, (px[i], py[i]))
                ddx_, ddy_ = px[i] - hx_, py[i] - hy_
                dd_ = math.hypot(ddx_, ddy_)
                if dd_ < 0.01: ddx_, ddy_, dd_ = 1.0, 0.0, 1.0
                flee_mag = MASS * K_FLEE * h
                fx[i] += flee_mag * ddx_ / dd_
                fy[i] += flee_mag * ddy_ / dd_

        # ── Clamp forces & integrate (all agents simultaneously) ─────────────
        for i, agent in enumerate(active):
            f_mag = math.hypot(fx[i], fy[i])
            if f_mag > MASS * 15.0:   # cap at 15 m/s²
                scale_ = MASS * 15.0 / f_mag
                fx[i] *= scale_
                fy[i] *= scale_

            agent.vx += (fx[i] / MASS) * dt
            agent.vy += (fy[i] / MASS) * dt

            # Hard speed cap
            v = math.hypot(agent.vx, agent.vy)
            if v > V_MAX_ABS:
                agent.vx *= V_MAX_ABS / v
                agent.vy *= V_MAX_ABS / v

            agent.x += agent.vx * dt
            agent.y += agent.vy * dt
            agent.age += dt
