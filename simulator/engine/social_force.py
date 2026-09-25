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
    def __init__(self, node_positions: Dict[str, Tuple[float, float]],
                 node_floors: Dict[str, int] = None,
                 edge_list: List = None):
        self.node_positions = node_positions
        self.node_floors = node_floors or {}
        # Build bidirectional edge widths mapping
        self.edge_widths: Dict[Tuple[str, str], float] = {}
        if edge_list:
            for item in edge_list:
                src, tgt, dist, width = item[0], item[1], item[2], item[3]
                self.edge_widths[(src, tgt)] = float(width)
                self.edge_widths[(tgt, src)] = float(width)

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
            # ── Normal State: Dwelling at assigned workstation room ──────────
            if agent.state.value == "normal":
                agent.wander_timer -= dt
                dist_to_wander = math.hypot(agent.wander_target_x - px[i], agent.wander_target_y - py[i])
                if agent.wander_timer <= 0.0 or dist_to_wander < 0.2:
                    agent.wander_timer = random.uniform(3.0, 7.0)
                    ang = random.uniform(0, 2 * math.pi)
                    rad = random.uniform(0.2, 1.0)
                    agent.wander_target_x = agent.anchor_x + rad * math.cos(ang)
                    agent.wander_target_y = agent.anchor_y + rad * math.sin(ang)

                dx_ = agent.wander_target_x - px[i]
                dy_ = agent.wander_target_y - py[i]
                dist = math.hypot(dx_, dy_)
                if dist > 0.08:
                    v_slow = min(0.30, dist * 0.5)
                    e_x, e_y = dx_ / dist, dy_ / dist
                    fx[i] += MASS * (e_x * v_slow - vx[i]) / TAU
                    fy[i] += MASS * (e_y * v_slow - vy[i]) / TAU
                else:
                    fx[i] += MASS * (0.0 - vx[i]) / TAU
                    fy[i] += MASS * (0.0 - vy[i]) / TAU

            # ── Reacting State: Recognition Hesitation ────────────────────────
            elif agent.state.value == "reacting":
                # Promptly decelerate to a halt
                fx[i] += MASS * (0.0 - vx[i]) / (TAU * 0.4)
                fy[i] += MASS * (0.0 - vy[i]) / (TAU * 0.4)

            # ── Evacuation States: Moving strictly along corridors ────────────
            else:
                if not agent.path:
                    fx[i] += MASS * (0.0 - vx[i]) / TAU
                    fy[i] += MASS * (0.0 - vy[i]) / TAU
                else:
                    target_id = agent.path[0]
                    curr_id   = agent.current_node or target_id

                    # 1. Cross-floor stairwell / elevator transition
                    target_floor = self.node_floors.get(target_id, agent.floor)
                    if target_floor != agent.floor:
                        agent.floor = target_floor
                        agent.current_node = target_id
                        agent.path = agent.path[1:]
                        tx_, ty_ = self.node_positions.get(target_id, (agent.x, agent.y))
                        agent.x, agent.y = tx_, ty_
                        continue

                    # 2. Check if reached waypoint node
                    tx_, ty_ = self.node_positions.get(target_id, (px[i], py[i]))
                    d_target = math.hypot(tx_ - px[i], ty_ - py[i])
                    if d_target < 1.0: # Reached node!
                        agent.current_node = target_id
                        agent.path = agent.path[1:]
                        if not agent.path:
                            continue
                        next_id = agent.path[0]
                        next_floor = self.node_floors.get(next_id, agent.floor)
                        if next_floor != agent.floor:
                            agent.floor = next_floor
                            agent.current_node = next_id
                            agent.path = agent.path[1:]
                            nx_, ny_ = self.node_positions.get(next_id, (agent.x, agent.y))
                            agent.x, agent.y = nx_, ny_
                        continue

                    # 3. Corridor Guidance Along Edge (curr_id -> target_id)
                    cx_, cy_ = self.node_positions.get(curr_id, (px[i], py[i]))
                    vx_edge = tx_ - cx_
                    vy_edge = ty_ - cy_
                    L_edge = math.hypot(vx_edge, vy_edge)

                    if L_edge < 0.1:
                        ux_edge, uy_edge = 0.0, 0.0
                    else:
                        ux_edge = vx_edge / L_edge
                        uy_edge = vy_edge / L_edge

                    # Corridor width & max allowable lateral deviation
                    w_corridor = self.edge_widths.get((curr_id, target_id), 2.5)
                    r_corridor = max(0.35, (w_corridor / 2.0) - 0.20)

                    # Projection onto corridor centerline
                    s_proj = (px[i] - cx_) * ux_edge + (py[i] - cy_) * uy_edge
                    s_clamped = max(0.0, min(L_edge, s_proj))
                    proj_x = cx_ + s_clamped * ux_edge
                    proj_y = cy_ + s_clamped * uy_edge

                    dx_perp = px[i] - proj_x
                    dy_perp = py[i] - proj_y
                    d_perp = math.hypot(dx_perp, dy_perp)

                    # Wall Restoring Force (keeps pedestrian on the corridor path)
                    if d_perp > 0.15:
                        fx[i] += -MASS * 8.0 * dx_perp
                        fy[i] += -MASS * 8.0 * dy_perp

                    # Lookahead goal vector strictly on the corridor centerline
                    s_lookahead = min(L_edge, s_clamped + 2.5)
                    target_look_x = cx_ + s_lookahead * ux_edge
                    target_look_y = cy_ + s_lookahead * uy_edge

                    dx_goal = target_look_x - px[i]
                    dy_goal = target_look_y - py[i]
                    d_goal = math.hypot(dx_goal, dy_goal)

                    if d_goal >= 0.05:
                        # Local crowd density for Weidmann speed reduction
                        diff_sq = (px - px[i])**2 + (py - py[i])**2
                        local_count = float(np.sum(diff_sq < 9.0) - 1)
                        local_density = local_count / (math.pi * 9.0)
                        v_des = weidmann_speed(agent.desired_speed, local_density)

                        e_x = dx_goal / d_goal
                        e_y = dy_goal / d_goal
                        fx[i] += MASS * (e_x * v_des - vx[i]) / TAU
                        fy[i] += MASS * (e_y * v_des - vy[i]) / TAU

            # ── Agent-Agent Mutual Repulsion (same floor only) ────────────────
            for j in range(n):
                if j == i or active[j].floor != agent.floor:
                    continue
                ddx, ddy = px[i] - px[j], py[i] - py[j]
                d = math.hypot(ddx, ddy)
                if d < 0.01 or d > 3.0:
                    continue
                r_ij = R_AGENT * 2
                mag = A_REPULSE * math.exp((r_ij - d) / B_REPULSE)
                fx[i] += mag * ddx / d
                fy[i] += mag * ddy / d

            # ── Hazard flee force ─────────────────────────────────────────────
            h = hazard_levels.get(agent.current_node, 0.0)
            if h > 0.05:
                hx_, hy_ = self.node_positions.get(agent.current_node, (px[i], py[i]))
                ddx_, ddy_ = px[i] - hx_, py[i] - hy_
                dd_ = math.hypot(ddx_, ddy_)
                if dd_ < 0.01: ddx_, ddy_, dd_ = 1.0, 0.0, 1.0
                flee_mag = MASS * K_FLEE * h
                fx[i] += flee_mag * ddx_ / dd_
                fy[i] += flee_mag * ddy_ / dd_

        # ── Integration and Hard Corridor Wall Clamping ──────────────────────
        for i, agent in enumerate(active):
            f_mag = math.hypot(fx[i], fy[i])
            if f_mag > MASS * 15.0:
                scale_ = MASS * 15.0 / f_mag
                fx[i] *= scale_
                fy[i] *= scale_

            agent.vx += (fx[i] / MASS) * dt
            agent.vy += (fy[i] / MASS) * dt

            v = math.hypot(agent.vx, agent.vy)
            if v > V_MAX_ABS:
                agent.vx *= V_MAX_ABS / v
                agent.vy *= V_MAX_ABS / v

            agent.x += agent.vx * dt
            agent.y += agent.vy * dt
            agent.age += dt

            # ── Hard Boundary Enforcements (No wall penetration) ─────────────
            if agent.state.value == "normal":
                # Clamp within 1.5m of room workstation anchor
                d_anch = math.hypot(agent.x - agent.anchor_x, agent.y - agent.anchor_y)
                if d_anch > 1.4:
                    scale_a = 1.4 / d_anch
                    agent.x = agent.anchor_x + (agent.x - agent.anchor_x) * scale_a
                    agent.y = agent.anchor_y + (agent.y - agent.anchor_y) * scale_a

            elif agent.state.value in ("moving", "rerouting", "danger") and agent.path:
                curr_id = agent.current_node or agent.path[0]
                target_id = agent.path[0]
                if self.node_floors.get(target_id, agent.floor) == agent.floor:
                    cx_, cy_ = self.node_positions.get(curr_id, (agent.x, agent.y))
                    tx_, ty_ = self.node_positions.get(target_id, (agent.x, agent.y))
                    vx_e = tx_ - cx_
                    vy_e = ty_ - cy_
                    L_e = math.hypot(vx_e, vy_e)
                    if L_e > 0.1:
                        ux_e, uy_e = vx_e / L_e, vy_e / L_e
                        w_c = self.edge_widths.get((curr_id, target_id), 2.5)
                        r_c = max(0.35, (w_c / 2.0) - 0.15)

                        s_ = (agent.x - cx_) * ux_e + (agent.y - cy_) * uy_e
                        s_c = max(0.0, min(L_e, s_))
                        p_cx = cx_ + s_c * ux_e
                        p_cy = cy_ + s_c * uy_e
                        perp_x = agent.x - p_cx
                        perp_y = agent.y - p_cy
                        d_perp = math.hypot(perp_x, perp_y)
                        if d_perp > r_c:
                            scale_p = r_c / d_perp
                            agent.x = p_cx + perp_x * scale_p
                            agent.y = p_cy + perp_y * scale_p
                            # Zero out velocity into wall
                            v_tangent = agent.vx * ux_e + agent.vy * uy_e
                            agent.vx = v_tangent * ux_e
                            agent.vy = v_tangent * uy_e
