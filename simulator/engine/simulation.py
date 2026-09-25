"""
Main simulation loop — fixed-timestep accumulator pattern.
Speed multiplier correctly controls how many physics steps run per wall-second.
"""
from __future__ import annotations
import sys, os, asyncio, math, random, time
from typing import Dict, List, Optional, Callable, Awaitable

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from .pedestrian import Pedestrian, PedestrianState, spawn_pedestrians
from .hazard_model import HazardModel, HazardType, DANGER_THRESHOLD, LETHAL_THRESHOLD
from .dijkstra_router import DijkstraRouter
from .social_force import SocialForceModel


def _build_adjacency_with_dist(edge_list):
    """Returns {node: [(neighbour, distance_m), ...]}"""
    adj = {}
    for src, tgt, dist, *_ in edge_list:
        adj.setdefault(src, []).append((tgt, float(dist)))
        adj.setdefault(tgt, []).append((src, float(dist)))
    return adj


def _load_building():
    sys.path.insert(0, _REPO)
    try:
        from sim.services.environment.src.floorplan_generator import generate_industrial_plant
        from core.graph.types import NodeType
        building = generate_industrial_plant()

        node_positions, node_floors, node_types, node_capacities = {}, {}, {}, {}
        edge_list, exits = [], []

        for floor_level, floor in building.floors.items():
            for nid, node in floor.nodes.items():
                node_positions[nid] = node.position
                node_floors[nid]    = floor_level
                node_types[nid]     = node.type.value
                node_capacities[nid]= node.capacity
                if node.type == NodeType.EXIT:
                    exits.append(nid)
            for edge in floor.edges:
                edge_list.append((edge.source, edge.target, edge.distance, edge.width))
        for edge in building.cross_floor_edges:
            edge_list.append((edge.source, edge.target, edge.distance, edge.width))

        b_name = building.name.replace("LBP ", "").strip()
        return (node_positions, node_floors, node_types,
                node_capacities, edge_list, exits, b_name)
    except ImportError as e:
        raise RuntimeError(f"Could not import building model: {e}")


class Simulation:
    DT = 0.1            # physics timestep (seconds) — keep ≤0.1 for SFM stability

    def __init__(self):
        (self.node_positions, self.node_floors, self.node_types,
         self.node_capacities, self.edge_list, self.exits,
         self.building_name) = _load_building()

        adj_with_dist = _build_adjacency_with_dist(self.edge_list)
        for nid in self.node_positions:
            adj_with_dist.setdefault(nid, [])

        self.hazard  = HazardModel(adj_with_dist)
        self.router  = DijkstraRouter()
        self.router.build(self.edge_list, self.exits, self.node_floors)
        self.sfm     = SocialForceModel(self.node_positions)

        self.agents: List[Pedestrian] = []
        self.t: float        = 0.0
        self.running: bool   = True   # auto-start
        self.alarm_active: bool = False  # Facility operating normally by default
        self.speed: float    = 2.0    # 2× default so movement is clearly visible
        self.num_evacuees: int = 60
        self._last_reroute: float = 0.0
        self._accumulator: float  = 0.0
        self._broadcast_cb: Optional[Callable[..., Awaitable]] = None

        self._init_agents()

    # ── Control API ───────────────────────────────────────────────────────────
    def set_broadcast(self, cb): self._broadcast_cb = cb
    def play(self):  self.running = True
    def pause(self): self.running = False

    def reset(self, num_evacuees=None):
        if num_evacuees: self.num_evacuees = num_evacuees
        self.t = 0.0
        self._accumulator = 0.0
        self._last_reroute = 0.0
        self.alarm_active = False
        self.hazard.reset()
        self.router.update_weights({}, set())
        self._init_agents()
        self.running = True   # auto-resume on reset

    def set_speed(self, speed): self.speed = max(0.1, min(10.0, speed))

    def trigger_alarm(self):
        """Sound the facility-wide evacuation alarm (initiates egress reaction)."""
        self.alarm_active = True
        for agent in self.agents:
            if agent.state == PedestrianState.NORMAL:
                agent.state = PedestrianState.REACTING

    def reset_alarm(self):
        """Cancel alarm and return safe workers to workstation duty."""
        self.alarm_active = False
        for agent in self.agents:
            if agent.state in (PedestrianState.REACTING, PedestrianState.MOVING, PedestrianState.REROUTING):
                agent.state = PedestrianState.NORMAL
                agent.path = []
                agent.vx = agent.vy = 0.0

    def toggle_alarm(self):
        if self.alarm_active:
            self.reset_alarm()
        else:
            self.trigger_alarm()

    def inject_disaster(self, node_id, htype_str, intensity):
        try:    htype = HazardType(htype_str)
        except: return False
        if node_id not in self.node_positions: return False
        self.hazard.inject(node_id, htype, min(1.0, max(0.0, intensity)))
        # Automatically trigger facility evacuation alarm upon hazard injection
        self.trigger_alarm()
        self._reroute_all(force=True)
        return True

    # ── Main async loop (fixed-timestep accumulator) ──────────────────────────
    async def run_loop(self):
        BROADCAST_INTERVAL = 0.08   # ~12 Hz
        LOOP_SLEEP         = 0.012  # ~80 Hz loop
        last_broadcast = time.monotonic()
        last_loop      = time.monotonic()

        while True:
            now     = time.monotonic()
            wall_dt = min(now - last_loop, 0.1)   # cap to avoid spiral-of-death
            last_loop = now

            if self.running:
                # Accumulate sim-time to advance
                self._accumulator += wall_dt * self.speed
                # Run as many full DT-steps as the accumulator allows
                steps = 0
                while self._accumulator >= self.DT and steps < 20:
                    self._tick()
                    self._accumulator -= self.DT
                    steps += 1

            if now - last_broadcast >= BROADCAST_INTERVAL:
                if self._broadcast_cb:
                    await self._broadcast_cb(self.state_dict())
                last_broadcast = now

            await asyncio.sleep(LOOP_SLEEP)

    # ── Single physics tick ───────────────────────────────────────────────────
    def _tick(self):
        self.t += self.DT
        self.hazard.step(self.DT)
        self.router.update_weights(self.hazard.levels, self.hazard.blocked)

        if self.t - self._last_reroute >= 3.0:
            self._reroute_all()
            self._last_reroute = self.t

        self._update_states()
        self.sfm.step(self.agents, self.hazard.levels, self.hazard.blocked, self.DT)

    def _reroute_all(self, force=False):
        for agent in self.agents:
            if agent.state in (PedestrianState.NORMAL, PedestrianState.REACTING,
                               PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                continue
            needs = (
                force
                or not agent.path
                or self.router.path_has_hazard(agent.path, self.hazard.levels, 0.4)
                or self.router.is_blocked_path(agent.path, self.hazard.blocked)
            )
            if needs:
                src = agent.current_node or _nearest_node(
                    agent.x, agent.y, self.node_positions, agent.floor)
                new_path = self.router.get_path(src, agent.floor)
                if new_path:
                    agent.path = new_path[1:]
                    if agent.state == PedestrianState.MOVING:
                        agent.state = PedestrianState.REROUTING

    def _update_states(self):
        for agent in self.agents:
            if agent.state in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                continue

            h = self.hazard.levels.get(agent.current_node, 0.0)

            # 1. Casualty check
            if h >= LETHAL_THRESHOLD:
                agent.state = PedestrianState.CASUALTY
                agent.vx = agent.vy = 0.0
                agent.path = []
                continue

            # 2. Reached exit check
            if agent.current_node in self.exits:
                agent.state = PedestrianState.EVACUATED
                agent.vx = agent.vy = 0.0
                agent.path = []
                continue

            # 3. Danger override: if local hazard threatens room, react immediately
            if h >= DANGER_THRESHOLD:
                agent.state = PedestrianState.DANGER
                if not agent.path or self.router.is_blocked_path(agent.path, self.hazard.blocked):
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, agent.floor)
                    p = self.router.get_path(src, agent.floor)
                    if p: agent.path = p[1:]
                continue

            # 4. Normal Operating Regime (No alarm active, safe room)
            if not self.alarm_active and h < 0.15:
                if agent.state != PedestrianState.NORMAL:
                    agent.state = PedestrianState.NORMAL
                    agent.path = []
                    agent.vx = agent.vy = 0.0
                continue

            # 5. Emergency Alarm Transition (Alarm active or hazard nearby)
            if agent.state == PedestrianState.NORMAL:
                agent.state = PedestrianState.REACTING

            if agent.state == PedestrianState.REACTING:
                agent.reaction_delay -= self.DT
                if agent.reaction_delay <= 0.0:
                    agent.state = PedestrianState.MOVING
                    # Compute safe egress path now that worker has reacted
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, agent.floor)
                    p = self.router.get_path(src, agent.floor)
                    if p: agent.path = p[1:]

            elif agent.state in (PedestrianState.MOVING, PedestrianState.REROUTING):
                if not agent.path:
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, agent.floor)
                    p = self.router.get_path(src, agent.floor)
                    if p: agent.path = p[1:]
                if agent.state == PedestrianState.REROUTING and agent.path:
                    agent.state = PedestrianState.MOVING

    def _init_agents(self):
        self.agents = spawn_pedestrians(
            self.num_evacuees, self.node_positions, self.node_floors)
        # Workers begin in NORMAL operating state dwelling at their stations (no exit paths initially)
        for agent in self.agents:
            agent.state = PedestrianState.NORMAL
            agent.path = []

    # ── Serialisation ─────────────────────────────────────────────────────────
    def state_dict(self):
        return {
            "t":            round(self.t, 1),
            "running":      self.running,
            "alarm_active": self.alarm_active,
            "speed":        self.speed,
            "pedestrians":  [a.to_dict() for a in self.agents],
            "hazards":      self.hazard.to_dict(),
            "metrics":      self.metrics(),
        }

    def metrics(self):
        counts = {s: 0 for s in PedestrianState}
        for a in self.agents: counts[a.state] += 1
        return {
            "total":        len(self.agents),
            "normal":       counts[PedestrianState.NORMAL],
            "reacting":     counts[PedestrianState.REACTING],
            "moving":       counts[PedestrianState.MOVING],
            "rerouting":    counts[PedestrianState.REROUTING],
            "in_danger":    counts[PedestrianState.DANGER],
            "evacuated":    counts[PedestrianState.EVACUATED],
            "casualties":   counts[PedestrianState.CASUALTY],
            "alarm_active": self.alarm_active,
            "sim_time":     round(self.t, 1),
        }

    def building_dict(self):
        nodes = {}
        for nid, pos in self.node_positions.items():
            nodes[nid] = {
                "id": nid, "x": pos[0], "y": pos[1],
                "floor": self.node_floors.get(nid, 1),
                "type":  self.node_types.get(nid, "ROOM"),
                "capacity": self.node_capacities.get(nid, 10),
                "is_exit": nid in self.exits,
            }
        edges_out = [{"source": s, "target": t, "distance": d, "width": w}
                     for s, t, d, w in self.edge_list]
        return {"name": self.building_name, "nodes": nodes,
                "edges": edges_out, "exits": self.exits}


def _nearest_node(x, y, positions, floor):
    best, best_d = None, math.inf
    for nid, (nx_, ny_) in positions.items():
        d = math.hypot(x - nx_, y - ny_)
        if d < best_d:
            best_d = d; best = nid
    return best or ""
