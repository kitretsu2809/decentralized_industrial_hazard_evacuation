"""
Main simulation loop for the LBP Baseline Simulator.
Orchestrates: HazardModel → DijkstraRouter → SocialForceModel → Pedestrian states.
"""
from __future__ import annotations
import sys, os
import asyncio
import math
import random
import time
from typing import Dict, List, Optional, Set, Callable, Awaitable

# Add repo root to sys.path so we can import core/
_REPO = os.path.join(os.path.dirname(__file__), "..", "..", )
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from .pedestrian import Pedestrian, PedestrianState, spawn_pedestrians
from .hazard_model import HazardModel, HazardType, DANGER_THRESHOLD, LETHAL_THRESHOLD
from .dijkstra_router import DijkstraRouter
from .social_force import SocialForceModel


def _build_adjacency(edge_list):
    adj = {}
    for src, tgt, *_ in edge_list:
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)
    return adj


def _load_building():
    """Load the industrial plant building graph from the existing core module."""
    sys.path.insert(0, _REPO)
    try:
        from sim.services.environment.src.floorplan_generator import generate_industrial_plant
        from core.graph.types import NodeType
        building = generate_industrial_plant()

        node_positions: Dict[str, tuple] = {}
        node_floors: Dict[str, int]      = {}
        node_types: Dict[str, str]       = {}
        node_capacities: Dict[str, int]  = {}
        edge_list: List[tuple]           = []
        exits: List[str]                 = []

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

        # Cross-floor edges
        for edge in building.cross_floor_edges:
            edge_list.append((edge.source, edge.target, edge.distance, edge.width))

        return node_positions, node_floors, node_types, node_capacities, edge_list, exits, building.name

    except ImportError as e:
        raise RuntimeError(f"Could not import building model: {e}\n"
                           f"Run from the repo root: python run_simulator.py")


class Simulation:
    DT = 0.1          # physics timestep seconds
    REROUTE_INTERVAL = 3.0   # seconds between periodic reroutes

    def __init__(self):
        # Load building
        (self.node_positions, self.node_floors, self.node_types,
         self.node_capacities, self.edge_list, self.exits,
         self.building_name) = _load_building()

        adj = _build_adjacency(self.edge_list)
        # Ensure all nodes exist in adjacency
        for nid in self.node_positions:
            adj.setdefault(nid, [])

        # Sub-systems
        self.hazard  = HazardModel(adj)
        self.router  = DijkstraRouter()
        self.router.build(self.edge_list, self.exits, self.node_floors)
        self.sfm     = SocialForceModel(self.node_positions)

        # State
        self.agents: List[Pedestrian] = []
        self.t: float        = 0.0
        self.running: bool   = False
        self.speed: float    = 1.0    # time-scale multiplier
        self.num_evacuees: int = 60
        self._last_reroute: float = 0.0
        self._broadcast_cb: Optional[Callable[..., Awaitable]] = None
        self._task: Optional[asyncio.Task] = None

        self._init_agents()

    # ── Public control API ────────────────────────────────────────────────────

    def set_broadcast(self, cb: Callable[..., Awaitable]) -> None:
        self._broadcast_cb = cb

    def play(self) -> None:
        self.running = True

    def pause(self) -> None:
        self.running = False

    def reset(self, num_evacuees: Optional[int] = None) -> None:
        if num_evacuees is not None:
            self.num_evacuees = num_evacuees
        self.t       = 0.0
        self.running = False
        self.hazard.reset()
        self.router.update_weights({}, set())
        self._init_agents()

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, min(speed, 10.0))

    def inject_disaster(self, node_id: str, htype_str: str, intensity: float) -> bool:
        try:
            htype = HazardType(htype_str)
        except ValueError:
            return False
        if node_id not in self.node_positions:
            return False
        self.hazard.inject(node_id, htype, min(1.0, max(0.0, intensity)))
        # Force immediate reroute for all agents whose path passes through hazard
        self._reroute_all(force=True)
        return True

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Async loop: advances physics at DT, broadcasts state at ~10 Hz."""
        BROADCAST_INTERVAL = 0.10    # seconds wall-time between broadcasts
        last_broadcast = time.monotonic()

        while True:
            if self.running:
                real_dt = self.DT * self.speed
                self._tick(real_dt)

            now = time.monotonic()
            if now - last_broadcast >= BROADCAST_INTERVAL:
                if self._broadcast_cb:
                    await self._broadcast_cb(self.state_dict())
                last_broadcast = now

            await asyncio.sleep(max(0, self.DT - 0.005))

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        self.t += self.DT   # simulation time (not wall time)

        # 1. Advance hazard diffusion
        self.hazard.step(self.DT)

        # 2. Update router weights
        self.router.update_weights(self.hazard.levels, self.hazard.blocked)

        # 3. Periodic reroute
        if self.t - self._last_reroute >= self.REROUTE_INTERVAL:
            self._reroute_all()
            self._last_reroute = self.t

        # 4. Update pedestrian states
        self._update_states()

        # 5. Social Force Model step
        self.sfm.step(self.agents, self.hazard.levels, self.hazard.blocked, self.DT)

    def _reroute_all(self, force: bool = False) -> None:
        """Recompute Dijkstra paths for agents that need rerouting."""
        for agent in self.agents:
            if agent.state in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                continue
            needs = (
                force
                or not agent.path
                or self.router.path_has_hazard(agent.path, self.hazard.levels, 0.4)
                or self.router.is_blocked_path(agent.path, self.hazard.blocked)
            )
            if needs:
                new_path = self.router.get_path(agent.current_node or _nearest_node(
                    agent.x, agent.y, self.node_positions, agent.floor), agent.floor)
                if new_path:
                    agent.path = new_path[1:]  # first element is current node
                    if agent.state == PedestrianState.MOVING:
                        agent.state = PedestrianState.REROUTING
                    asyncio.get_event_loop().call_later(
                        random.uniform(0.5, 2.0),
                        lambda a=agent: setattr(a, "state", PedestrianState.MOVING)
                        if a.state == PedestrianState.REROUTING else None
                    )

    def _update_states(self) -> None:
        for agent in self.agents:
            if agent.state in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                continue

            # Check if reached exit
            if agent.current_node in self.exits:
                agent.state = PedestrianState.EVACUATED
                agent.vx, agent.vy, agent.path = 0.0, 0.0, []
                continue

            # Check hazard at current node
            h = self.hazard.levels.get(agent.current_node, 0.0)
            if h >= LETHAL_THRESHOLD:
                agent.state = PedestrianState.CASUALTY
                agent.vx, agent.vy, agent.path = 0.0, 0.0, []
            elif h >= DANGER_THRESHOLD:
                agent.state = PedestrianState.DANGER
            elif agent.state == PedestrianState.DANGER:
                agent.state = PedestrianState.MOVING

            # If no path, try to reroute
            if not agent.path and agent.state not in (
                    PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                src = agent.current_node or _nearest_node(
                    agent.x, agent.y, self.node_positions, agent.floor)
                path = self.router.get_path(src, agent.floor)
                if path:
                    agent.path = path[1:]

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _init_agents(self) -> None:
        self.agents = spawn_pedestrians(
            self.num_evacuees, self.node_positions, self.node_floors)
        # Compute initial Dijkstra paths
        for agent in self.agents:
            src = agent.current_node
            path = self.router.get_path(src, agent.floor)
            agent.path = path[1:] if path else []

    # ── State serialisation ───────────────────────────────────────────────────

    def state_dict(self) -> dict:
        m = self.metrics()
        return {
            "t":           round(self.t, 1),
            "running":     self.running,
            "speed":       self.speed,
            "pedestrians": [a.to_dict() for a in self.agents],
            "hazards":     self.hazard.to_dict(),
            "metrics":     m,
        }

    def metrics(self) -> dict:
        counts = {s: 0 for s in PedestrianState}
        for a in self.agents:
            counts[a.state] += 1
        return {
            "total":      len(self.agents),
            "moving":     counts[PedestrianState.MOVING],
            "rerouting":  counts[PedestrianState.REROUTING],
            "in_danger":  counts[PedestrianState.DANGER],
            "evacuated":  counts[PedestrianState.EVACUATED],
            "casualties": counts[PedestrianState.CASUALTY],
            "sim_time":   round(self.t, 1),
        }

    def building_dict(self) -> dict:
        """Return full building graph for frontend rendering."""
        nodes = {}
        for nid, pos in self.node_positions.items():
            nodes[nid] = {
                "id":       nid,
                "x":        pos[0],
                "y":        pos[1],
                "floor":    self.node_floors.get(nid, 1),
                "type":     self.node_types.get(nid, "ROOM"),
                "capacity": self.node_capacities.get(nid, 10),
                "is_exit":  nid in self.exits,
            }

        edges_out = []
        for src, tgt, dist, width in self.edge_list:
            edges_out.append({"source": src, "target": tgt,
                              "distance": dist, "width": width})

        return {
            "name":  self.building_name,
            "nodes": nodes,
            "edges": edges_out,
            "exits": self.exits,
        }


# ── Utility ───────────────────────────────────────────────────────────────────

def _nearest_node(x: float, y: float, positions: Dict[str, tuple],
                  floor: int) -> str:
    best, best_d = None, math.inf
    for nid, (nx_, ny_) in positions.items():
        d = math.hypot(x - nx_, y - ny_)
        if d < best_d:
            best_d = d
            best   = nid
    return best or ""
