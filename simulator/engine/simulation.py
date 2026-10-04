"""
Main simulation loop — fixed-timestep accumulator pattern.
Speed multiplier correctly controls how many physics steps run per wall-second.
"""
from __future__ import annotations
import sys, os, asyncio, math, random, time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Awaitable

import numpy as np

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


def _compute_static_exit_distances(edge_list, exits, node_positions):
    """Computes shortest topological distances from every node to the nearest exit."""
    import networkx as nx
    G = nx.Graph()
    for src, tgt, dist, *_ in edge_list:
        # Exclude elevator/hoist shafts from emergency egress paths (NFPA 101)
        if "hoist" in src and "hoist" in tgt:
            continue
        G.add_edge(src, tgt, weight=float(dist))
    for n in node_positions:
        if n not in G:
            G.add_node(n)
    exit_dists = {}
    for node in node_positions:
        dists = []
        for ex in exits:
            try:
                d = nx.shortest_path_length(G, node, ex, weight="weight")
                dists.append(d)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        exit_dists[node] = min(dists) if dists else 100.0
    return exit_dists


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
    POLICY_DECISION_DT = 1.0  # seconds between live neural signboard decisions

    def __init__(self):
        (self.node_positions, self.node_floors, self.node_types,
         self.node_capacities, self.edge_list, self.exits,
         self.building_name) = _load_building()

        adj_with_dist = _build_adjacency_with_dist(self.edge_list)
        for nid in self.node_positions:
            adj_with_dist.setdefault(nid, [])

        self.hazard  = HazardModel(adj_with_dist, exits=self.exits, node_floors=self.node_floors)
        self.router  = DijkstraRouter()
        self.router.build(self.edge_list, self.exits, self.node_floors)
        self.sfm     = SocialForceModel(self.node_positions, self.node_floors, self.edge_list)
        self.static_exit_dists = _compute_static_exit_distances(
            self.edge_list, self.exits, self.node_positions
        )

        self.agents: List[Pedestrian] = []
        self.t: float        = 0.0
        self.running: bool   = True   # auto-start
        self.alarm_active: bool = False  # Facility operating normally by default
        self.speed: float    = 2.0    # 2× default so movement is clearly visible
        self.num_evacuees: int = 60
        self._last_reroute: float = 0.0
        self._accumulator: float  = 0.0
        self._broadcast_cb: Optional[Callable[..., Awaitable]] = None

        self.policy_mode: str = "dijkstra"  # "dijkstra" | "marl" | "static"
        self.policy_multipliers: Dict[Tuple[str, str], float] = {}
        self.signboard_actions: Dict[str, np.ndarray] = {}
        self._externally_controlled_signboards: bool = False
        self._last_policy_decision: float = -math.inf
        self._static_paths: Dict[str, List[str]] = {}
        self.active_comm_nodes: Optional[Set[str]] = None
        self.policy_model = None
        self._policy_edge_index = None
        self._policy_action_masks = None
        self._policy_hidden = None
        self._last_hazards = {}
        self._last_crowds = {}
        self._compute_static_paths()
        self._init_policy()

        self._init_agents()

    def _init_policy(self):
        try:
            import torch
            from simulator.policy.st_tba_gat import ST_TBA_GAT
            self.sorted_nodes = sorted(list(self.node_positions.keys()))
            self.node_to_idx = {nid: i for i, nid in enumerate(self.sorted_nodes)}

            self.agent_neighbors = {nid: [] for nid in self.sorted_nodes}
            for s, t, d, w in self.edge_list:
                if t not in self.agent_neighbors[s]: self.agent_neighbors[s].append(t)
                if s not in self.agent_neighbors[t]: self.agent_neighbors[t].append(s)

            edges = []
            for s, nbrs in self.agent_neighbors.items():
                u_idx = self.node_to_idx[s]
                for nbr in nbrs:
                    v_idx = self.node_to_idx[nbr]
                    edges.append([u_idx, v_idx])
            self._policy_edge_index = torch.tensor(edges, dtype=torch.long).t()
            self._policy_action_masks = torch.zeros(
                len(self.sorted_nodes), 7, dtype=torch.bool
            )
            for idx, nid in enumerate(self.sorted_nodes):
                nbrs = self.agent_neighbors[nid]
                for k in range(min(6, len(nbrs))):
                    nbr = nbrs[k]
                    is_hoist = "hoist" in nid and "hoist" in nbr
                    is_dead_end = len(self.agent_neighbors.get(nbr, [])) <= 1 and nbr not in self.exits
                    if not is_hoist and not is_dead_end:
                        self._policy_action_masks[idx, k] = True
                self._policy_action_masks[idx, 6] = True  # Standby option always valid

            directional_ckpt = os.environ.get(
                "ST_TBA_GAT_CHECKPOINT",
                os.path.join(_REPO, "checkpoints", "directional", "best_policy.pt"),
            )
            legacy_ckpt = os.path.join(_REPO, "checkpoints", "best_policy.pt")
            ckpt_path = directional_ckpt if os.path.exists(directional_ckpt) else legacy_ckpt
            if os.path.exists(ckpt_path):
                try:
                    self.policy_model = ST_TBA_GAT.load_checkpoint(ckpt_path, device="cpu")
                except Exception:
                    self.policy_model = ST_TBA_GAT(node_dim=41, hidden_dim=64, max_corridors=6)
            else:
                self.policy_model = ST_TBA_GAT(node_dim=41, hidden_dim=64, max_corridors=6)
            self.policy_model.eval()
            self._policy_hidden = torch.zeros(len(self.sorted_nodes), 64)
            self._last_hazards = {nid: 0.0 for nid in self.sorted_nodes}
            self._last_crowds = {nid: 0.0 for nid in self.sorted_nodes}
        except Exception as e:
            self.policy_model = None

    def set_policy_mode(self, mode: str):
        if mode in ("dijkstra", "marl", "static"):
            self.policy_mode = mode
            if mode == "marl":
                self._init_policy()
                self._externally_controlled_signboards = False
                self._reset_policy_memory()
                self._refresh_live_policy_actions(force=True)
            if mode == "static":
                self.policy_multipliers = {}
                self.router.update_weights({}, set())  # Reset to static geometric distance
            self._reroute_all(force=True)

    def _compute_static_paths(self):
        """Precomputes immutable blueprint shortest paths for NFPA 101 static signage."""
        import networkx as nx
        g_geom = nx.Graph()
        for s, t, d, w in self.edge_list:
            if "hoist" in s and "hoist" in t:
                continue  # Elevators/hoists are strictly excluded from static emergency exit paths
            g_geom.add_edge(s, t, weight=d)
        self._static_paths = {}
        for nid in self.node_positions:
            best_cost = math.inf
            best_p = []
            for ex in self.exits:
                try:
                    p = nx.dijkstra_path(g_geom, nid, ex, weight="weight")
                    c = nx.dijkstra_path_length(g_geom, nid, ex, weight="weight")
                    if c < best_cost:
                        best_cost = c
                        best_p = p
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            self._static_paths[nid] = best_p

    def _reset_policy_memory(self):
        """Resets recurrent inference state at scenario and controller boundaries."""
        if self.policy_model is None:
            return
        import torch
        self._policy_hidden = torch.zeros(len(self.sorted_nodes), self.policy_model.hidden_dim)
        self._last_hazards = {nid: 0.0 for nid in self.sorted_nodes}
        self._last_crowds = {nid: 0.0 for nid in self.sorted_nodes}
        self._last_policy_decision = -math.inf

    def set_signboard_actions(
        self,
        actions: Dict[str, Any],
        externally_controlled: bool = False,
    ):
        """Sets direct four-state commands for every signboard controller.

        In MARL rollouts the environment supplies an external action once per decision
        interval.  The flag keeps the GUI's local inference loop from replacing that
        sampled action before the physics has consumed it.
        """
        self.signboard_actions = {
            node_id: np.asarray(action, dtype=np.int64).copy()
            for node_id, action in actions.items()
        }
        self._externally_controlled_signboards = externally_controlled

    def get_next_waypoint(
        self,
        current_node: str,
        floor: int,
        agent_id: int = 0,
        prev_node: str = "",
    ) -> Optional[str]:
        """
        Determines the immediate next hop waypoint for a pedestrian at current_node.
        In MARL mode: Strictly queries the local signboard directional arrow.
        In Static mode: Follows fixed geometric exit signs.
        In Dijkstra mode: Follows centralized server shortest safe path.
        Evacuees strictly obey local illuminated physical signboards.
        """
        if not current_node or current_node in self.exits:
            return None

        nbrs = self.agent_neighbors.get(current_node, [])
        if not nbrs:
            return None

        # Immediate exit check: if an unblocked exit is directly adjacent, take it!
        for nbr in nbrs:
            if nbr in self.exits and self.hazard.levels.get(nbr, 0.0) < 0.80 and nbr not in self.hazard.blocked:
                return nbr

        # ── 1. MARL MODE: Local Signboard Guidance (Hop-by-Hop) ──────────
        if self.policy_mode == "marl":
            acts = self.signboard_actions.get(current_node, None)
            green_corridors = []
            amber_corridors = []
            normal_corridors = []

            for k, nbr in enumerate(nbrs[:6]):
                act_code = int(acts[k]) if (acts is not None and k < len(acts)) else 0
                hn = self.hazard.levels.get(nbr, 0.0)
                is_blocked = (
                    current_node in self.hazard.blocked
                    or nbr in self.hazard.blocked
                    or act_code == 2
                    or hn >= 0.80
                    or ("hoist" in current_node and "hoist" in nbr)  # Elevators/hoists de-energized during emergency
                )

                if is_blocked:
                    continue  # Red X: impassable barrier

                # GREEN_ARROW is the controller's direct route command.
                if act_code == 3:
                    green_corridors.append(nbr)
                elif act_code == 1:
                    amber_corridors.append(nbr)
                else:
                    normal_corridors.append(nbr)

            # 1. Prioritize designated green arrow corridors moving forward (away from prev_node)
            forward_green = [nbr for nbr in green_corridors if nbr != prev_node]
            chosen_green = forward_green if forward_green else green_corridors

            if chosen_green:
                for nbr in chosen_green:
                    if nbr in self.exits:
                        return nbr

                # Dynamic Flow Splitting:
                # If multiple green corridors are designated, partition arriving evacuees proportionally
                if len(chosen_green) > 1:
                    return chosen_green[agent_id % len(chosen_green)]

                primary = chosen_green[0]

                # Check downstream bottleneck crowding at primary corridor
                primary_cap = max(1, self.node_capacities.get(primary, 10))
                primary_occ = 0
                for p in self.agents:
                    if p.state.value in ("moving", "reacting", "danger", "rerouting"):
                        if p.current_node == primary or (p.path and p.path[0] == primary):
                            primary_occ += 1

                # If primary is jammed (density >= 0.50 of capacity) and a safe forward alternative exists:
                forward_alts = [
                    nbr for nbr in (normal_corridors + amber_corridors)
                    if nbr != prev_node and nbr != primary and self.hazard.levels.get(nbr, 0.0) < 0.30
                ]
                if forward_alts and (primary_occ / float(primary_cap)) >= 0.50:
                    # Dynamically divert 50% of incoming arrivals to the uncongested parallel corridor
                    if agent_id % 2 == 1:
                        return forward_alts[0]

                return primary

            # 2. If green arrow is absent, advance along forward open corridors
            forward_open = [nbr for nbr in (normal_corridors + amber_corridors) if nbr != prev_node]
            if forward_open:
                for nbr in forward_open:
                    if nbr in self.exits:
                        return nbr
                return forward_open[agent_id % len(forward_open)]

            # 3. Only backtrack if at an absolute dead-end with zero forward unblocked corridors
            all_open = green_corridors or normal_corridors or amber_corridors
            if all_open:
                for nbr in all_open:
                    if nbr in self.exits:
                        return nbr
                return all_open[agent_id % len(all_open)]
            return None

        # ── 2. STATIC NFPA MODE: Fixed geometric signs ───────────────────
        elif self.policy_mode == "static":
            if not self._static_paths:
                self._compute_static_paths()
            path = self._static_paths.get(current_node, [])
            return path[1] if len(path) >= 2 else None

        # ── 3. DIJKSTRA MODE: Centralized dynamic server ─────────────────
        else:
            p = self.router.get_path(current_node, floor)
            return p[1] if len(p) >= 2 else None


    # ── Control API ───────────────────────────────────────────────────────────
    def set_broadcast(self, cb): self._broadcast_cb = cb
    def play(self):  self.running = True
    def pause(self): self.running = False

    def reset(self, num_evacuees=None, cluster_node=None, cluster_ratio=0.0):
        if num_evacuees: self.num_evacuees = num_evacuees
        self.t = 0.0
        self._accumulator = 0.0
        self._last_reroute = 0.0
        self.alarm_active = False
        self.hazard.reset()
        self.router.update_weights({}, set())
        self.signboard_actions = {}
        self._externally_controlled_signboards = False
        self._reset_policy_memory()
        self._init_agents(cluster_node=cluster_node, cluster_ratio=cluster_ratio)
        if self.policy_mode == "marl":
            self._refresh_live_policy_actions(force=True)
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
        from .hazard_model import HAZARD_ALIASES
        if isinstance(htype_str, str):
            htype = HAZARD_ALIASES.get(htype_str.upper(), None)
            if htype is None:
                try:    htype = HazardType(htype_str)
                except: htype = HazardType.GAS_RELEASE
        else:
            htype = htype_str
        if node_id not in self.node_positions: return False
        self.hazard.inject(node_id, htype, min(1.0, max(0.0, intensity)))
        # Automatically trigger facility evacuation alarm upon hazard injection
        self.trigger_alarm()
        if self.policy_mode == "marl" and not self._externally_controlled_signboards:
            self._refresh_live_policy_actions(force=True)
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
        if self.policy_mode == "static":
            pass  # NFPA 101 Static signage has fixed egress paths (pure geometric distance)
        elif self.policy_mode == "marl":
            if not self._externally_controlled_signboards:
                self._refresh_live_policy_actions()
        else:
            self.router.update_weights(
                self.hazard.levels, self.hazard.blocked, active_nodes=self.active_comm_nodes
            )

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

            target_id = agent.path[0] if agent.path else None
            needs_reroute = not agent.path
            target_is_blocked = False

            if target_id:
                tgt_h = self.hazard.levels.get(target_id, 0.0)
                tgt_blk = target_id in self.hazard.blocked
                curr_blk = agent.current_node in self.hazard.blocked
                is_edge_blk = (
                    tgt_h >= 0.80
                    or tgt_blk
                    or curr_blk
                    or ("hoist" in agent.current_node and "hoist" in target_id)
                )
                if not is_edge_blk and self.policy_mode == "marl":
                    acts = self.signboard_actions.get(agent.current_node)
                    nbrs = self.agent_neighbors.get(agent.current_node, [])
                    if acts is not None and target_id in nbrs:
                        k = nbrs.index(target_id)
                        if int(acts[k]) == 2:  # Corridor explicitly blocked by safety interlock
                            is_edge_blk = True

                if is_edge_blk:
                    target_is_blocked = True
                    needs_reroute = True

            if needs_reroute:
                if target_id and target_is_blocked:
                    # Corridor ahead is impassable: smoothly reverse direction along the same corridor
                    orig_node = agent.current_node
                    if orig_node and orig_node != target_id:
                        agent.path = [orig_node]
                        agent.prev_node = target_id
                        agent.current_node = target_id
                        agent.state = PedestrianState.REROUTING
                    else:
                        src = _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                        next_hop = self.get_next_waypoint(
                            src, agent.floor, agent.id,
                            prev_node=getattr(agent, "prev_node", ""),
                        )
                        if next_hop:
                            agent.current_node = src
                            agent.path = [next_hop]
                            agent.state = PedestrianState.MOVING
                elif not agent.path:
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                    next_hop = self.get_next_waypoint(
                        src, agent.floor, agent.id,
                        prev_node=getattr(agent, "prev_node", ""),
                    )
                    if next_hop:
                        agent.current_node = src
                        agent.path = [next_hop]
                        agent.state = PedestrianState.MOVING
                    else:
                        agent.path = []
                        agent.state = PedestrianState.REROUTING

    def _update_states(self):
        for agent in self.agents:
            if agent.state in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                continue

            h = self.hazard.levels.get(agent.current_node, 0.0)

            # 1. ISO 13571:2012 Fractional Effective Dose (FED) Toxicity & Flashover
            if h > 0.02:
                # Accumulate toxic dosage: 30.0 s at C=1.0 is lethal
                agent.fed += (h / 30.0) * self.DT

            if agent.fed >= 1.0 or h >= 0.95:
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
                if not agent.path:
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                    next_hop = self.get_next_waypoint(
                        src, agent.floor, agent.id,
                        prev_node=getattr(agent, "prev_node", ""),
                    )
                    if next_hop:
                        agent.path = [next_hop]
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
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                    next_hop = self.get_next_waypoint(
                        src, agent.floor, agent.id,
                        prev_node=getattr(agent, "prev_node", ""),
                    )
                    if next_hop:
                        agent.path = [next_hop]

            elif agent.state in (PedestrianState.MOVING, PedestrianState.REROUTING, PedestrianState.DANGER):
                if not agent.path:
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                    next_hop = self.get_next_waypoint(
                        src, agent.floor, agent.id,
                        prev_node=getattr(agent, "prev_node", ""),
                    )
                    if next_hop:
                        agent.path = [next_hop]
                        if agent.state == PedestrianState.REROUTING:
                            agent.state = PedestrianState.MOVING

    def _init_agents(self, cluster_node=None, cluster_ratio=0.0):
        self.agents = spawn_pedestrians(
            self.num_evacuees, self.node_positions, self.node_floors,
            cluster_node=cluster_node, cluster_ratio=cluster_ratio)
        # Workers begin in NORMAL operating state dwelling at their stations (no exit paths initially)
        for agent in self.agents:
            agent.state = PedestrianState.NORMAL
            agent.path = []

    # ── Direct ST-TBA-GAT inference and serialisation ─────────────────────────
    def _policy_features(self):
        """Builds the same 34-feature local observation used during MAPPO training."""
        import torch

        node_counts = {nid: 0 for nid in self.sorted_nodes}
        for ped in self.agents:
            if ped.state not in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                if ped.current_node in node_counts:
                    node_counts[ped.current_node] += 1

        edge_geometry = {}
        for source, target, distance, width in self.edge_list:
            edge_geometry[(source, target)] = (distance, width)
            edge_geometry[(target, source)] = (distance, width)

        observations = []
        for node_id in self.sorted_nodes:
            capacity = max(1, self.node_capacities.get(node_id, 10))
            population = node_counts[node_id]
            density = min(2.0, population / float(capacity))
            hazard = self.hazard.levels.get(node_id, 0.0)
            features = [
                hazard,
                density,
                self.node_floors.get(node_id, 1) / 3.0,
                1.0 if "stair" in node_id else 0.0,
                1.0 if node_id in self.exits else 0.0,
                min(1.0, capacity / 50.0),
                hazard - self._last_hazards.get(node_id, 0.0),
                density - self._last_crowds.get(node_id, 0.0),
                min(1.0, self.t / 120.0),
                1.0 if self.alarm_active else 0.0,
                min(1.0, self.static_exit_dists.get(node_id, 50.0) / 100.0),
            ]
            for corridor_idx in range(6):
                neighbours = self.agent_neighbors[node_id]
                if corridor_idx >= len(neighbours):
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                neighbour = neighbours[corridor_idx]
                neighbour_capacity = max(1, self.node_capacities.get(neighbour, 10))
                neighbour_density = min(
                    2.0, node_counts.get(neighbour, 0) / float(neighbour_capacity)
                )
                distance, width = edge_geometry.get((node_id, neighbour), (10.0, 2.0))
                features.extend([
                    self.hazard.levels.get(neighbour, 0.0),
                    neighbour_density,
                    min(1.0, distance / 30.0),
                    min(1.0, width / 5.0),
                    min(1.0, self.static_exit_dists.get(neighbour, 50.0) / 100.0),
                ])
            observations.append(features)
            self._last_hazards[node_id] = hazard
            self._last_crowds[node_id] = density
        return torch.tensor(observations, dtype=torch.float32)

    def _refresh_live_policy_actions(self, force: bool = False):
        """Runs one local policy decision; it never consults a routing algorithm."""
        if (
            self.policy_mode != "marl"
            or self.policy_model is None
            or self._policy_edge_index is None
            or self._externally_controlled_signboards
            or (not force and self.t - self._last_policy_decision < self.POLICY_DECISION_DT)
        ):
            return
        try:
            import torch

            node_features = self._policy_features()
            with torch.no_grad():
                actions, _, self._policy_hidden, _ = self.policy_model.act(
                    node_features=node_features,
                    edge_index=self._policy_edge_index,
                    hidden_state=self._policy_hidden,
                    action_masks=self._policy_action_masks,
                    deterministic=True,
                )
            actions_np = actions.cpu().numpy()
            signboard_dict = {}
            for idx, node_id in enumerate(self.sorted_nodes):
                act_k = int(actions_np[idx]) if actions_np.ndim == 1 else int(actions_np[idx, 0])
                nbrs = self.agent_neighbors.get(node_id, [])
                phys_act = np.zeros(6, dtype=np.int64)
                loc_h = self.hazard.levels.get(node_id, 0.0)

                arrow_idx = None
                if act_k < len(nbrs):
                    nbr = nbrs[act_k]
                    nbr_h = self.hazard.levels.get(nbr, 0.0)
                    is_dead_end = len(self.agent_neighbors.get(nbr, [])) <= 1 and nbr not in self.exits
                    is_hoist = "hoist" in node_id and "hoist" in nbr
                    if loc_h < 0.80 and nbr_h < 0.80 and not is_dead_end and not is_hoist:
                        arrow_idx = act_k

                # When alarm is active, ensure every controller shows an authoritative exit route
                if self.alarm_active and arrow_idx is None and node_id not in self.exits:
                    safe_nbrs = []
                    for c_idx, neighbour in enumerate(nbrs[:6]):
                        nbr_h = self.hazard.levels.get(neighbour, 0.0)
                        is_dead_end = len(self.agent_neighbors.get(neighbour, [])) <= 1 and neighbour not in self.exits
                        is_hoist = "hoist" in node_id and "hoist" in neighbour
                        if loc_h < 0.80 and nbr_h < 0.80 and not is_dead_end and not is_hoist:
                            safe_nbrs.append((c_idx, neighbour))
                    if safe_nbrs:
                        best_c_idx, _ = min(safe_nbrs, key=lambda pair: self.static_exit_dists.get(pair[1], 999.0))
                        arrow_idx = best_c_idx

                arrow_indices = set()
                if arrow_idx is not None:
                    arrow_indices.add(arrow_idx)

                # Density-aware secondary flow-splitting arrow:
                # If primary corridor is crowded (density >= 40%), illuminate an alternative safe forward corridor
                if arrow_idx is not None and len(nbrs) > 1:
                    primary_nbr = nbrs[arrow_idx]
                    p_occ = sum(1 for p in self.agents if p.current_node == primary_nbr and p.state.value in ("moving", "reacting", "danger", "rerouting"))
                    p_cap = max(1, self.node_capacities.get(primary_nbr, 10))
                    if (p_occ / float(p_cap)) >= 0.40:
                        for c_idx, neighbour in enumerate(nbrs[:6]):
                            if c_idx != arrow_idx:
                                nbr_h = self.hazard.levels.get(neighbour, 0.0)
                                is_dead_end = len(self.agent_neighbors.get(neighbour, [])) <= 1 and neighbour not in self.exits
                                is_hoist = "hoist" in node_id and "hoist" in neighbour
                                if loc_h < 0.80 and nbr_h < 0.30 and not is_dead_end and not is_hoist:
                                    arrow_indices.add(c_idx)
                                    break

                for c_idx, neighbour in enumerate(nbrs[:6]):
                    nbr_h = self.hazard.levels.get(neighbour, 0.0)
                    is_hoist = "hoist" in node_id and "hoist" in neighbour
                    if loc_h >= 0.80 or nbr_h >= 0.80 or is_hoist:
                        phys_act[c_idx] = 2  # BLOCKED
                    elif c_idx in arrow_indices:
                        phys_act[c_idx] = 3  # ARROW
                    elif nbr_h >= 0.30:
                        phys_act[c_idx] = 1  # CAUTION (moderate smoke/hazard)
                    else:
                        phys_act[c_idx] = 0  # NORMAL
                signboard_dict[node_id] = phys_act

            self.set_signboard_actions(signboard_dict)
            self._last_policy_decision = self.t
            self.policy_inference_error = None
        except Exception as error:
            # Preserve the last known-safe signboard command if inference fails.
            self.policy_inference_error = str(error)

    def _marl_signboards(self) -> Dict[str, List[Dict]]:
        """Serialises cached four-state actions without changing the policy decision."""
        signboards: Dict[str, List[Dict]] = {}
        for node_id in self.sorted_nodes:
            node_signs = []
            upstream_hazard = self.hazard.levels.get(node_id, 0.0)
            actions = self.signboard_actions.get(node_id, np.zeros(6, dtype=np.int64))
            for corridor_idx, neighbour in enumerate(self.agent_neighbors.get(node_id, [])[:6]):
                downstream_hazard = self.hazard.levels.get(neighbour, 0.0)
                action_code = int(actions[corridor_idx]) if corridor_idx < len(actions) else 0
                if (
                    node_id in self.hazard.blocked
                    or neighbour in self.hazard.blocked
                    or upstream_hazard >= 0.80
                    or downstream_hazard >= 0.80
                    or action_code == 2
                    or ("hoist" in node_id and "hoist" in neighbour)
                ):
                    action = "BLOCKED"
                elif action_code == 1 or downstream_hazard >= 0.30:
                    action = "CAUTION"
                elif action_code == 3:
                    action = "ARROW"
                else:
                    action = "NORMAL"
                node_signs.append({
                    "target": neighbour,
                    "action": action,
                    "hazard": round(max(upstream_hazard, downstream_hazard), 2),
                })
            if node_signs:
                signboards[node_id] = node_signs
        return signboards

    def compute_signboards(self) -> Dict[str, List[Dict]]:
        """Returns signboard state for the active policy without hidden rerouting."""
        if self.policy_mode == "marl":
            if not self.signboard_actions and not self._externally_controlled_signboards:
                self._refresh_live_policy_actions(force=True)
            return self._marl_signboards()

        if self.router.G is None:
            return {}

        # Dijkstra remains available only as the explicitly selected centralized
        # benchmark.  It is not part of ST-TBA-GAT sign selection or movement.
        import networkx as nx
        adjacency = {}
        for source, target, _distance, _width in self.edge_list:
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        try:
            exit_paths = nx.multi_source_dijkstra_path(
                self.router.G, self.exits, weight="weight"
            )
        except Exception:
            exit_paths = {}

        signboards: Dict[str, List[Dict]] = {}
        for node_id in self.node_positions:
            neighbours = sorted(adjacency.get(node_id, set()))
            if not neighbours:
                continue
            next_hop = None
            if node_id in exit_paths and len(exit_paths[node_id]) >= 2:
                next_hop = exit_paths[node_id][-2]
            upstream_hazard = self.hazard.levels.get(node_id, 0.0)
            node_signs = []
            for neighbour in neighbours:
                downstream_hazard = self.hazard.levels.get(neighbour, 0.0)
                combined_hazard = max(upstream_hazard, downstream_hazard)
                if (
                    node_id in self.hazard.blocked
                    or neighbour in self.hazard.blocked
                    or combined_hazard >= 0.80
                ):
                    action = "BLOCKED"
                elif combined_hazard >= 0.30:
                    action = "CAUTION"
                elif self.alarm_active and neighbour == next_hop:
                    action = "ARROW"
                elif not self.alarm_active and neighbour == next_hop:
                    action = "NORMAL_ARROW"
                else:
                    action = "NORMAL"
                node_signs.append({
                    "target": neighbour,
                    "action": action,
                    "hazard": round(combined_hazard, 2),
                })
            signboards[node_id] = node_signs
        return signboards

    def state_dict(self):
        return {
            "t":            round(self.t, 1),
            "running":      self.running,
            "alarm_active": self.alarm_active,
            "policy_mode":  self.policy_mode,
            "speed":        self.speed,
            "pedestrians":  [a.to_dict() for a in self.agents],
            "hazards":      self.hazard.to_dict(),
            "signboards":   self.compute_signboards(),
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


def _nearest_node(x, y, positions, node_floors=None, floor=1):
    best, best_d = None, math.inf
    for nid, (nx_, ny_) in positions.items():
        if node_floors and node_floors.get(nid, 1) != floor:
            continue
        d = math.hypot(x - nx_, y - ny_)
        if d < best_d:
            best_d = d; best = nid
    return best or ""
