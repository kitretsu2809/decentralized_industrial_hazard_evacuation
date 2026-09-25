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

        self.hazard  = HazardModel(adj_with_dist, exits=self.exits)
        self.router  = DijkstraRouter()
        self.router.build(self.edge_list, self.exits, self.node_floors)
        self.sfm     = SocialForceModel(self.node_positions, self.node_floors, self.edge_list)

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
        self.active_comm_nodes: Optional[Set[str]] = None
        self.policy_model = None
        self._policy_edge_index = None
        self._policy_hidden = None
        self._last_hazards = {}
        self._last_crowds = {}
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

            ckpt_path = os.path.join(_REPO, "checkpoints", "best_policy.pt")
            if os.path.exists(ckpt_path):
                self.policy_model = ST_TBA_GAT.load_checkpoint(ckpt_path, device="cpu")
            else:
                self.policy_model = ST_TBA_GAT(node_dim=34, hidden_dim=64, max_corridors=6)
            self.policy_model.eval()
            self._policy_hidden = torch.zeros(len(self.sorted_nodes), 64)
            self._last_hazards = {nid: 0.0 for nid in self.sorted_nodes}
            self._last_crowds = {nid: 0.0 for nid in self.sorted_nodes}
        except Exception as e:
            self.policy_model = None

    def set_policy_mode(self, mode: str):
        if mode in ("dijkstra", "marl", "static"):
            self.policy_mode = mode
            if mode == "marl" and self.policy_model is None:
                self._init_policy()
            if mode == "static":
                self.policy_multipliers = {}
                self.router.update_weights({}, set())  # Reset to static geometric distance
            self._reroute_all(force=True)

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
            self.router.update_weights(
                self.hazard.levels, self.hazard.blocked, self.policy_multipliers
            )
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
        if self.policy_mode == "static":
            # Static NFPA signage has no environmental awareness; pedestrians cannot reroute around hazards
            for agent in self.agents:
                if not agent.path and agent.state in (PedestrianState.MOVING, PedestrianState.REACTING, PedestrianState.DANGER):
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                    p = self.router.get_path(src, agent.floor)
                    if p: agent.path = p[1:]
            return

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
                target_id = agent.path[0] if agent.path else None
                target_h = self.hazard.levels.get(target_id, 0.0) if target_id else 1.0
                target_blocked = (target_id in self.hazard.blocked) if target_id else True

                # If forward waypoint target_id is safe and unblocked, continue forward through target_id
                if target_id and target_h < 0.4 and not target_blocked:
                    new_path = self.router.get_path(target_id, agent.floor)
                    if new_path:
                        agent.path = new_path
                else:
                    # Forward waypoint is dangerous or blocked: retreat to current node or nearest safe node on current floor
                    curr_h = self.hazard.levels.get(agent.current_node, 0.0)
                    curr_blocked = agent.current_node in self.hazard.blocked
                    if agent.current_node and curr_h < 0.4 and not curr_blocked:
                        src = agent.current_node
                    else:
                        src = _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                    new_path = self.router.get_path(src, agent.floor)
                    if new_path:
                        agent.path = new_path[1:]

                if agent.state == PedestrianState.MOVING and needs and force:
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
                if self.policy_mode != "static":
                    if not agent.path or self.router.is_blocked_path(agent.path, self.hazard.blocked):
                        src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
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
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
                    p = self.router.get_path(src, agent.floor)
                    if p: agent.path = p[1:]

            elif agent.state in (PedestrianState.MOVING, PedestrianState.REROUTING):
                if not agent.path:
                    src = agent.current_node or _nearest_node(agent.x, agent.y, self.node_positions, self.node_floors, agent.floor)
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
    def compute_signboards(self) -> Dict[str, List[Dict]]:
        """
        Computes the dynamic directional signage state for each edge router node.
        Supports both:
          - 'marl': ST-TBA-GAT neural policy edge guidance.
          - 'dijkstra': Classical shortest safe path.
        """
        if self.router.G is None:
            return {}

        import networkx as nx
        adj = {}
        for s, t, d, w in self.edge_list:
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)

        # ── 1. ST-TBA-GAT Neural MARL Policy ─────────────────────────────────
        if self.policy_mode == "marl" and self.policy_model is not None and self._policy_edge_index is not None:
            try:
                import torch
                node_counts = {nid: 0 for nid in self.sorted_nodes}
                for ped in self.agents:
                    if ped.state not in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                        if ped.current_node in node_counts:
                            node_counts[ped.current_node] += 1

                obs_list = []
                for nid in self.sorted_nodes:
                    cap = max(1, self.node_capacities.get(nid, 10))
                    pop = node_counts.get(nid, 0)
                    rho = min(2.0, pop / float(cap))
                    h = self.hazard.levels.get(nid, 0.0)
                    floor = self.node_floors.get(nid, 1) / 3.0
                    is_stair = 1.0 if "stair" in nid else 0.0
                    is_exit = 1.0 if nid in self.exits else 0.0
                    cap_norm = min(1.0, cap / 50.0)
                    delta_h = h - self._last_hazards.get(nid, 0.0)
                    delta_rho = rho - self._last_crowds.get(nid, 0.0)
                    time_ratio = min(1.0, self.t / 120.0)
                    alarm_flag = 1.0 if self.alarm_active else 0.0

                    vec = [h, rho, floor, is_stair, is_exit, cap_norm, delta_h, delta_rho, time_ratio, alarm_flag]
                    nbrs = self.agent_neighbors.get(nid, [])
                    for k in range(6):
                        if k < len(nbrs):
                            nbr = nbrs[k]
                            n_h = self.hazard.levels.get(nbr, 0.0)
                            n_pop = node_counts.get(nbr, 0)
                            n_cap = max(1, self.node_capacities.get(nbr, 10))
                            n_rho = min(2.0, n_pop / float(n_cap))
                            dist, width = 10.0, 2.0
                            for s, t, d, w in self.edge_list:
                                if (s == nid and t == nbr) or (t == nid and s == nbr):
                                    dist, width = d, w
                                    break
                            vec.extend([n_h, n_rho, min(1.0, dist / 30.0), min(1.0, width / 5.0)])
                        else:
                            vec.extend([0.0, 0.0, 0.0, 0.0])
                    obs_list.append(vec)
                    self._last_hazards[nid] = h
                    self._last_crowds[nid] = rho

                nf = torch.tensor(obs_list, dtype=torch.float32)
                with torch.no_grad():
                    actions_t, _, self._policy_hidden, _ = self.policy_model.act(
                        node_features=nf,
                        edge_index=self._policy_edge_index,
                        hidden_state=self._policy_hidden,
                        deterministic=True,
                    )
                actions_np = actions_t.cpu().numpy()

                # Apply neural edge weights to graph
                for i, u in enumerate(self.sorted_nodes):
                    nbrs = self.agent_neighbors.get(u, [])
                    for k, v in enumerate(nbrs[:6]):
                        act_code = int(actions_np[i, k])
                        cost = 1.0 if act_code == 0 else (5.0 if act_code == 1 else 999.0)
                        if self.router.G.has_edge(u, v):
                            bw = self.router.G[u][v].get("base_weight", 1.0)
                            self.router.G[u][v]["weight"] = bw * cost

                try:
                    exit_paths = nx.multi_source_dijkstra_path(
                        self.router.G, self.exits, weight="weight"
                    )
                except Exception:
                    exit_paths = {}

                signboards: Dict[str, List[Dict]] = {}
                for i, u in enumerate(self.sorted_nodes):
                    nbrs = self.agent_neighbors.get(u, [])
                    if not nbrs:
                        continue
                    next_hop = None
                    if u in exit_paths and len(exit_paths[u]) >= 2:
                        next_hop = exit_paths[u][-2]
                    hu = self.hazard.levels.get(u, 0.0)
                    u_blk = u in self.hazard.blocked

                    node_signs = []
                    for k, v in enumerate(nbrs[:6]):
                        hv = self.hazard.levels.get(v, 0.0)
                        he = max(hu, hv)
                        act_code = int(actions_np[i, k])
                        if u_blk or (v in self.hazard.blocked) or act_code == 2 or he >= 0.80:
                            action = "BLOCKED"
                        elif act_code == 1 or he >= 0.30:
                            action = "CAUTION"
                        elif self.alarm_active and v == next_hop:
                            action = "ARROW"
                        elif not self.alarm_active and v == next_hop:
                            action = "NORMAL_ARROW"
                        else:
                            action = "NORMAL"

                        node_signs.append({
                            "target": v,
                            "action": action,
                            "hazard": round(he, 2),
                        })
                    signboards[u] = node_signs
                return signboards
            except Exception:
                pass

        # ── 2. Classical Dijkstra Baseline ───────────────────────────────────
        try:
            exit_paths = nx.multi_source_dijkstra_path(
                self.router.G, self.exits, weight="weight"
            )
        except Exception:
            exit_paths = {}

        signboards: Dict[str, List[Dict]] = {}
        for u in self.node_positions:
            nbrs = adj.get(u, set())
            if not nbrs:
                continue

            next_hop = None
            if u in exit_paths and len(exit_paths[u]) >= 2:
                next_hop = exit_paths[u][-2]

            hu = self.hazard.levels.get(u, 0.0)
            u_blk = u in self.hazard.blocked

            node_signs = []
            for v in nbrs:
                hv = self.hazard.levels.get(v, 0.0)
                he = max(hu, hv)
                v_blk = v in self.hazard.blocked
                is_blocked = u_blk or v_blk or (he >= 0.80)
                is_caution = (he >= 0.30)

                if is_blocked:
                    action = "BLOCKED"
                elif is_caution:
                    action = "CAUTION"
                elif self.alarm_active and v == next_hop:
                    action = "ARROW"
                elif not self.alarm_active and v == next_hop:
                    action = "NORMAL_ARROW"
                else:
                    action = "NORMAL"

                node_signs.append({
                    "target": v,
                    "action": action,
                    "hazard": round(he, 2)
                })

            signboards[u] = node_signs

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
