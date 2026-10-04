"""
PettingZoo ParallelEnv Multi-Agent Industrial Evacuation Environment.

Standards-compliant gymnasium/PettingZoo ParallelEnv decoupling physics from rendering:
1. Crowd dynamics governed by Weidmann Fundamental Diagram & Helbing Doorway Bottleneck Arching Breakdown:
       Q(t) = Q_nominal * max(0.10, 1.0 - beta * (rho_u - rho_crit)^2)
2. Compressive Crowd Asphyxia (Crush Stress Index):
       CSI_i(t) = integral max(0, rho_local(tau) - 4.5) dtau >= 15.0
3. Toxic hazard dispersion governed by Graph Laplacian advection-diffusion with positive-pressure exit havens.
4. Life-safety tenability governed by ISO 13571 Fractional Effective Dose (FED >= 1.0).
5. Stochastic human compliance to dynamic IoT signage:
       P(comply_i) = clip(1.0 - 0.4 * rho_local - 0.5 * C_local, 0.20, 1.0)
6. Guaranteed cycle-free acyclic egress kinematics with elevator emergency lockout.
"""

import functools
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv

from envs.fundamental_diagram import FundamentalDiagram, WeidmannParams
from envs.hazard_simulator import HazardSimulator
from envs.comm_channel import DegradedCommChannel
from sim.services.environment.src.floorplan_generator import generate_industrial_plant
from core.graph.types import NodeType


class PedestrianState:
    """Tracks individual pedestrian kinematics, compliance, and physiological tenability."""
    def __init__(self, ped_id: int, current_node: int, initial_node: int):
        self.ped_id = ped_id
        self.current_node = current_node
        self.prev_node: Optional[int] = None
        self.target_node: Optional[int] = None
        self.edge_progress: float = 0.0
        self.edge_length: float = 0.0
        self.speed: float = 1.34
        self.fed: float = 0.0
        self.csi: float = 0.0
        self.is_evacuated: bool = False
        self.is_casualty: bool = False
        self.casualty_cause: Optional[str] = None  # "crush" or "toxic"
        self.travel_time: float = 0.0
        self.wait_time: float = 0.0
        self.reroute_count: int = 0
        self.is_compliant: bool = True


class EvacuationParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "evacuation_parallel_v1"}

    def __init__(
        self,
        building=None,
        num_pedestrians: int = 250,
        max_steps: int = 180,
        dt: float = 1.0,
        max_corridors: int = 6,
        comm_radius: float = 18.0,
        packet_loss_rate: float = 0.15,
        render_mode: Optional[str] = None,
        seed: Optional[int] = 42
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.max_steps = max_steps
        self.num_pedestrians = num_pedestrians
        self.max_corridors = max_corridors
        self.rng = np.random.default_rng(seed)

        if building is None:
            self.building = generate_industrial_plant()
        else:
            self.building = building

        self._build_networkx_graph()

        self.fd = FundamentalDiagram(WeidmannParams(
            v0=1.34,
            rho_jam=5.4,
            gamma=1.913,
            q_specific=1.33,
            min_velocity=0.05,
            rho_crit=3.5,
            beta_arching=0.25,
            csi_threshold=4.5,
            csi_fatal=15.0
        ))
        self.hazard_sim = HazardSimulator(
            num_nodes=self.num_nodes,
            adj_matrix=self.adj_matrix,
            diffusion_coeff=0.15,
            decay_rate=0.012,
            dt=self.dt,
            fed_threshold=60.0,
            critical_hazard_cutoff=0.60,
            exit_indices=self.exit_indices
        )
        self.comm_channel = DegradedCommChannel(
            comm_radius=comm_radius,
            packet_loss_rate=packet_loss_rate,
            critical_hazard_cutoff=0.60,
            seed=seed
        )

        self.agents = [f"node_{i}" for i in range(self.num_nodes) if i not in self.exit_indices]
        self.possible_agents = list(self.agents)

        self._action_spaces = {
            agent: spaces.Discrete(self.max_corridors) for agent in self.possible_agents
        }

        self.obs_dim = (1 + self.max_corridors) * 5 + self.max_corridors * 3 + self.max_corridors
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.pedestrians: List[PedestrianState] = []
        self.current_step = 0
        self.total_evacuated = 0
        self.total_casualties = 0
        self.total_crush_casualties = 0
        self.total_toxic_casualties = 0

    def _build_networkx_graph(self):
        self.G = nx.Graph()
        self.node_id_to_idx: Dict[str, int] = {}
        self.idx_to_node_id: Dict[int, str] = {}
        self.node_positions: Dict[int, Tuple[float, float, float]] = {}
        self.node_capacities: Dict[int, float] = {}
        self.node_types: Dict[int, NodeType] = {}
        self.exit_indices: Set[int] = set()
        self.stairwell_indices: Set[int] = set()
        self.elevator_indices: Set[int] = set()

        idx = 0
        for floor_num, floor in self.building.floors.items():
            for node_id, node in floor.nodes.items():
                self.node_id_to_idx[node_id] = idx
                self.idx_to_node_id[idx] = node_id
                self.node_positions[idx] = (node.position[0], node.position[1], float(floor_num))
                self.node_capacities[idx] = float(node.capacity)
                self.node_types[idx] = node.type

                if node.type == NodeType.EXIT:
                    self.exit_indices.add(idx)
                elif node.type == NodeType.STAIRWELL:
                    self.stairwell_indices.add(idx)
                elif node.type == NodeType.ELEVATOR:
                    self.elevator_indices.add(idx)

                self.G.add_node(idx, floor=floor_num, type=node.type.value)
                idx += 1

        self.num_nodes = idx
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.edge_info: Dict[Tuple[int, int], Dict[str, float]] = {}

        # Intra-floor edges
        for floor_num, floor in self.building.floors.items():
            for edge in floor.edges:
                u = self.node_id_to_idx[edge.source]
                v = self.node_id_to_idx[edge.target]
                self.adj_matrix[u, v] = 1.0
                self.adj_matrix[v, u] = 1.0
                self.G.add_edge(u, v, length=edge.distance, width=edge.width, throughput=edge.max_throughput)
                self.edge_info[(u, v)] = {'length': edge.distance, 'width': edge.width, 'throughput': edge.max_throughput}
                self.edge_info[(v, u)] = {'length': edge.distance, 'width': edge.width, 'throughput': edge.max_throughput}

        # Inter-floor vertical edges (stairs/elevators)
        for edge in self.building.cross_floor_edges:
            u = self.node_id_to_idx[edge.source]
            v = self.node_id_to_idx[edge.target]
            # NFPA 101 emergency protocol: Elevators are strictly prohibited during fire/toxic events
            if u in self.elevator_indices or v in self.elevator_indices:
                continue

            self.adj_matrix[u, v] = 1.0
            self.adj_matrix[v, u] = 1.0
            self.G.add_edge(u, v, length=edge.distance, width=edge.width, throughput=edge.max_throughput)
            self.edge_info[(u, v)] = {'length': edge.distance, 'width': edge.width, 'throughput': edge.max_throughput}
            self.edge_info[(v, u)] = {'length': edge.distance, 'width': edge.width, 'throughput': edge.max_throughput}

        self.neighbors: Dict[int, List[int]] = {
            i: sorted(list(self.G.neighbors(i))) for i in range(self.num_nodes)
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.comm_channel.set_seed(seed)

        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.total_evacuated = 0
        self.total_casualties = 0
        self.total_crush_casualties = 0
        self.total_toxic_casualties = 0

        self.hazard_sim.reset()

        crit_indices = [
            i for i in range(self.num_nodes)
            if self.node_types[i] in (NodeType.ROOM, NodeType.CORRIDOR, NodeType.INTERSECTION)
            and i not in self.exit_indices
        ]
        if not crit_indices:
            crit_indices = [i for i in range(self.num_nodes) if i not in self.exit_indices]

        disaster_nodes = options.get('disaster_nodes', None) if options else None
        if disaster_nodes is None:
            source = int(self.rng.choice(crit_indices))
            disaster_nodes = [source]

        self.hazard_sim.clear_sources()
        for src in disaster_nodes:
            self.hazard_sim.set_source(src, injection_rate=0.35)
            self.hazard_sim.concentration[src] = 0.70

        num_peds = options.get('num_pedestrians', self.num_pedestrians) if options else self.num_pedestrians
        self.num_pedestrians = num_peds

        self.pedestrians.clear()
        spawnable_nodes = [
            i for i in range(self.num_nodes)
            if i not in self.exit_indices and i not in disaster_nodes and i not in self.elevator_indices
        ]

        for p_id in range(self.num_pedestrians):
            start_node = int(self.rng.choice(spawnable_nodes))
            ped = PedestrianState(ped_id=p_id, current_node=start_node, initial_node=start_node)
            self.pedestrians.append(ped)

        obs = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def _get_node_features(self) -> np.ndarray:
        feats = np.zeros((self.num_nodes, 5), dtype=np.float32)
        node_counts = np.zeros(self.num_nodes, dtype=np.float32)
        for p in self.pedestrians:
            if not p.is_evacuated and not p.is_casualty:
                node_counts[p.current_node] += 1.0

        for i in range(self.num_nodes):
            c_haz = float(self.hazard_sim.concentration[i])
            cap = self.node_capacities[i]
            area = max(1.0, cap * 1.5)
            rho = float(node_counts[i] / area)
            is_exit = 1.0 if i in self.exit_indices else 0.0
            is_stair = 1.0 if i in self.stairwell_indices else 0.0
            cap_ratio = float(node_counts[i] / cap)
            feats[i] = [c_haz, rho, is_exit, is_stair, cap_ratio]

        return feats

    def _get_observation(self, agent: str, all_node_feats: np.ndarray) -> np.ndarray:
        node_idx = int(agent.split('_')[1])
        nbrs = self.neighbors[node_idx]

        local_node_feats = np.zeros((1 + self.max_corridors, 5), dtype=np.float32)
        local_node_feats[0] = all_node_feats[node_idx]

        local_edge_feats = np.zeros((self.max_corridors, 3), dtype=np.float32)
        action_mask = np.zeros(self.max_corridors, dtype=np.float32)

        for slot, nbr in enumerate(nbrs[:self.max_corridors]):
            local_node_feats[1 + slot] = all_node_feats[nbr]
            edge_data = self.edge_info.get((node_idx, nbr), {'length': 10.0, 'width': 2.0, 'throughput': 10.0})
            active_peds = sum(1 for p in self.pedestrians if p.current_node == node_idx and p.target_node == nbr)
            local_edge_feats[slot] = [edge_data['length'], edge_data['width'], float(active_peds)]
            if edge_data['throughput'] > 0:
                action_mask[slot] = 1.0

        obs_flat = np.concatenate([
            local_node_feats.flatten(),
            local_edge_feats.flatten(),
            action_mask
        ]).astype(np.float32)
        return obs_flat

    def _get_all_observations(self) -> Dict[str, np.ndarray]:
        all_node_feats = self._get_node_features()
        return {agent: self._get_observation(agent, all_node_feats) for agent in self.agents}

    def state(self) -> np.ndarray:
        node_feats = self._get_node_features()
        global_summary = np.array([
            float(self.total_evacuated),
            float(self.total_casualties),
            float(self.current_step),
            float(np.mean(self.hazard_sim.concentration))
        ], dtype=np.float32)
        return np.concatenate([node_feats.flatten(), global_summary])

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        self.current_step += 1

        # 1. Update Hazard Physics via Graph Laplacian Diffusion
        hazard_conc = self.hazard_sim.step()

        # 2. Build dynamic routing graph combining hazard inflation, router guidance, and egress potential
        router_target: Dict[int, int] = {}
        for agent, action_idx in actions.items():
            node_idx = int(agent.split('_')[1])
            nbrs = self.neighbors[node_idx]
            if 0 <= action_idx < len(nbrs):
                router_target[node_idx] = nbrs[action_idx]

        G_dyn = nx.DiGraph()
        G_static = nx.DiGraph()

        for (u, v), info in self.edge_info.items():
            if info.get('throughput', 1.0) <= 0:
                continue
            L = info.get('length', 10.0)
            c_u = float(hazard_conc[u]) if u < len(hazard_conc) else 0.0
            c_v = float(hazard_conc[v]) if v < len(hazard_conc) else 0.0
            max_c = max(c_u, c_v)

            # Reflexive safety: if corridor enters toxic zone, mark impassable
            if max_c >= 0.60:
                cost = 1e8
            else:
                cost = L * (1.0 + 20.0 * (max_c ** 1.5))

            # Apply router guidance preference (incentivize designated edge)
            if u in router_target and router_target[u] == v:
                cost *= 0.65

            G_dyn.add_edge(u, v, weight=cost)
            G_static.add_edge(u, v, weight=L)

        try:
            dist_to_exit = nx.multi_source_dijkstra_path_length(
                G_dyn.reverse(), sources=list(self.exit_indices), weight='weight'
            )
        except Exception:
            dist_to_exit = {}

        try:
            static_dist_to_exit = nx.multi_source_dijkstra_path_length(
                G_static.reverse(), sources=list(self.exit_indices), weight='weight'
            )
        except Exception:
            static_dist_to_exit = {}

        # 3. Node occupancies & density calculation
        node_occupancy = np.zeros(self.num_nodes, dtype=np.float64)
        for p in self.pedestrians:
            if not p.is_evacuated and not p.is_casualty:
                node_occupancy[p.current_node] += 1.0

        step_evacuated = 0
        step_casualties = 0
        step_crush = 0
        step_toxic = 0
        total_delay = 0.0

        # 4. Advance Pedestrian Traversal along Weidmann Fundamental Diagram & Helbing Arching
        for p in self.pedestrians:
            if p.is_evacuated or p.is_casualty:
                continue

            u = p.current_node
            local_conc = hazard_conc[u]
            cap_u = max(1.0, self.node_capacities.get(u, 10.0))
            area_u = max(1.0, cap_u * 1.5)
            rho_local = float(node_occupancy[u] / area_u)

            # --- DUAL MORTALITY CHECK ---
            # A. Toxic Inhalation (ISO 13571 FED)
            fed_inc = self.hazard_sim.compute_fed_delta(local_conc)
            p.fed += fed_inc

            # B. Compressive Crowd Asphyxia (Crush Stress Index)
            p.csi, is_crush_fatal = self.fd.update_csi(p.csi, rho_local, self.dt)

            if p.fed >= 1.0:
                p.is_casualty = True
                p.casualty_cause = "toxic"
                self.total_casualties += 1
                self.total_toxic_casualties += 1
                step_casualties += 1
                step_toxic += 1
                continue

            if is_crush_fatal:
                p.is_casualty = True
                p.casualty_cause = "crush"
                self.total_casualties += 1
                self.total_crush_casualties += 1
                step_casualties += 1
                step_crush += 1
                continue

            if u in self.exit_indices:
                p.is_evacuated = True
                self.total_evacuated += 1
                step_evacuated += 1
                continue

            p.travel_time += self.dt

            # --- STOCHASTIC HUMAN COMPLIANCE ---
            p_comply = float(np.clip(1.0 - 0.4 * rho_local - 0.5 * local_conc, 0.20, 1.0))
            ped_obeys = bool(self.rng.random() < p_comply)
            p.is_compliant = ped_obeys

            # If pedestrian is choosing next target node at junction u:
            if p.target_node is None or p.target_node == u:
                valid_nbrs = [n for n in self.neighbors[u] if (u, n) in G_dyn.edges]
                if not valid_nbrs:
                    continue

                if ped_obeys:
                    # Dynamic guided egress descent towards nearest safe exit
                    best_next = min(valid_nbrs, key=lambda v: G_dyn[u][v]['weight'] + dist_to_exit.get(v, 1e8))
                else:
                    # Panic herd behavior: greedy uncoordinated shortest path to exit
                    best_next = min(valid_nbrs, key=lambda v: G_static[u][v]['weight'] + static_dist_to_exit.get(v, 1e8))

                p.target_node = best_next
                e_info = self.edge_info.get((u, best_next), {'length': 10.0, 'width': 2.0, 'throughput': 10.0})
                p.edge_length = e_info['length']
                p.edge_progress = 0.0

            # Advance kinematics along corridor (u -> target_node)
            v_node = p.target_node
            e_info = self.edge_info.get((u, v_node), {'length': 10.0, 'width': 2.0, 'throughput': 10.0})
            corridor_width = e_info['width']
            edge_cap = e_info['throughput']
            length = e_info['length']

            edge_area = max(1.0, length * corridor_width)
            edge_density = min(5.4, node_occupancy[u] / edge_area)

            # Helbing doorway bottleneck arching breakdown at constrictions
            _, effective_speed = self.fd.helbing_bottleneck_flow(
                nominal_capacity=edge_cap,
                density=edge_density,
                doorway_width=corridor_width
            )
            p.speed = max(0.05, effective_speed)
            p.edge_progress += p.speed * self.dt

            if p.edge_progress >= length:
                p.prev_node = u
                p.current_node = v_node
                p.target_node = None
                p.edge_progress = 0.0

                if v_node in self.exit_indices:
                    p.is_evacuated = True
                    self.total_evacuated += 1
                    step_evacuated += 1
            else:
                total_delay += 0.05

        # 5. Reward Calculation with Crowd Density Balancing (Claim 9)
        active_peds = self.num_pedestrians - (self.total_evacuated + self.total_casualties)
        
        # Density variance penalty across parallel corridors
        active_node_counts = [node_occupancy[i] for i in range(self.num_nodes) if i not in self.exit_indices]
        density_var = float(np.var(active_node_counts)) if active_node_counts else 0.0

        step_reward = (
            step_evacuated * 25.0
            - step_casualties * 100.0
            - 0.05 * density_var
            - active_peds * 0.01
            - total_delay * 0.1
        )
        rewards = {agent: float(step_reward) for agent in self.agents}

        # 6. Termination / Truncation Check
        all_resolved = (self.total_evacuated + self.total_casualties) >= self.num_pedestrians
        truncated = self.current_step >= self.max_steps
        terminated = all_resolved

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        if terminated or truncated:
            self.agents = []

        obs = self._get_all_observations() if not (terminated or truncated) else {}
        infos = {
            agent: {
                'total_evacuated': self.total_evacuated,
                'total_casualties': self.total_casualties,
                'total_crush_casualties': self.total_crush_casualties,
                'total_toxic_casualties': self.total_toxic_casualties,
                'active_pedestrians': active_peds,
                'step': self.current_step
            } for agent in self.possible_agents
        }

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "human":
            active = self.num_pedestrians - (self.total_evacuated + self.total_casualties)
            print(f"[Step {self.current_step:03d}] Evacuated: {self.total_evacuated}/{self.num_pedestrians} | "
                  f"Casualties: {self.total_casualties} (Crush: {self.total_crush_casualties}, Toxic: {self.total_toxic_casualties}) | "
                  f"Active: {active}")

    def close(self):
        pass
