"""
PettingZoo Multi-Agent Environment: IndustrialEvacuationEnv.
Wraps the continuous Helbing Social Force Model physics engine into a
standardized parallel Multi-Agent Gymnasium interface.
Each edge router node in the 36-node industrial chemical plant is an agent.
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import networkx as nx
import gymnasium
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv

from simulator.engine.simulation import Simulation
from simulator.engine.pedestrian import PedestrianState
from simulator.engine.hazard_model import HazardType


class IndustrialEvacuationEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for 36-node edge router network.
    Observation space: local sensor features + 4-byte gossip + incident corridor states.
    Action space: MultiDiscrete([4] * max_corridors) for incident directional signage:
      0: NORMAL (no active direction)
      1: CAUTION (amber detour)
      2: BLOCKED (red X safety blockade)
      3: GREEN ARROW (the policy's direct directional instruction)
    """
    metadata = {"name": "industrial_evacuation_v1", "render_modes": ["human"]}

    MAX_CORRIDORS: int = 6
    DECISION_DT: float = 2.5  # Decision interval: policy steps every 2.5 sim-seconds (25 SFM ticks)
    MAX_SIM_TIME: float = 120.0  # Max episode length in simulated seconds

    def __init__(
        self,
        num_evacuees: int = 60,
        random_disasters: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.num_evacuees = num_evacuees
        self.max_sim_time = max(120.0, 90.0 + 0.15 * num_evacuees)
        self.random_disasters = random_disasters
        self.sim = Simulation()
        self.sim.num_evacuees = num_evacuees
        self.exit_dists = self.sim.static_exit_dists

        self.agents: List[str] = sorted(list(self.sim.node_positions.keys()))
        self.possible_agents: List[str] = self.agents.copy()
        self.node_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(self.agents)}
        self.idx_to_node: Dict[int, str] = {i: nid for i, nid in enumerate(self.agents)}

        # Build incident adjacency per agent
        self.agent_neighbors: Dict[str, List[str]] = {}
        for nid in self.agents:
            self.agent_neighbors[nid] = []
        for s, t, d, w in self.sim.edge_list:
            if t not in self.agent_neighbors[s]:
                self.agent_neighbors[s].append(t)
            if s not in self.agent_neighbors[t]:
                self.agent_neighbors[t].append(s)

        # Graph Edge Index for PyTorch GNN (2, E)
        edges = []
        for s, nbrs in self.agent_neighbors.items():
            u_idx = self.node_to_idx[s]
            for nbr in nbrs:
                v_idx = self.node_to_idx[nbr]
                edges.append([u_idx, v_idx])
        self.edge_index = np.array(edges, dtype=np.int64).T  # (2, E)

        # Observation dimension: 11 base features + (MAX_CORRIDORS * 5) neighbor features
        self.base_feat_dim = 11
        self.obs_dim = self.base_feat_dim + (self.MAX_CORRIDORS * 5)

        self.observation_spaces: Dict[str, Box] = {
            agent: Box(low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        # Discrete choice: 0..MAX_CORRIDORS-1 = primary green exit corridor, MAX_CORRIDORS = standby
        self.action_spaces: Dict[str, gymnasium.spaces.Discrete] = {
            agent: gymnasium.spaces.Discrete(self.MAX_CORRIDORS + 1)
            for agent in self.possible_agents
        }
        self.action_space = gymnasium.spaces.Discrete(self.MAX_CORRIDORS + 1)

        # Normalized Team Reward Weights (bounded episode returns in [-20, +15])
        self.weights = reward_weights or {
            "evac": 10.0,      # Reward on confirmed egress completion
            "cas": 20.0,       # High penalty on casualty incidence / trapped occupants
            "progress": 2.0,   # Dense step reward for moving closer to exits
            "flip": 0.05,      # Penalty on gratuitous signboard changes
            "cong": 0.5,       # Spatial density variance penalty
            "jam": 1.0,        # Doorway arching & local jamming density penalty
            "cycle": 1.5,      # Graph-level penalty for closed routing cycles
            "unreach": 1.0,    # Graph-level penalty for nodes with no path to exits
        }

        # Episode state tracking
        self.last_evacuated: int = 0
        self.last_casualties: int = 0
        self.last_progress_finished: int = 0
        self.stagnation_timer: float = 0.0
        self.prev_actions: Dict[str, int] = {nid: self.MAX_CORRIDORS for nid in self.agents}
        self.last_hazards: Dict[str, float] = {nid: 0.0 for nid in self.agents}
        self.last_crowds: Dict[str, float] = {nid: 0.0 for nid in self.agents}

    def get_action_mask(self, agent_id: str) -> np.ndarray:
        """Returns boolean mask of length MAX_CORRIDORS + 1 (True = valid corridor choice or standby)."""
        mask = np.zeros(self.MAX_CORRIDORS + 1, dtype=bool)
        nbrs = self.agent_neighbors.get(agent_id, [])
        local_h = self.sim.hazard.levels.get(agent_id, 0.0)
        curr_dist = self.exit_dists.get(agent_id, 50.0)
        num_valid = min(len(nbrs), self.MAX_CORRIDORS)

        # Check if at least one forward corridor (toward exit) is safe and passable
        has_safe_forward = False
        for k in range(num_valid):
            nbr = nbrs[k]
            nbr_h = self.sim.hazard.levels.get(nbr, 0.0)
            nbr_dist = self.exit_dists.get(nbr, 50.0)
            is_hoist = "hoist" in agent_id and "hoist" in nbr
            is_dead_end = len(self.agent_neighbors.get(nbr, [])) <= 1 and nbr not in self.sim.exits
            if local_h < 0.30 and nbr_h < 0.30 and not is_hoist and not is_dead_end:
                if nbr_dist <= curr_dist + 2.0:
                    has_safe_forward = True
                    break

        for k in range(num_valid):
            nbr = nbrs[k]
            nbr_h = self.sim.hazard.levels.get(nbr, 0.0)
            nbr_dist = self.exit_dists.get(nbr, 50.0)
            is_hoist = "hoist" in agent_id and "hoist" in nbr
            is_dead_end = len(self.agent_neighbors.get(nbr, [])) <= 1 and nbr not in self.sim.exits
            if local_h >= 0.80 or nbr_h >= 0.80 or is_hoist or is_dead_end:
                continue

            # In safe conditions with clear forward paths, mask out deep reverse corridors
            if has_safe_forward and local_h < 0.30 and nbr_h < 0.30:
                if nbr_dist > curr_dist + 4.0:
                    continue  # Prevents pointless backward loops when the exit path is completely clear

            mask[k] = True

        # Safety fallback: ensure at least one physical corridor is available
        if not np.any(mask[:num_valid]):
            for k in range(num_valid):
                nbr = nbrs[k]
                nbr_h = self.sim.hazard.levels.get(nbr, 0.0)
                is_hoist = "hoist" in agent_id and "hoist" in nbr
                is_dead_end = len(self.agent_neighbors.get(nbr, [])) <= 1 and nbr not in self.sim.exits
                if local_h < 0.80 and nbr_h < 0.80 and not is_hoist and not is_dead_end:
                    mask[k] = True

        mask[self.MAX_CORRIDORS] = True  # Standby option is always valid
        return mask

    def get_all_action_masks(self) -> np.ndarray:
        """Returns (N, MAX_CORRIDORS + 1) boolean array."""
        masks = np.zeros((len(self.agents), self.MAX_CORRIDORS + 1), dtype=bool)
        for i, agent in enumerate(self.agents):
            masks[i] = self.get_action_mask(agent)
        return masks

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        opts = options or {}
        num_evac = opts.get("num_evacuees", self.num_evacuees)
        self.num_evacuees = num_evac
        # Realistic egress duration ceiling (watchdog timeout: 105s base + crowd buffer, terminates early when done)
        self.max_sim_time = min(220.0, 105.0 + 0.35 * num_evac)
        self.last_progress_finished = 0
        self.stagnation_timer = 0.0
        cluster_node = opts.get("cluster_node")
        cluster_ratio = float(opts.get("cluster_ratio", 0.0))
        self.sim.reset(num_evacuees=num_evac, cluster_node=cluster_node, cluster_ratio=cluster_ratio)
        self.sim.set_policy_mode(opts.get("policy_mode", "marl"))

        # Inject disaster scenario
        if opts.get("inject", True):
            target_node = opts.get("node_id")
            hazard_type = opts.get("hazard_type", "GAS_RELEASE")
            intensity = opts.get("intensity", 0.85)

            if not target_node and self.random_disasters:
                candidates = [
                    "tank_farm_a", "tank_farm_b", "reactor_1", "reactor_2",
                    "hazmat_basin", "compressor_shed", "pipe_rack_junc_1",
                    "corridor_perimeter_n", "corridor_perimeter_s", "loading_bay"
                ]
                target_node = random.choice(candidates)
                hazard_type = random.choice(["GAS_RELEASE", "THERMAL_FIRE", "TOXIC_PLUME"])

            if target_node:
                self.sim.inject_disaster(target_node, hazard_type, intensity)

        # Trigger evacuation alarm
        self.sim.trigger_alarm()

        self.last_evacuated = 0
        self.last_casualties = 0
        self.prev_actions = {a: self.MAX_CORRIDORS for a in self.agents}
        self.last_hazards = {nid: self.sim.hazard.levels.get(nid, 0.0) for nid in self.agents}
        self.last_crowds = {nid: 0.0 for nid in self.agents}

        obs = self._get_observations()
        infos = {agent: {"action_mask": self.get_action_mask(agent)} for agent in self.agents}
        return obs, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Constructs observation vector for each agent."""
        # Compute local occupant counts per node
        node_counts = {nid: 0 for nid in self.agents}
        for ped in self.sim.agents:
            if ped.state not in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                if ped.current_node in node_counts:
                    node_counts[ped.current_node] += 1

        obs_dict = {}
        for agent in self.agents:
            cap = max(1, self.sim.node_capacities.get(agent, 10))
            pop = node_counts.get(agent, 0)
            rho = min(2.0, pop / float(cap))
            h = self.sim.hazard.levels.get(agent, 0.0)
            floor = self.sim.node_floors.get(agent, 1) / 3.0
            is_stair = 1.0 if "stair" in agent else 0.0
            is_exit = 1.0 if agent in self.sim.exits else 0.0
            cap_norm = min(1.0, cap / 50.0)

            # 4-byte sparse gossip differentials: Delta H, Delta rho
            delta_h = h - self.last_hazards.get(agent, 0.0)
            delta_rho = rho - self.last_crowds.get(agent, 0.0)
            time_ratio = min(1.0, self.sim.t / max(1.0, self.max_sim_time))
            alarm_flag = 1.0 if self.sim.alarm_active else 0.0
            exit_dist_norm = min(1.0, self.exit_dists.get(agent, 50.0) / 100.0)

            vec = [
                h, rho, floor, is_stair, is_exit,
                cap_norm, delta_h, delta_rho, time_ratio, alarm_flag,
                exit_dist_norm,
            ]

            # Neighbor features for up to MAX_CORRIDORS
            nbrs = self.agent_neighbors.get(agent, [])
            for k in range(self.MAX_CORRIDORS):
                if k < len(nbrs):
                    nbr = nbrs[k]
                    n_h = self.sim.hazard.levels.get(nbr, 0.0)
                    n_pop = node_counts.get(nbr, 0)
                    n_cap = max(1, self.sim.node_capacities.get(nbr, 10))
                    n_rho = min(2.0, n_pop / float(n_cap))
                    # Find distance and width
                    dist, width = 10.0, 2.0
                    for s, t, d, w in self.sim.edge_list:
                        if (s == agent and t == nbr) or (t == agent and s == nbr):
                            dist, width = d, w
                            break
                    n_exit_norm = min(1.0, self.exit_dists.get(nbr, 50.0) / 100.0)
                    vec.extend([n_h, n_rho, min(1.0, dist / 30.0), min(1.0, width / 5.0), n_exit_norm])
                else:
                    vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            obs_dict[agent] = np.array(vec, dtype=np.float32)

            self.last_hazards[agent] = h
            self.last_crowds[agent] = rho

        return obs_dict

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Executes one policy step:
          1. Translates single primary exit arrow actions into physical 6-corridor signboard states.
          2. Runs continuous SFM physics sub-steps (DECISION_DT seconds).
          3. Evaluates cooperative multi-objective team reward with dense progress shaping.
        """
        flipping_penalty = 0.0
        safety_interventions = 0
        safe_actions: Dict[str, np.ndarray] = {}

        # Count active pedestrians per node before step logic to gauge crowd pressure
        node_occupancies = {}
        for p in self.sim.agents:
            if p.state not in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                node_occupancies[p.current_node] = node_occupancies.get(p.current_node, 0) + 1

        for agent, act in actions.items():
            act_idx = int(np.asarray(act).item()) if isinstance(act, (np.ndarray, list)) else int(act)
            prev_act = self.prev_actions.get(agent, self.MAX_CORRIDORS)
            nbrs = self.agent_neighbors.get(agent, [])
            local_hazard = self.sim.hazard.levels.get(agent, 0.0)
            cap = max(1, self.sim.node_capacities.get(agent, 10))
            current_rho = min(2.0, node_occupancies.get(agent, 0) / float(cap))
            delta_h = abs(local_hazard - self.last_hazards.get(agent, 0.0))
            delta_rho = abs(current_rho - self.last_crowds.get(agent, 0.0))

            # Build physical 6-corridor action array for simulation engine:
            # 0: NORMAL, 1: CAUTION, 2: BLOCKED, 3: GREEN ARROW
            action_vec = np.zeros(self.MAX_CORRIDORS, dtype=np.int64)

            for k, nbr in enumerate(nbrs[:self.MAX_CORRIDORS]):
                neighbor_hazard = self.sim.hazard.levels.get(nbr, 0.0)
                if local_hazard >= 0.80 or neighbor_hazard >= 0.80:
                    action_vec[k] = 2  # BLOCKED (Red X)
                    if k == act_idx:
                        safety_interventions += 1
                elif k == act_idx:
                    action_vec[k] = 3  # GREEN ARROW
                elif neighbor_hazard >= 0.30:
                    action_vec[k] = 1  # CAUTION (Amber Detour)
                else:
                    action_vec[k] = 0  # NORMAL (Open standby)

            # Anti-flipping penalty: penalize changing arrows unless local hazard or crowd pressure shifted
            if act_idx != prev_act and delta_h < 0.15 and delta_rho < 0.25:
                flipping_penalty += 1.0

            safe_actions[agent] = action_vec
            self.prev_actions[agent] = act_idx

        # Update local signboard actions directly in the simulation engine.
        self.sim.set_signboard_actions(safe_actions, externally_controlled=True)

        # Track active pedestrian exit distances BEFORE advancing physics
        dist_before = {
            p.id: self.exit_dists.get(p.current_node, 50.0)
            for p in self.sim.agents
            if p.state not in (PedestrianState.EVACUATED, PedestrianState.CASUALTY)
        }

        # 2. Advance continuous SFM physics by DECISION_DT
        ticks = int(round(self.DECISION_DT / self.sim.DT))
        for _ in range(ticks):
            self.sim._tick()

        # Track active pedestrian exit distances AFTER advancing physics
        progress_sum = 0.0
        for pid, d_start in dist_before.items():
            ped = self.sim.agents[pid]
            if ped.state == PedestrianState.EVACUATED:
                progress_sum += d_start  # Reached exit, completed all remaining distance
            elif ped.state != PedestrianState.CASUALTY:
                d_end = self.exit_dists.get(ped.current_node, 50.0)
                progress_sum += (d_start - d_end)

        active_count = max(1, len(dist_before))
        r_progress = float(np.clip(progress_sum / (active_count * self.DECISION_DT * 1.34), -1.5, 1.5))

        # 3. Compute metrics & rewards
        metrics = self.sim.metrics()
        evac_now = metrics["evacuated"]
        cas_now = metrics["casualties"]

        delta_evac = evac_now - self.last_evacuated
        delta_cas = cas_now - self.last_casualties

        self.last_evacuated = evac_now
        self.last_casualties = cas_now

        # Spatial density variance across nodes and doorway arching/jamming penalty
        node_occupancies_post = {}
        for p in self.sim.agents:
            if p.state not in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                node_occupancies_post[p.current_node] = node_occupancies_post.get(p.current_node, 0) + 1
        node_densities = [
            node_occupancies_post.get(nid, 0) / float(max(1, self.sim.node_capacities.get(nid, 10)))
            for nid in self.agents
        ]
        cong_penalty = float(np.var(node_densities)) if node_densities else 0.0
        # Jamming penalty: penalizes local density spikes exceeding critical jamming threshold (rho >= 1.5)
        jamming_penalty = float(np.mean([max(0.0, d - 1.5) ** 2 for d in node_densities])) if node_densities else 0.0

        # Scale-invariant normalization across crowd density N
        n_total = max(1, self.sim.num_evacuees)
        prop_evac = float(delta_evac) / n_total
        prop_cas = float(delta_cas) / n_total
        in_transit = max(0, self.sim.num_evacuees - (evac_now + cas_now))

        # Check termination / truncation: terminates naturally when in_transit == 0
        total_finished = evac_now + cas_now
        is_terminated = (total_finished >= self.sim.num_evacuees)

        # Stagnation early-stopping: if no progress for 30s while people remain in-transit
        # (grace window of 40s allows distant occupants to walk from upper floors/wings)
        if total_finished > self.last_progress_finished:
            self.last_progress_finished = total_finished
            self.stagnation_timer = 0.0
        elif in_transit > 0:
            self.stagnation_timer += self.DECISION_DT

        is_stagnated = (self.sim.t >= 40.0 and self.stagnation_timer >= 30.0 and in_transit > 0)
        is_truncated = (self.sim.t >= self.max_sim_time) or is_stagnated

        # In-transit urgency penalty: gentle per-step nudging to evacuate quickly
        urgency_penalty = 0.01 * (float(in_transit) / float(n_total))
        # Watchdog / Stagnation trapped penalty: assessed once when episode truncates or stagnates with trapped occupants
        trapped_penalty = (self.weights["cas"] * (float(in_transit) / float(n_total))) if is_truncated and in_transit > 0 else 0.0

        # Graph-level Topological Cycle & Reachability Analysis
        G_arrow = nx.DiGraph()
        G_arrow.add_nodes_from(self.agents)
        for agent, act_vec in safe_actions.items():
            if agent in self.sim.exits:
                continue
            nbrs = self.agent_neighbors.get(agent, [])
            for k, code in enumerate(act_vec):
                if code == 3 and k < len(nbrs):  # GREEN ARROW
                    nbr = nbrs[k]
                    if self.sim.hazard.levels.get(agent, 0.0) < 0.80 and self.sim.hazard.levels.get(nbr, 0.0) < 0.80:
                        G_arrow.add_edge(agent, nbr)

        # Detect circular routing loops (directed cycles)
        cycle_nodes = set()
        for cyc in nx.simple_cycles(G_arrow):
            cycle_nodes.update(cyc)
        cycle_penalty = float(len(cycle_nodes)) / float(max(1, len(self.agents)))

        # Detect non-exit nodes with no directed path to any perimeter exit
        G_rev = G_arrow.reverse()
        reachable_from_exits = set()
        for ex in self.sim.exits:
            if ex in G_rev:
                reachable_from_exits.update(nx.descendants(G_rev, ex))
                reachable_from_exits.add(ex)
        non_exits = set(self.agents) - set(self.sim.exits)
        unreachable_nodes = non_exits - reachable_from_exits
        unreach_penalty = float(len(unreachable_nodes)) / float(max(1, len(non_exits)))

        # Team reward shared equally across all edge router agents
        team_reward = (
            self.weights["evac"] * prop_evac
            - self.weights["cas"] * prop_cas
            + self.weights.get("progress", 2.0) * r_progress
            - self.weights["cong"] * cong_penalty
            - self.weights.get("jam", 1.0) * jamming_penalty
            - self.weights["flip"] * (flipping_penalty / max(1, len(self.agents)))
            - self.weights.get("cycle", 1.5) * cycle_penalty
            - self.weights.get("unreach", 1.0) * unreach_penalty
            - urgency_penalty
            - trapped_penalty
        )

        rewards = {agent: team_reward for agent in self.agents}

        terminations = {agent: is_terminated for agent in self.agents}
        truncations = {agent: is_truncated for agent in self.agents}

        obs = self._get_observations()
        infos = {
            agent: {
                "action_mask": self.get_action_mask(agent),
                "evacuated": evac_now,
                "casualties": cas_now,
                "sim_time": self.sim.t,
                "progress_reward": r_progress,
                "safety_interventions": safety_interventions,
                "cycle_penalty": cycle_penalty,
                "unreach_penalty": unreach_penalty,
            }
            for agent in self.agents
        }

        return obs, rewards, terminations, truncations, infos
