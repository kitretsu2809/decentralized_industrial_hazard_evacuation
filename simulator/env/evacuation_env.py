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
    Action space: MultiDiscrete([3] * max_corridors) for incident directional signage:
      0: ALLOW (Green Arrow)
      1: REDIRECT (Amber Caution detour)
      2: BLOCK (Red X blockade)
    """
    metadata = {"name": "industrial_evacuation_v1", "render_modes": ["human"]}

    MAX_CORRIDORS: int = 6
    DECISION_DT: float = 1.0  # Decision interval: policy steps every 1.0 sim-second (10 SFM ticks)
    MAX_SIM_TIME: float = 120.0  # Max episode length in simulated seconds

    def __init__(
        self,
        num_evacuees: int = 60,
        random_disasters: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.num_evacuees = num_evacuees
        self.random_disasters = random_disasters
        self.sim = Simulation()
        self.sim.num_evacuees = num_evacuees

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

        # Observation dimension: 10 base features + (MAX_CORRIDORS * 4) neighbor features
        self.base_feat_dim = 10
        self.obs_dim = self.base_feat_dim + (self.MAX_CORRIDORS * 4)

        self.observation_spaces: Dict[str, Box] = {
            agent: Box(low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces: Dict[str, MultiDiscrete] = {
            agent: MultiDiscrete([3] * self.MAX_CORRIDORS)
            for agent in self.possible_agents
        }

        # Scale-Invariant Life-Safety Reward Weights (Normalized by total headcount N)
        self.weights = reward_weights or {
            "evac": 100.0,    # Scale-invariant reward: 100.0 * (delta_evac / N_total)
            "cas": 500.0,     # Dominant life-safety penalty: 500.0 * (delta_cas / N_total)
            "cas_flat": 5.0,  # Zero-tolerance penalty on any step where casualties occur
            "cong": 0.5,      # Penalty on corridor density variance
            "flip": 0.05,     # Penalty on rapid signage flickering
        }

        # Episode state tracking
        self.last_evacuated: int = 0
        self.last_casualties: int = 0
        self.prev_actions: Dict[str, np.ndarray] = {}
        self.last_hazards: Dict[str, float] = {nid: 0.0 for nid in self.agents}
        self.last_crowds: Dict[str, float] = {nid: 0.0 for nid in self.agents}

    def get_action_mask(self, agent_id: str) -> np.ndarray:
        """Returns boolean mask of length MAX_CORRIDORS (True = valid incident corridor)."""
        mask = np.zeros(self.MAX_CORRIDORS, dtype=bool)
        num_valid = min(len(self.agent_neighbors[agent_id]), self.MAX_CORRIDORS)
        mask[:num_valid] = True
        return mask

    def get_all_action_masks(self) -> np.ndarray:
        """Returns (N, MAX_CORRIDORS) boolean array."""
        masks = np.zeros((len(self.agents), self.MAX_CORRIDORS), dtype=bool)
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
        self.sim.reset(num_evacuees=num_evac)
        self.sim.set_policy_mode(opts.get("policy_mode", "marl"))

        # Inject disaster scenario
        if opts.get("inject", True):
            target_node = opts.get("node_id")
            hazard_type = opts.get("hazard_type", "GAS_RELEASE")
            intensity = opts.get("intensity", 0.85)

            if not target_node and self.random_disasters:
                candidates = [
                    "tank_farm_a", "tank_farm_b", "reactor_1", "reactor_2",
                    "hazmat_basin", "compressor_shed", "pipe_track_junc_1",
                    "corridor_f2_lab", "corridor_f3_mech", "loading_bay"
                ]
                target_node = random.choice(candidates)
                hazard_type = random.choice(["GAS_RELEASE", "THERMAL_FIRE", "TOXIC_PLUME"])

            if target_node:
                self.sim.inject_disaster(target_node, hazard_type, intensity)

        # Trigger evacuation alarm
        self.sim.trigger_alarm()

        self.last_evacuated = 0
        self.last_casualties = 0
        self.prev_actions = {a: np.zeros(self.MAX_CORRIDORS, dtype=np.int64) for a in self.agents}
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
            time_ratio = min(1.0, self.sim.t / self.MAX_SIM_TIME)
            alarm_flag = 1.0 if self.sim.alarm_active else 0.0

            vec = [
                h, rho, floor, is_stair, is_exit,
                cap_norm, delta_h, delta_rho, time_ratio, alarm_flag
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
                    vec.extend([n_h, n_rho, min(1.0, dist / 30.0), min(1.0, width / 5.0)])
                else:
                    vec.extend([0.0, 0.0, 0.0, 0.0])

            obs_dict[agent] = np.array(vec, dtype=np.float32)

            self.last_hazards[agent] = h
            self.last_crowds[agent] = rho

        return obs_dict

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Executes one policy step:
          1. Translates actions to signboards and corridor routing penalties.
          2. Runs continuous SFM physics sub-steps (DECISION_DT seconds).
          3. Evaluates cooperative multi-objective team reward.
        """
        # 1. Update signboards and edge router penalties
        edge_penalties = {}
        flipping_penalty = 0.0

        for agent, act in actions.items():
            nbrs = self.agent_neighbors.get(agent, [])
            prev_act = self.prev_actions.get(agent, np.zeros(self.MAX_CORRIDORS, dtype=np.int64))

            for k, nbr in enumerate(nbrs[:self.MAX_CORRIDORS]):
                a = int(act[k])
                if a != prev_act[k]:
                    flipping_penalty += 1.0

                # Determine edge cost based on action:
                # 0 = ALLOW: standard or slightly prioritized (1.0)
                # 1 = REDIRECT: caution detour penalty (5.0)
                # 2 = BLOCK: impassable barrier (999.0)
                edge_cost = 1.0 if a == 0 else (5.0 if a == 1 else 999.0)
                edge_penalties[(agent, nbr)] = edge_cost
                edge_penalties[(nbr, agent)] = edge_cost

            self.prev_actions[agent] = act.copy()

        # Update DijkstraRouter edge costs to guide pedestrian egress steering
        self._apply_policy_to_router(edge_penalties)

        # 2. Advance continuous SFM physics by DECISION_DT
        ticks = int(round(self.DECISION_DT / self.sim.DT))
        for _ in range(ticks):
            self.sim._tick()

        # 3. Compute metrics & rewards
        metrics = self.sim.metrics()
        evac_now = metrics["evacuated"]
        cas_now = metrics["casualties"]

        delta_evac = evac_now - self.last_evacuated
        delta_cas = cas_now - self.last_casualties

        self.last_evacuated = evac_now
        self.last_casualties = cas_now

        # Corridor density balance variance penalty
        active_counts = [
            metrics["moving"], metrics["reacting"], metrics["in_danger"], metrics["rerouting"]
        ]
        cong_penalty = float(np.var(active_counts)) / (self.sim.num_evacuees ** 2 + 1e-6)

        # Scale-invariant normalization across crowd density N in [20, 600]
        n_total = max(1, self.sim.num_evacuees)
        prop_evac = float(delta_evac) / n_total
        prop_cas = float(delta_cas) / n_total
        flat_cas_penalty = self.weights.get("cas_flat", 5.0) if delta_cas > 0 else 0.0

        # Team reward shared equally across all edge router agents
        team_reward = (
            self.weights["evac"] * prop_evac
            - self.weights["cas"] * prop_cas
            - flat_cas_penalty
            - self.weights["cong"] * cong_penalty
            - self.weights["flip"] * (flipping_penalty / max(1, len(self.agents) * self.MAX_CORRIDORS))
        )

        rewards = {agent: team_reward for agent in self.agents}

        # Check termination / truncation
        total_finished = evac_now + cas_now
        is_terminated = (total_finished >= self.sim.num_evacuees)
        is_truncated = (self.sim.t >= self.MAX_SIM_TIME)

        terminations = {agent: is_terminated for agent in self.agents}
        truncations = {agent: is_truncated for agent in self.agents}

        obs = self._get_observations()
        infos = {
            agent: {
                "action_mask": self.get_action_mask(agent),
                "evacuated": evac_now,
                "casualties": cas_now,
                "sim_time": self.sim.t,
            }
            for agent in self.agents
        }

        return obs, rewards, terminations, truncations, infos

    def _apply_policy_to_router(self, edge_penalties: Dict[Tuple[str, str], float]):
        """Injects policy edge weights into the simulation navigation graph."""
        self.sim.policy_multipliers = edge_penalties
        if self.sim.policy_mode == "marl":
            self.sim.router.update_weights(
                self.sim.hazard.levels,
                self.sim.hazard.blocked,
                self.sim.policy_multipliers,
            )
