"""
MAPPO (Multi-Agent Proximal Policy Optimization) Training & Benchmark Pipeline
for Decentralized Industrial Hazard Evacuation using ST-TBA-GAT.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_REPO_DIR = Path(__file__).resolve().parent.parent.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from simulator.env.evacuation_env import IndustrialEvacuationEnv
from simulator.policy.st_tba_gat import ST_TBA_GAT
from simulator.engine.pedestrian import PedestrianState
from core.policy.routing_actions import (
    direction_mask_to_action,
    standby_action,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ValueStandardizer:
    """
    Running mean and variance standardizer for value targets (PopArt-style normalization).
    Normalizes targets to mean 0, variance 1 to stabilize value loss gradients
    across changing crowd sizes and disaster intensities.
    """
    def __init__(self, clip_val: float = 10.0, eps: float = 1e-5):
        self.clip_val = clip_val
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, values: np.ndarray | torch.Tensor):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        flat = np.asarray(values, dtype=np.float64).flatten()
        if len(flat) == 0:
            return
        batch_mean = float(np.mean(flat))
        batch_var = float(np.var(flat))
        batch_count = float(len(flat))

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m2 / max(1.0, tot_count)

        self.mean = new_mean
        self.var = max(1e-4, new_var)
        self.count = tot_count

    def normalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        std = (self.var + self.eps) ** 0.5
        if isinstance(x, torch.Tensor):
            return torch.clamp((x - self.mean) / std, -self.clip_val, self.clip_val)
        return np.clip((x - self.mean) / std, -self.clip_val, self.clip_val)

    def denormalize(self, x: torch.Tensor | float) -> torch.Tensor | float:
        std = (self.var + self.eps) ** 0.5
        return x * std + self.mean


class RolloutBuffer:
    """Stores transitions for MAPPO policy and value updates."""
    def __init__(self, num_agents: int, obs_dim: int, max_corridors: int, hidden_dim: int):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.max_corridors = max_corridors
        self.hidden_dim = hidden_dim
        self.clear()

    def clear(self):
        self.node_features: List[np.ndarray] = []  # (N, D)
        self.hidden_states: List[np.ndarray] = []  # (N, H)
        self.actions: List[np.ndarray] = []        # (N,)
        self.action_log_probs: List[np.ndarray] = []  # (N,)
        self.action_masks: List[np.ndarray] = []   # (N, NUM_ACTIONS)
        self.rewards: List[float] = []             # scalar
        self.values: List[float] = []              # scalar
        # True only for an MDP terminal state. Time-limit/stagnation truncations
        # must bootstrap from the critic instead of being treated as terminal.
        self.terminals: List[bool] = []            # scalar
        self.teacher_actions: List[np.ndarray] = []  # (N,)

    def add(
        self,
        node_features: np.ndarray,
        hidden_states: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        action_masks: np.ndarray,
        reward: float,
        value: float,
        terminated: bool,
        teacher_actions: Optional[np.ndarray] = None,
    ):
        self.node_features.append(node_features)
        self.hidden_states.append(hidden_states)
        self.actions.append(actions)
        self.action_log_probs.append(action_log_probs)
        self.action_masks.append(action_masks)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminals.append(terminated)
        if teacher_actions is not None:
            self.teacher_actions.append(teacher_actions)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates GAE advantages and discounted returns."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            non_terminal = 0.0 if self.terminals[t] else 1.0
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        return returns, norm_advantages


class MAPPOTrainer:
    """
    Centralized Training with Decentralized Execution (CTDE) MAPPO Trainer.
    """
    def __init__(
        self,
        env: Optional[IndustrialEvacuationEnv] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        ppo_batch_size: int = 16,
        device: Optional[str] = None,
        save_dir: str = "checkpoints",
        min_evacuees: int = 20,
        max_evacuees: int = 600,
        curriculum: bool = True,
        num_episodes: int = 100,
        resume_checkpoint: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.min_evacuees = min_evacuees
        self.max_evacuees = max_evacuees
        self.curriculum = curriculum
        self.num_episodes = num_episodes
        self.env = env or IndustrialEvacuationEnv(num_evacuees=min_evacuees)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = max(1, ppo_batch_size)
        self.save_dir = save_dir

        self.num_agents = len(self.env.agents)
        self.obs_dim = self.env.obs_dim
        self.max_corridors = self.env.MAX_CORRIDORS
        self.hidden_dim = 64

        # Initialize ST-TBA-GAT policy network
        self.policy = ST_TBA_GAT(
            node_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            max_corridors=self.max_corridors,
            gat_heads=4,
            gat_layers=2,
        ).to(self.device)
        self.start_episode = 1
        self.resume_checkpoint = resume_checkpoint
        self.resume_metrics: Dict[str, Any] = {}
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        if resume_checkpoint:
            self._resume_policy(resume_checkpoint)

        # PPO requires the rollout and update likelihoods to describe the same
        # policy. Gradients still work in eval mode, while GAT dropout is disabled.
        self.policy.eval()
        self.edge_index = torch.tensor(
            self.env.edge_index, dtype=torch.long, device=self.device
        )
        self.buffer = RolloutBuffer(
            self.num_agents, self.obs_dim, self.max_corridors, self.hidden_dim
        )
        self.value_normalizer = ValueStandardizer()

    def _resume_policy(self, checkpoint_path: str):
        """Restores a compatible policy and continues after its saved episode.

        The inference checkpoint remains portable, while optional optimizer state
        allows a training job to continue without silently discarding Adam moments.
        Incompatible checkpoints fail loudly instead of restarting from random
        weights at a late curriculum stage.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_obs_dim = int(payload.get("node_dim", self.obs_dim))
        checkpoint_corridors = int(payload.get("max_corridors", self.max_corridors))
        if checkpoint_obs_dim != self.obs_dim or checkpoint_corridors != self.max_corridors:
            raise ValueError(
                "Resume checkpoint is incompatible with this environment: "
                f"checkpoint node_dim={checkpoint_obs_dim}, max_corridors={checkpoint_corridors}; "
                f"environment node_dim={self.obs_dim}, max_corridors={self.max_corridors}."
            )

        self.policy.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        else:
            logger.warning(
                "Checkpoint %s has no optimizer state; model weights were restored, "
                "but Adam moments start fresh.",
                checkpoint_path,
            )
        self.resume_metrics = dict(payload.get("extra_meta", {}))
        previous_episode = int(self.resume_metrics.get("episode", 0))
        self.start_episode = max(1, previous_episode + 1)
        logger.info(
            "Resumed policy from %s; continuing at episode %d.",
            checkpoint_path,
            self.start_episode,
        )

    @staticmethod
    def _curriculum_stage(tau: float, enabled: bool = True) -> int:
        """Returns a monotonic difficulty rank used for checkpoint comparison."""
        if not enabled:
            return 0
        if tau < 0.25:
            return 1
        if tau < 0.50:
            return 2
        if tau < 0.75:
            return 3
        return 4

    @staticmethod
    def _checkpoint_rank(metrics: Dict[str, Any]) -> Tuple[int, float, float, float]:
        """Ranks comparable training checkpoints without favoring easy stages."""
        return (
            int(metrics.get("curriculum_stage", 0)),
            float(metrics.get("survival_rate", 0.0)),
            float(metrics.get("clearance_rate", 0.0)),
            float(metrics.get("reward", float("-inf"))),
        )

    def compute_teacher_actions(self, masks_np: np.ndarray) -> np.ndarray:
        """
        Generates congestion-aware, hazard-weighted teacher actions for all agents.
        Uses shortest path over dynamic hazard-and-congestion weighted graph.
        """
        import networkx as nx
        hazard_levels = self.env.sim.hazard.levels
        blocked = self.env.sim.hazard.blocked

        # Local crowd counts
        node_counts = {nid: 0 for nid in self.env.agents}
        for ped in self.env.sim.agents:
            if ped.state not in (PedestrianState.EVACUATED, PedestrianState.CASUALTY):
                if ped.current_node in node_counts:
                    node_counts[ped.current_node] += 1

        G = self.env.sim.router.G
        exit_paths = {}
        if G is not None:
            # Dynamically weigh edges with hazard and congestion penalty
            for u, v, data in G.edges(data=True):
                base_dist = data.get("distance", 10.0)
                h = max(hazard_levels.get(u, 0.0), hazard_levels.get(v, 0.0))
                if u in blocked or v in blocked or h >= 0.80:
                    w = 1e9
                else:
                    cap_u = max(1, self.env.sim.node_capacities.get(u, 10))
                    cap_v = max(1, self.env.sim.node_capacities.get(v, 10))
                    cong = max(node_counts.get(u, 0) / cap_u, node_counts.get(v, 0) / cap_v)
                    w = base_dist * (1.0 + 10.0 * h + 3.0 * min(2.0, cong))
                G[u][v]["weight"] = w

            try:
                exit_paths = nx.multi_source_dijkstra_path(
                    G, self.env.sim.exits, weight="weight"
                )
            except Exception:
                exit_paths = {}

        teacher_actions = np.zeros(self.num_agents, dtype=np.int64)
        for i, agent in enumerate(self.env.agents):
            direction_slots = self.env.directional_neighbors.get(agent, ())
            chosen_action = self.env.STANDBY_ACTION
            if agent in exit_paths and len(exit_paths[agent]) >= 2:
                next_hop = exit_paths[agent][-2]
                if next_hop in direction_slots:
                    k = direction_slots.index(next_hop)
                    cand_act = direction_mask_to_action(1 << k, self.max_corridors)
                    if masks_np[i, cand_act]:
                        chosen_action = cand_act

            # Fallback if preferred action masked
            if not masks_np[i, chosen_action]:
                valid_indices = np.flatnonzero(masks_np[i])
                chosen_action = int(valid_indices[0]) if len(valid_indices) > 0 else self.env.STANDBY_ACTION

            teacher_actions[i] = chosen_action

        return teacher_actions

    def evaluate_canonical(self) -> Dict[str, float]:
        """Runs fixed 3-scenario benchmark suite to evaluate policy without curriculum bias."""
        canonical_scenarios = [
            {"node_id": "reactor_2", "hazard_type": "FIRE", "intensity": 0.85, "num_evacuees": 150},
            {"node_id": "pipe_rack_junc_1", "hazard_type": "GAS_RELEASE", "intensity": 0.90, "num_evacuees": 200},
            {"node_id": "stair_north_f1", "hazard_type": "FIRE", "intensity": 0.90, "num_evacuees": 250},
        ]
        survs = []
        clears = []
        cass = []
        times = []

        for idx, scen in enumerate(canonical_scenarios):
            obs, _ = self.env.reset(seed=42 + idx, options={**scen, "policy_mode": "marl"})
            hs = torch.zeros(self.num_agents, self.hidden_dim, device=self.device)
            while True:
                obs_array = np.stack([obs[agent] for agent in self.env.agents])
                nf = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
                masks_np = self.env.get_all_action_masks()
                masks_t = torch.tensor(masks_np, dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    acts_t, _, hs, _ = self.policy.act(
                        node_features=nf,
                        edge_index=self.edge_index,
                        hidden_state=hs,
                        action_masks=masks_t,
                        deterministic=True,
                    )
                acts_np = acts_t.cpu().numpy()
                act_dict = {agent: acts_np[i] for i, agent in enumerate(self.env.agents)}
                obs, _, terms, truncs, _ = self.env.step(act_dict)
                if any(terms.values()) or any(truncs.values()):
                    break
            m = self.env.sim.metrics()
            tot = m.get("total", scen["num_evacuees"])
            survs.append(m.get("survival_rate", (tot - m["casualties"]) / max(1, tot) * 100.0))
            clears.append(m.get("clearance_rate", m["evacuated"] / max(1, tot) * 100.0))
            cass.append(m["casualties"])
            times.append(m["sim_time"])

        return {
            "val_survival_rate": round(float(np.mean(survs)), 1),
            "val_clearance_rate": round(float(np.mean(clears)), 1),
            "val_casualties": round(float(np.mean(cass)), 1),
            "val_sim_time": round(float(np.mean(times)), 1),
        }

    def train_episode(self, episode_idx: int) -> Dict[str, float]:
        """Runs a single episode rollout and performs PPO update."""
        # Continuous Monotonic Curriculum Progression
        total_eps = max(1, getattr(self, "num_episodes", 100))
        tau = min(1.0, max(0.0, (episode_idx - 1) / max(1, total_eps - 1)))
        curriculum_stage = self._curriculum_stage(tau, self.curriculum)

        if self.curriculum:
            # Pedagogical scenario stages based on normalized progress tau in [0, 1]
            if tau < 0.25:
                stage_name = "Stage 1: Primary Arterial Cut"
                target_node = random.choice(["reactor_2", "pipe_rack_junc_1", "pump_house", "corridor_perimeter_n"])
                hazard_type = random.choice(["FIRE", "GAS_RELEASE"])
            elif tau < 0.50:
                stage_name = "Stage 2: Vertical Stairwell Flash"
                target_node = random.choice(["stair_north_f1", "stair_south_f1", "stair_north_f2", "stair_south_f2"])
                hazard_type = random.choice(["FIRE", "GAS_RELEASE"])
            elif tau < 0.75:
                stage_name = "Stage 3: Bottleneck Arching & Flow-Splitting"
                target_node = random.choice([
                    "pipe_rack_junc_1", "corridor_perimeter_n", "reactor_2",
                    "stair_north_f1", "stair_south_f1", "loading_bay",
                    "tank_farm_a", "tank_farm_b", "hazmat_basin"
                ])
                hazard_type = random.choice(["CHEMICAL_SPILL", "GAS_RELEASE", "THERMAL_FIRE"])
            else:
                stage_name = "Stage 4: Compound Disaster (Stress 600)"
                target_node = random.choice([
                    "pipe_rack_junc_1", "corridor_perimeter_n", "reactor_1", "reactor_2",
                    "stair_north_f1", "stair_south_f1", "tank_farm_a",
                    "compressor_shed", "hazmat_basin"
                ])
                hazard_type = random.choice(["EXPLOSION", "CHEMICAL_SPILL", "FIRE", "TOXIC_PLUME"])

            # High-density crowd cluster scenarios in Stage 3 & 4 at bottleneck nodes
            cluster_node = None
            cluster_ratio = 0.0
            if tau >= 0.50 and random.random() < 0.75:
                cluster_candidates = [
                    "pipe_rack_junc_1", "corridor_perimeter_n", "reactor_2",
                    "stair_north_f1", "stair_south_f1", "loading_bay"
                ]
                cluster_node = random.choice(cluster_candidates)
                cluster_ratio = random.uniform(0.35, 0.60)

            # Smooth monotonic headcount ramp from min_evacuees to max_evacuees: N(tau) = N_min + (N_max - N_min) * tau^1.15
            n_base = self.min_evacuees + (self.max_evacuees - self.min_evacuees) * (tau ** 1.15)
            # Narrow jitter (+/- 4%) prevents discrete overfitting while maintaining monotonic hardness
            jitter = int(random.uniform(-0.04, 0.04) * n_base)
            num_evac = int(np.clip(n_base + jitter, self.min_evacuees, self.max_evacuees))

            # Smooth monotonic hazard intensity ramp from 0.40 (mild) to 0.95 (flashover/blast)
            h_base = 0.40 + 0.55 * tau
            intensity = float(np.clip(h_base + random.uniform(-0.02, 0.02), 0.35, 0.98))

            # Smooth dual-disaster probability ramp: 0% during initial learning, climbing to 55%
            dual_chance = 0.0 if tau < 0.35 else min(0.55, (tau - 0.35) * 0.85)
        else:
            stage_name = "Uniform Random Sampling"
            num_evac = random.randint(self.min_evacuees, self.max_evacuees)
            candidates = [
                "tank_farm_a", "tank_farm_b", "reactor_1", "reactor_2",
                "hazmat_basin", "compressor_shed", "pipe_rack_junc_1", "pipe_rack_junc_2",
                "loading_bay", "pump_house", "corridor_perimeter_n", "corridor_perimeter_s",
                "stair_north_f1", "stair_south_f1"
            ]
            target_node = random.choice(candidates)
            hazard_type = random.choice(["GAS_RELEASE", "FIRE", "EXPLOSION", "CHEMICAL_SPILL"])
            intensity = random.uniform(0.75, 0.98)
            dual_chance = 0.30

            cluster_node = None
            cluster_ratio = 0.0
            if random.random() < 0.50:
                cluster_node = random.choice([
                    "pipe_rack_junc_1", "corridor_perimeter_n", "reactor_2",
                    "stair_north_f1", "stair_south_f1", "loading_bay"
                ])
                cluster_ratio = random.uniform(0.30, 0.55)

        reset_opts = {
            "num_evacuees": num_evac,
            "node_id": target_node,
            "hazard_type": hazard_type,
            "intensity": intensity,
            "inject": True,
        }
        if cluster_node:
            reset_opts["cluster_node"] = cluster_node
            reset_opts["cluster_ratio"] = cluster_ratio

        obs, infos = self.env.reset(options=reset_opts)

        # Configure Dynamic Cascading Disaster (Mid-Episode Flare-up / Secondary Rupture)
        # Instead of injecting at t=0, the secondary disaster erupts mid-evacuation (step 6-14, ~15-35s)
        # while evacuees are actively traversing corridors, forcing the model to master real-time dynamic rerouting!
        dual_injected = False
        sec_trigger_step = -1
        sec_target = None
        sec_htype = None
        sec_intensity = 0.0
        if random.random() < dual_chance:
            candidates = [
                "tank_farm_a", "tank_farm_b", "reactor_1", "reactor_2",
                "hazmat_basin", "loading_bay", "compressor_shed", "pipe_rack_junc_1"
            ]
            valid_cands = [c for c in candidates if c != target_node]
            if valid_cands:
                sec_target = random.choice(valid_cands)
                sec_htype = random.choice(["FIRE", "GAS_RELEASE", "EXPLOSION"])
                sec_intensity = random.uniform(0.70, 0.95)
                # Triggers mid-egress between 15s and 35s while people are walking!
                sec_trigger_step = random.randint(6, 14)

        self.buffer.clear()

        hidden_state = torch.zeros(
            self.num_agents, self.hidden_dim, device=self.device
        )
        total_reward = 0.0
        total_safety_interventions = 0
        step_count = 0

        while True:
            # Mid-Episode Dynamic Disaster Trigger:
            if step_count == sec_trigger_step and sec_target and not dual_injected:
                self.env.sim.inject_disaster(sec_target, sec_htype, sec_intensity)
                dual_injected = True
                # Trigger real-time rerouting for moving evacuees heading toward the newly ignited room
                for p in self.env.sim.agents:
                    if p.path and sec_target in p.path:
                        p.path = []
                        p.state = PedestrianState.REROUTING
                # The disaster was injected after the previous environment step.
                # Refresh observations so the policy sees the new hazard immediately.
                obs = self.env.get_observations()

            # Prepare batch tensor for all agents: (N, obs_dim)
            obs_array = np.stack([obs[agent] for agent in self.env.agents])
            node_features = torch.tensor(
                obs_array, dtype=torch.float32, device=self.device
            )

            # Query live, real-time action masks as hazards diffuse and erupt
            masks_np = self.env.get_all_action_masks()
            live_action_masks = torch.tensor(
                masks_np, dtype=torch.bool, device=self.device
            )

            with torch.no_grad():
                actions_t, log_probs_t, new_hidden, val_t = self.policy.act(
                    node_features=node_features,
                    edge_index=self.edge_index,
                    hidden_state=hidden_state,
                    action_masks=live_action_masks,
                    deterministic=False,
                )

            actions_np = actions_t.cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            val_scalar = float(self.value_normalizer.denormalize(float(val_t.cpu().item())))
            old_hidden_np = hidden_state.cpu().numpy()

            # Generate congestion-aware, hazard-weighted teacher actions for auxiliary guidance
            teacher_act_np = self.compute_teacher_actions(masks_np)

            # Execute environment step
            action_dict = {
                agent: actions_np[i] for i, agent in enumerate(self.env.agents)
            }
            next_obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
            total_safety_interventions += int(
                infos[self.env.agents[0]].get("safety_interventions", 0)
            )

            reward = rewards[self.env.agents[0]]
            terminated = any(terminations.values())
            truncated = any(truncations.values())
            done = terminated or truncated

            self.buffer.add(
                node_features=obs_array,
                hidden_states=old_hidden_np,
                actions=actions_np,
                action_log_probs=log_probs_np,
                action_masks=masks_np,
                reward=reward,
                value=val_scalar,
                terminated=terminated,
                teacher_actions=teacher_act_np,
            )

            total_reward += reward
            step_count += 1
            obs = next_obs
            hidden_state = new_hidden

            if done:
                break

        # Compute returns and update network
        last_val = 0.0
        if not any(terminations.values()):
            with torch.no_grad():
                obs_array = np.stack([obs[agent] for agent in self.env.agents])
                nf = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
                _, _, _, v = self.policy.act(nf, self.edge_index, hidden_state)
                last_val = float(self.value_normalizer.denormalize(float(v.cpu().item())))

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=last_val, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        # Update running value target standardizer (PopArt normalization)
        self.value_normalizer.update(returns)

        update_metrics = self._update_ppo(returns, advantages, tau=tau)

        metrics = self.env.sim.metrics()
        total_peds = metrics.get("total", self.env.sim.num_evacuees)
        survival_rate = metrics.get(
            "survival_rate",
            (total_peds - metrics["casualties"]) / max(1, total_peds) * 100.0,
        )
        clearance_rate = metrics.get(
            "clearance_rate",
            (metrics["evacuated"] / max(1, total_peds)) * 100.0,
        )
        in_transit = metrics.get(
            "in_transit",
            max(0, total_peds - metrics["evacuated"] - metrics["casualties"]),
        )
        sampled_actions = np.concatenate(self.buffer.actions) if self.buffer.actions else np.array([])
        standby_rate = float(
            np.mean(sampled_actions == self.env.STANDBY_ACTION)
        ) if sampled_actions.size else 0.0
        dual_rate = float(
            np.mean(sampled_actions > self.env.STANDBY_ACTION)
        ) if sampled_actions.size else 0.0
        safety_intervention_rate = total_safety_interventions / float(
            max(1, step_count * self.num_agents)
        )

        return {
            "episode": episode_idx,
            "reward": round(total_reward, 2),
            "steps": step_count,
            "total": total_peds,
            "evacuated": metrics["evacuated"],
            "casualties": metrics["casualties"],
            "in_transit": in_transit,
            "survival_rate": round(survival_rate, 1),
            "clearance_rate": round(clearance_rate, 1),
            "sim_time": metrics["sim_time"],
            "stage_name": stage_name,
            "curriculum_stage": curriculum_stage,
            "hazard_type": hazard_type,
            "intensity": intensity,
            "dual": dual_injected,
            "standby_rate": standby_rate,
            "dual_action_rate": dual_rate,
            "safety_intervention_rate": safety_intervention_rate,
            **update_metrics,
        }

    def _update_ppo(
        self,
        returns: np.ndarray,
        advantages: np.ndarray,
        tau: float = 1.0,
    ) -> Dict[str, float]:
        """Performs clipped PPO actor updates with active-agent weighting and PopArt normalization."""
        T = len(returns)
        if T == 0:
            return {"loss": 0.0, "pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "teacher_loss": 0.0}

        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Decay teacher imitation schedule: beta = 0.50 * max(0.0, 1.0 - tau)
        beta_teacher = 0.50 * max(0.0, 1.0 - tau)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_teacher_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_grad_norm = 0.0
        optimizer_steps = 0
        samples_seen = 0
        actor_before = [
            parameter.detach().clone()
            for parameter in self.policy.actor_mlp.parameters()
        ]

        # Keep dropout disabled while retaining gradients. Otherwise old and new
        # log-probabilities differ even before an optimizer step.
        self.policy.eval()

        for _epoch in range(self.ppo_epochs):
            permutation = torch.randperm(T).tolist()
            for batch_start in range(0, T, self.ppo_batch_size):
                batch_indices = permutation[batch_start:batch_start + self.ppo_batch_size]
                batch_pi_losses = []
                batch_v_losses = []
                batch_entropies = []
                batch_teacher_losses = []
                batch_kls = []
                batch_clip_fractions = []

                for t in batch_indices:
                    nf = torch.tensor(
                        self.buffer.node_features[t],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    hs = torch.tensor(
                        self.buffer.hidden_states[t],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    act = torch.tensor(
                        self.buffer.actions[t], dtype=torch.long, device=self.device
                    )
                    old_lp = torch.tensor(
                        self.buffer.action_log_probs[t],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    masks_t = torch.tensor(
                        self.buffer.action_masks[t],
                        dtype=torch.bool,
                        device=self.device,
                    )

                    val_pred, new_lp, entropy, logits = self.policy.evaluate_actions(
                        node_features=nf,
                        edge_index=self.edge_index,
                        hidden_state=hs,
                        actions=act,
                        action_masks=masks_t,
                        return_logits=True,
                    )

                    # Active-agent policy gradient weighting:
                    # Idle nodes (no hazard, no crowd) contribute lower variance to update
                    h_node = nf[:, 0]
                    rho_node = nf[:, 1]
                    active_weight = torch.clamp(h_node * 2.0 + rho_node * 2.0, min=0.15, max=1.0)
                    weight_sum = active_weight.sum().clamp(min=1e-5)

                    # Each router shares the cooperative team advantage at this step.
                    log_ratio = new_lp - old_lp
                    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                    adv = adv_t[t]

                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * adv
                    per_agent_pi_loss = -torch.min(surr1, surr2)
                    weighted_pi_loss = (per_agent_pi_loss * active_weight).sum() / weight_sum
                    batch_pi_losses.append(weighted_pi_loss)

                    # PopArt-style standardized return target
                    norm_ret = self.value_normalizer.normalize(ret_t[t])
                    batch_v_losses.append(
                        F.smooth_l1_loss(val_pred.squeeze(), norm_ret)
                    )
                    batch_entropies.append(entropy.mean())
                    batch_kls.append(((ratio - 1.0) - log_ratio).mean())
                    batch_clip_fractions.append(
                        ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                    )

                    # Auxiliary Teacher Guidance cross-entropy loss
                    if len(self.buffer.teacher_actions) > t and beta_teacher > 0.0:
                        teacher_act = torch.tensor(
                            self.buffer.teacher_actions[t], dtype=torch.long, device=self.device
                        )
                        teacher_ce = F.cross_entropy(logits, teacher_act, reduction="none")
                        batch_teacher_losses.append(
                            (teacher_ce * active_weight).sum() / weight_sum
                        )

                pi_loss = torch.stack(batch_pi_losses).mean()
                v_loss = torch.stack(batch_v_losses).mean()
                entropy_mean = torch.stack(batch_entropies).mean()
                if batch_teacher_losses:
                    teacher_loss = torch.stack(batch_teacher_losses).mean()
                else:
                    teacher_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    pi_loss
                    + self.value_coef * v_loss
                    - self.entropy_coef * entropy_mean
                    + beta_teacher * teacher_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_grad_norm += float(grad_norm)
                optimizer_steps += 1

                batch_size = len(batch_indices)
                samples_seen += batch_size
                total_pi_loss += float(pi_loss.item()) * batch_size
                total_v_loss += float(v_loss.item()) * batch_size
                total_entropy += float(entropy_mean.item()) * batch_size
                total_teacher_loss += float(teacher_loss.item()) * batch_size
                total_approx_kl += float(torch.stack(batch_kls).mean().item()) * batch_size
                total_clip_fraction += float(
                    torch.stack(batch_clip_fractions).mean().item()
                ) * batch_size

        denom = max(1, samples_seen)
        actor_parameter_delta = float(
            torch.sqrt(
                sum(
                    torch.sum((parameter.detach() - before) ** 2)
                    for parameter, before in zip(
                        self.policy.actor_mlp.parameters(),
                        actor_before,
                    )
                )
            ).item()
        )
        return {
            "pi_loss": round(total_pi_loss / denom, 4),
            "v_loss": round(total_v_loss / denom, 4),
            "entropy": round(total_entropy / denom, 4),
            "teacher_loss": round(total_teacher_loss / denom, 4),
            "approx_kl": round(total_approx_kl / denom, 6),
            "clip_fraction": round(total_clip_fraction / denom, 4),
            "grad_norm": round(total_grad_norm / max(1, optimizer_steps), 6),
            "actor_parameter_delta": round(actor_parameter_delta, 8),
        }

    def train(self, num_episodes: int = 50):
        self.num_episodes = num_episodes
        os.makedirs(self.save_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"🚀 STARTING ST-TBA-GAT MAPPO TRAINING")
        print(f"   Episodes: {num_episodes} | Device: {self.device} | Architecture: Pure PyTorch ST-TBA-GAT")
        if self.resume_checkpoint:
            print(f"   Resuming from episode {self.start_episode}: {self.resume_checkpoint}")
        print("=" * 80 + "\n")

        best_rank: Tuple[int, float, float, float] = (
            self._checkpoint_rank(self.resume_metrics)
            if self.resume_metrics
            else (-1, float("-inf"), float("-inf"), float("-inf"))
        )

        for ep in range(self.start_episode, num_episodes + 1):
            t0 = time.time()
            metrics = self.train_episode(ep)
            dt = time.time() - t0

            surv = metrics["survival_rate"]
            clear = metrics["clearance_rate"]
            rew = metrics["reward"]
            cas = metrics["casualties"]
            evac = metrics["evacuated"]
            total = metrics.get("total", evac + cas)
            sim_t = metrics.get("sim_time", 0.0)
            in_transit = metrics.get("in_transit", max(0, total - evac - cas))

            stage_str = metrics.get("stage_name", "Curriculum").split(":")[0]
            haz_str = metrics.get("hazard_type", "HAZ")[:4]
            int_val = metrics.get("intensity", 0.0)
            dual_str = "+D" if metrics.get("dual", False) else "  "

            print(
                f"Ep {ep:3d}/{num_episodes:3d} [{stage_str}] | "
                f"N: {total:3d} | "
                f"Haz: {haz_str}{dual_str} ({int_val:.2f}) | "
                f"Evac: {evac:3d} | Cas: {cas:2d} | In-Trans: {in_transit:2d} | "
                f"Surv: {surv:5.1f}% | Clear: {clear:5.1f}% | "
                f"Sim: {sim_t:4.1f}s | "
                f"Wall: {dt:4.1f}s | "
                f"R: {rew:6.2f} | "
                f"Loss: (π={metrics['pi_loss']:.3f}, V={metrics['v_loss']:.3f}, T={metrics.get('teacher_loss', 0.0):.3f}) | "
                f"KL: {metrics.get('approx_kl', 0.0):.4f} | "
                f"H: {metrics.get('entropy', 0.0):.2f} | "
                f"Δθ: {metrics.get('actor_parameter_delta', 0.0):.2e} | "
                f"Dual: {100.0 * metrics.get('dual_action_rate', 0.0):4.1f}% | "
                f"Standby: {100.0 * metrics.get('standby_rate', 0.0):4.1f}%"
            )

            # Compare only like-for-like curriculum maturity first. A high score
            # on the easy opening stage must not permanently beat a later policy.
            current_rank = self._checkpoint_rank(metrics)
            if current_rank > best_rank:
                best_rank = current_rank
                best_path = os.path.join(self.save_dir, "best_policy.pt")
                self.policy.save_checkpoint(
                    best_path,
                    extra_meta=metrics,
                    optimizer_state_dict=self.optimizer.state_dict(),
                )
                if os.path.basename(self.save_dir) == "directional":
                    root_best = os.path.join(os.path.dirname(self.save_dir), "best_policy.pt")
                    try:
                        self.policy.save_checkpoint(
                            root_best,
                            extra_meta=metrics,
                            optimizer_state_dict=self.optimizer.state_dict(),
                        )
                    except Exception:
                        pass
                else:
                    dir_best = os.path.join(self.save_dir, "directional", "best_policy.pt")
                    try:
                        os.makedirs(os.path.dirname(dir_best), exist_ok=True)
                        self.policy.save_checkpoint(
                            dir_best,
                            extra_meta=metrics,
                            optimizer_state_dict=self.optimizer.state_dict(),
                        )
                    except Exception:
                        pass

            # Save latest checkpoint
            latest_path = os.path.join(self.save_dir, "st_tba_gat_latest.pt")
            self.policy.save_checkpoint(
                latest_path,
                extra_meta=metrics,
                optimizer_state_dict=self.optimizer.state_dict(),
            )
            if os.path.basename(self.save_dir) == "directional":
                root_latest = os.path.join(os.path.dirname(self.save_dir), "st_tba_gat_latest.pt")
                try:
                    self.policy.save_checkpoint(
                        root_latest,
                        extra_meta=metrics,
                        optimizer_state_dict=self.optimizer.state_dict(),
                    )
                except Exception:
                    pass
            else:
                dir_latest = os.path.join(self.save_dir, "directional", "st_tba_gat_latest.pt")
                try:
                    os.makedirs(os.path.dirname(dir_latest), exist_ok=True)
                    self.policy.save_checkpoint(
                        dir_latest,
                        extra_meta=metrics,
                        optimizer_state_dict=self.optimizer.state_dict(),
                    )
                except Exception:
                    pass

            # Periodic canonical validation check (every 5 episodes and final episode)
            if ep % 5 == 0 or ep == num_episodes:
                val_res = self.evaluate_canonical()
                print(
                    f"   ⭐ [CANONICAL EVAL Ep {ep:3d}] "
                    f"Surv: {val_res['val_survival_rate']:5.1f}% | "
                    f"Clear: {val_res['val_clearance_rate']:5.1f}% | "
                    f"Cas: {val_res['val_casualties']:4.1f} | "
                    f"Time: {val_res['val_sim_time']:4.1f}s"
                )

        print("\n" + "=" * 80)
        print(f"✅ TRAINING FINISHED! Best checkpoint rank: {best_rank}")
        print(f"   Checkpoint saved to: {os.path.join(self.save_dir, 'best_policy.pt')}")
        print("=" * 80 + "\n")


def run_benchmark(
    policy_path: Optional[str] = None,
    num_runs: int = 5,
    num_evacuees: int = 60,
    device: str = "cpu",
):
    """
    Runs side-by-side benchmark comparing:
      1. Static NFPA Baseline (Fixed exit paths, zero hazard awareness)
      2. Classical Dijkstra (Shortest path rerouting)
      3. ST-TBA-GAT (Ours) (Decentralized Spatio-Temporal Bottleneck-Aware GAT)
    """
    print("\n" + "=" * 85)
    print("📊 CAUSAL BENCHMARK: BASELINES, ACTION ABLATIONS, AND ST-TBA-GAT")
    print(f"   Runs per condition: {num_runs} | Evacuees per run: {num_evacuees}")
    print("=" * 85)

    env = IndustrialEvacuationEnv(num_evacuees=num_evacuees)
    device_obj = torch.device(device)

    # Load policy if provided, or initialize default
    if policy_path and os.path.exists(policy_path):
        policy = ST_TBA_GAT.load_checkpoint(policy_path, device=device)
        print(f"   Loaded trained policy: {policy_path}")
    else:
        policy = ST_TBA_GAT(
            node_dim=env.obs_dim,
            hidden_dim=64,
            max_corridors=env.MAX_CORRIDORS,
        ).to(device_obj)
        print("   Using initialized ST-TBA-GAT model")
    policy.eval()

    # Fixed random initialization provides a causal "architecture without
    # learning" control under the same action masks and simulator physics.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        untrained_policy = ST_TBA_GAT(
            node_dim=env.obs_dim,
            hidden_dim=64,
            max_corridors=env.MAX_CORRIDORS,
        ).to(device_obj)
    untrained_policy.eval()

    edge_index = torch.tensor(env.edge_index, dtype=torch.long, device=device_obj)

    scenarios = [
        {"node_id": "reactor_1", "hazard_type": "FIRE", "intensity": 0.95},
        {"node_id": "tank_farm_a", "hazard_type": "GAS_RELEASE", "intensity": 0.95},
        {"node_id": "hazmat_basin", "hazard_type": "CHEMICAL_SPILL", "intensity": 0.95},
        {"node_id": "compressor_shed", "hazard_type": "EXPLOSION", "intensity": 0.95},
        {"node_id": "pipe_rack_junc_1", "hazard_type": "FIRE", "intensity": 0.95},
    ]

    results = {
        "static": [],
        "dijkstra_ideal": [],
        "dijkstra_severed": [],
        "dijkstra_hazard": [],
        "standby": [],
        "random": [],
        "untrained": [],
        "st_tba_gat": [],
    }

    def run_marl_control(control: str, scen: Dict[str, Any], seed: int):
        obs, _ = env.reset(seed=seed, options={**scen, "policy_mode": "marl"})
        env.max_sim_time = min(240.0, 120.0 + 0.50 * num_evacuees)
        hs = torch.zeros(len(env.agents), 64, device=device_obj)
        rng = np.random.default_rng(seed + 10_000)
        active_policy = untrained_policy if control == "untrained" else policy

        while True:
            masks_np = env.get_all_action_masks()
            if control == "standby":
                acts_np = np.full(
                    len(env.agents),
                    standby_action(env.MAX_CORRIDORS),
                    dtype=np.int64,
                )
            elif control == "random":
                acts_np = np.asarray(
                    [int(rng.choice(np.flatnonzero(mask))) for mask in masks_np],
                    dtype=np.int64,
                )
            else:
                live_masks = torch.tensor(masks_np, dtype=torch.bool, device=device_obj)
                obs_array = np.stack([obs[agent] for agent in env.agents])
                nf = torch.tensor(obs_array, dtype=torch.float32, device=device_obj)
                with torch.no_grad():
                    acts_t, _, hs, _ = active_policy.act(
                        node_features=nf,
                        edge_index=edge_index,
                        hidden_state=hs,
                        action_masks=live_masks,
                        deterministic=True,
                    )
                acts_np = acts_t.cpu().numpy()

            act_dict = {agent: acts_np[i] for i, agent in enumerate(env.agents)}
            obs, _, terms, truncs, _ = env.step(act_dict)
            if any(terms.values()) or any(truncs.values()):
                return env.sim.metrics()

    def run_dijkstra_variant(variant: str, scen: Dict[str, Any], seed: int):
        import networkx as nx
        env.reset(seed=seed, options={**scen, "policy_mode": "dijkstra"})
        env.max_sim_time = min(240.0, 120.0 + 0.50 * num_evacuees)
        edges = list(env.sim.router.G.edges())
        if variant == "ideal":
            central_connected = set(env.sim.router.G.nodes())
        else:
            rng = random.Random(seed)
            severed_links = {e for e in edges if scen["node_id"] in e}
            severed_links.update(rng.sample(edges, int(len(edges) * 0.15)))
            comm_net = nx.Graph()
            for u, v in edges:
                if (u, v) not in severed_links and (v, u) not in severed_links:
                    comm_net.add_edge(u, v)
            central_connected = (
                nx.node_connected_component(comm_net, "control_room")
                if "control_room" in comm_net else set()
            )
        env.sim.active_comm_nodes = central_connected

        while True:
            if variant == "hazard":
                # Dynamically reweight graph with real-time hazard levels
                env.sim.router.update_weights(
                    env.sim.hazard.levels,
                    env.sim.hazard.blocked,
                    active_nodes=central_connected,
                )
            dijk_act = {}
            signs = env.sim.compute_signboards()
            for agent in env.agents:
                chosen_action = env.STANDBY_ACTION
                if agent in central_connected:
                    for s in signs.get(agent, []):
                        direction_slots = env.directional_neighbors.get(agent, ())
                        if s.get("action") == "ARROW" and s.get("target") in direction_slots:
                            direction_idx = direction_slots.index(s["target"])
                            chosen_action = direction_mask_to_action(
                                1 << direction_idx,
                                env.MAX_CORRIDORS,
                            )
                            break
                dijk_act[agent] = chosen_action

            _, _, terms, truncs, _ = env.step(dijk_act)
            if any(terms.values()) or any(truncs.values()):
                break
        return env.sim.metrics()

    for run_idx in range(min(num_runs, len(scenarios))):
        scen = scenarios[run_idx]
        seed = 100 + run_idx

        # ── 1. Static NFPA Baseline ───────────────────────────────────────────
        # Static signage has no environmental sensing; paths never redirect around fire/gas
        env.reset(seed=seed, options={**scen, "policy_mode": "static"})
        env.max_sim_time = min(240.0, 120.0 + 0.50 * num_evacuees)
        static_act = {a: env.STANDBY_ACTION for a in env.agents}
        while True:
            _, _, terms, truncs, _ = env.step(static_act)
            if any(terms.values()) or any(truncs.values()):
                break
        m_stat = env.sim.metrics()
        results["static"].append(m_stat)

        # ── 2. Dijkstra Baselines (Ideal, Severed, and Hazard-Weighted) ────────
        results["dijkstra_ideal"].append(run_dijkstra_variant("ideal", scen, seed))
        results["dijkstra_severed"].append(run_dijkstra_variant("severed", scen, seed))
        results["dijkstra_hazard"].append(run_dijkstra_variant("hazard", scen, seed))

        # ── 3. Causal MARL controls and trained policy ────────────────────────
        results["standby"].append(run_marl_control("standby", scen, seed))
        results["random"].append(run_marl_control("random", scen, seed))
        results["untrained"].append(run_marl_control("untrained", scen, seed))
        results["st_tba_gat"].append(run_marl_control("trained", scen, seed))

    # Aggregate metrics
    def summarize(runs):
        total = float(num_evacuees)
        surv = [r.get("survival_rate", (total - r["casualties"]) / max(1.0, total) * 100.0) for r in runs]
        clear = [r.get("clearance_rate", r["evacuated"] / max(1.0, total) * 100.0) for r in runs]
        cas = [r["casualties"] for r in runs]
        unfinished = [
            r.get("in_transit", max(0, total - r["evacuated"] - r["casualties"]))
            for r in runs
        ]
        adverse = [casualties + trapped for casualties, trapped in zip(cas, unfinished)]
        t = [r["sim_time"] for r in runs]
        return {
            "surv_mean": np.mean(surv),
            "surv_std": np.std(surv),
            "clear_mean": np.mean(clear),
            "clear_std": np.std(clear),
            "cas_mean": np.mean(cas),
            "cas_std": np.std(cas),
            "unfinished_mean": np.mean(unfinished),
            "unfinished_std": np.std(unfinished),
            "adverse_mean": np.mean(adverse),
            "time_mean": np.mean(t),
            "time_std": np.std(t),
        }

    summaries = {name: summarize(runs) for name, runs in results.items()}
    s_stat = summaries["static"]
    s_dijk = summaries["dijkstra_severed"]
    s_gat = summaries["st_tba_gat"]

    print("\n" + "-" * 115)
    print(
        f"{'CONTROLLER':<26} | {'SURVIVAL (%)':>14} | {'CLEARANCE (%)':>14} | "
        f"{'CASUALTIES':>12} | {'IN-TRANSIT':>12} | {'TIME (s)':>12}"
    )
    print("-" * 115)
    labels = {
        "static": "STATIC NFPA",
        "dijkstra_ideal": "IDEAL DIJKSTRA (100% COMM)",
        "dijkstra_severed": "SEVERED DIJKSTRA (LINK-CUT)",
        "dijkstra_hazard": "HAZARD-WEIGHTED DIJKSTRA",
        "standby": "MARL ALWAYS STANDBY",
        "random": "MARL RANDOM VALID",
        "untrained": "UNTRAINED ST-TBA-GAT",
        "st_tba_gat": "TRAINED ST-TBA-GAT (OURS)",
    }
    for name in (
        "static",
        "dijkstra_ideal",
        "dijkstra_severed",
        "dijkstra_hazard",
        "standby",
        "random",
        "untrained",
        "st_tba_gat",
    ):
        summary = summaries[name]
        print(
            f"{labels[name]:<26} | "
            f"{summary['surv_mean']:5.1f} ± {summary['surv_std']:4.1f} | "
            f"{summary['clear_mean']:5.1f} ± {summary['clear_std']:4.1f} | "
            f"{summary['cas_mean']:4.1f} ± {summary['cas_std']:3.1f} | "
            f"{summary['unfinished_mean']:4.1f} ± {summary['unfinished_std']:3.1f} | "
            f"{summary['time_mean']:5.1f} ± {summary['time_std']:4.1f}"
        )
    print("-" * 115)
    red_static = (
        (s_stat["adverse_mean"] - s_gat["adverse_mean"])
        / max(1e-6, s_stat["adverse_mean"])
        * 100.0
    )
    red_central = (
        (s_dijk["adverse_mean"] - s_gat["adverse_mean"])
        / max(1e-6, s_dijk["adverse_mean"])
        * 100.0
    )
    cas_red_static = (
        (s_stat["cas_mean"] - s_gat["cas_mean"])
        / max(1e-6, s_stat["cas_mean"])
        * 100.0
    )
    cas_red_central = (
        (s_dijk["cas_mean"] - s_gat["cas_mean"])
        / max(1e-6, s_dijk["cas_mean"])
        * 100.0
    )
    print(f"Casualty reduction vs Static NFPA: {cas_red_static:+.1f}% | Adverse outcome reduction: {red_static:+.1f}%")
    print(f"Casualty reduction vs Severed Dijkstra: {cas_red_central:+.1f}% | Adverse outcome reduction: {red_central:+.1f}%")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-TBA-GAT MAPPO Training & Benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=16,
        help="Number of rollout timesteps per PPO minibatch",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--min-evacuees", type=int, default=20, help="Minimum headcount per episode (e.g. 20)")
    parser.add_argument("--max-evacuees", type=int, default=600, help="Maximum headcount per episode (e.g. 600)")
    parser.add_argument("--curriculum", dest="curriculum", action="store_true", default=True, help="Enable 4-stage scenario curriculum (default: True)")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false", help="Disable 4-stage scenario curriculum")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Run only benchmark evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume model weights and curriculum at the episode after this checkpoint.",
    )
    args = parser.parse_args()

    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(args.save_dir, "best_policy.pt")
        run_benchmark(policy_path=ckpt, num_runs=5, device=args.device)
    else:
        trainer = MAPPOTrainer(
            lr=args.lr,
            ppo_batch_size=args.ppo_batch_size,
            device=args.device,
            save_dir=args.save_dir,
            min_evacuees=args.min_evacuees,
            max_evacuees=args.max_evacuees,
            curriculum=args.curriculum,
            resume_checkpoint=args.resume,
        )
        trainer.train(num_episodes=args.episodes)
        if args.eval:
            best_ckpt = os.path.join(args.save_dir, "best_policy.pt")
            run_benchmark(policy_path=best_ckpt, num_runs=5, device=args.device)
