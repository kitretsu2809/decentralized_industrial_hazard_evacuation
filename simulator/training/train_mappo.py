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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        self.actions: List[np.ndarray] = []        # (N, K)
        self.action_log_probs: List[np.ndarray] = []  # (N,)
        self.rewards: List[float] = []             # scalar
        self.values: List[float] = []              # scalar
        self.dones: List[bool] = []                # scalar

    def add(
        self,
        node_features: np.ndarray,
        hidden_states: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        reward: float,
        value: float,
        done: bool,
    ):
        self.node_features.append(node_features)
        self.hidden_states.append(hidden_states)
        self.actions.append(actions)
        self.action_log_probs.append(action_log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

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
            non_terminal = 0.0 if self.dones[t] else 1.0
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
        device: Optional[str] = None,
        save_dir: str = "checkpoints",
        min_evacuees: int = 20,
        max_evacuees: int = 600,
        curriculum: bool = True,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.min_evacuees = min_evacuees
        self.max_evacuees = max_evacuees
        self.curriculum = curriculum
        self.env = env or IndustrialEvacuationEnv(num_evacuees=min_evacuees)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
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

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        self.edge_index = torch.tensor(
            self.env.edge_index, dtype=torch.long, device=self.device
        )
        self.action_masks = torch.tensor(
            self.env.get_all_action_masks(), dtype=torch.bool, device=self.device
        )

        self.buffer = RolloutBuffer(
            self.num_agents, self.obs_dim, self.max_corridors, self.hidden_dim
        )

    def train_episode(self, episode_idx: int) -> Dict[str, float]:
        """Runs a single episode rollout and performs PPO update."""
        # 1. Vary crowd density across [min_evacuees, max_evacuees] (20 to 600)
        if self.curriculum:
            max_curr = int(self.min_evacuees + (self.max_evacuees - self.min_evacuees) * min(1.0, episode_idx / 40.0))
            num_evac = random.randint(self.min_evacuees, max(self.min_evacuees + 1, max_curr))
        else:
            num_evac = random.randint(self.min_evacuees, self.max_evacuees)

        # 2. Comprehensive scenario sampling across all floors & disaster types
        candidates = [
            "tank_farm_a", "tank_farm_b", "reactor_1", "reactor_2",
            "hazmat_basin", "compressor_shed", "pipe_rack_junc_1", "pipe_rack_junc_2",
            "loading_bay", "pump_house", "corridor_f2_lab", "corridor_f3_mech",
            "stair_north_f1", "stair_south_f1"
        ]
        target_node = random.choice(candidates)
        hazard_type = random.choice(["GAS_RELEASE", "FIRE", "EXPLOSION", "CHEMICAL_SPILL"])
        intensity = random.uniform(0.75, 0.98)

        obs, infos = self.env.reset(options={
            "num_evacuees": num_evac,
            "node_id": target_node,
            "hazard_type": hazard_type,
            "intensity": intensity,
            "inject": True,
        })

        # Dual disaster injection in 30% of episodes
        if random.random() < 0.30:
            sec_target = random.choice([c for c in candidates if c != target_node])
            self.env.sim.inject_disaster(sec_target, random.choice(["FIRE", "GAS_RELEASE"]), random.uniform(0.65, 0.85))

        self.buffer.clear()

        hidden_state = torch.zeros(
            self.num_agents, self.hidden_dim, device=self.device
        )
        total_reward = 0.0
        step_count = 0

        while True:
            # Prepare batch tensor for all agents: (N, obs_dim)
            obs_array = np.stack([obs[agent] for agent in self.env.agents])
            node_features = torch.tensor(
                obs_array, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                actions_t, log_probs_t, new_hidden, val_t = self.policy.act(
                    node_features=node_features,
                    edge_index=self.edge_index,
                    hidden_state=hidden_state,
                    action_masks=self.action_masks,
                    deterministic=False,
                )

            actions_np = actions_t.cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            val_scalar = float(val_t.cpu().item())
            old_hidden_np = hidden_state.cpu().numpy()

            # Execute environment step
            action_dict = {
                agent: actions_np[i] for i, agent in enumerate(self.env.agents)
            }
            next_obs, rewards, terminations, truncations, infos = self.env.step(action_dict)

            reward = rewards[self.env.agents[0]]
            done = any(terminations.values()) or any(truncations.values())

            self.buffer.add(
                node_features=obs_array,
                hidden_states=old_hidden_np,
                actions=actions_np,
                action_log_probs=log_probs_np,
                reward=reward,
                value=val_scalar,
                done=done,
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
                last_val = float(v.cpu().item())

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=last_val, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        update_metrics = self._update_ppo(returns, advantages)

        metrics = self.env.sim.metrics()
        survival_rate = (metrics["evacuated"] / max(1, self.env.sim.num_evacuees)) * 100.0

        return {
            "episode": episode_idx,
            "reward": round(total_reward, 2),
            "steps": step_count,
            "evacuated": metrics["evacuated"],
            "casualties": metrics["casualties"],
            "survival_rate": round(survival_rate, 1),
            "sim_time": metrics["sim_time"],
            **update_metrics,
        }

    def _update_ppo(self, returns: np.ndarray, advantages: np.ndarray) -> Dict[str, float]:
        """Performs PPO policy and value clipping updates."""
        T = len(returns)
        if T == 0:
            return {"loss": 0.0, "pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}

        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.ppo_epochs):
            for t in range(T):
                nf = torch.tensor(
                    self.buffer.node_features[t], dtype=torch.float32, device=self.device
                )
                hs = torch.tensor(
                    self.buffer.hidden_states[t], dtype=torch.float32, device=self.device
                )
                act = torch.tensor(
                    self.buffer.actions[t], dtype=torch.long, device=self.device
                )
                old_lp = torch.tensor(
                    self.buffer.action_log_probs[t], dtype=torch.float32, device=self.device
                )

                val_pred, new_lp, entropy = self.policy.evaluate_actions(
                    node_features=nf,
                    edge_index=self.edge_index,
                    hidden_state=hs,
                    actions=act,
                    action_masks=self.action_masks,
                )

                # Mean ratio across agents: (N,)
                ratio = torch.exp(new_lp - old_lp)
                adv = adv_t[t]  # scalar advantage for step t

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                pi_loss = -torch.min(surr1, surr2).mean()

                # Value loss: (1,) vs target scalar
                v_loss = F.mse_loss(val_pred.squeeze(), ret_t[t])

                ent_loss = -entropy.mean()

                loss = pi_loss + self.value_coef * v_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_pi_loss += float(pi_loss.item())
                total_v_loss += float(v_loss.item())
                total_entropy += float(entropy.mean().item())

        denom = max(1, self.ppo_epochs * T)
        return {
            "pi_loss": round(total_pi_loss / denom, 4),
            "v_loss": round(total_v_loss / denom, 4),
            "entropy": round(total_entropy / denom, 4),
        }

    def train(self, num_episodes: int = 50, eval_interval: int = 10):
        os.makedirs(self.save_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"🚀 STARTING ST-TBA-GAT MAPPO TRAINING")
        print(f"   Episodes: {num_episodes} | Device: {self.device} | Architecture: Pure PyTorch ST-TBA-GAT")
        print("=" * 80 + "\n")

        best_survival = 0.0

        for ep in range(1, num_episodes + 1):
            t0 = time.time()
            metrics = self.train_episode(ep)
            dt = time.time() - t0

            surv = metrics["survival_rate"]
            rew = metrics["reward"]
            cas = metrics["casualties"]
            evac = metrics["evacuated"]

            print(
                f"Ep {ep:3d}/{num_episodes:3d} | "
                f"Reward: {rew:6.2f} | "
                f"Surv: {surv:5.1f}% | "
                f"Evac: {evac:2d} | "
                f"Cas: {cas:2d} | "
                f"Loss: (π={metrics['pi_loss']:.3f}, V={metrics['v_loss']:.3f}) | "
                f"Time: {dt:.2f}s"
            )

            # Checkpoint best model
            if surv > best_survival or ep == num_episodes:
                best_survival = max(best_survival, surv)
                best_path = os.path.join(self.save_dir, "best_policy.pt")
                self.policy.save_checkpoint(best_path, extra_meta=metrics)

            # Save latest checkpoint
            latest_path = os.path.join(self.save_dir, "st_tba_gat_latest.pt")
            self.policy.save_checkpoint(latest_path, extra_meta=metrics)

        print("\n" + "=" * 80)
        print(f"✅ TRAINING FINISHED! Best Survival Rate: {best_survival:.1f}%")
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
    print("📊 EMPIRICAL BENCHMARK: STATIC BASELINE vs CLASSICAL DIJKSTRA vs ST-TBA-GAT")
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

    edge_index = torch.tensor(env.edge_index, dtype=torch.long, device=device_obj)
    action_masks = torch.tensor(env.get_all_action_masks(), dtype=torch.bool, device=device_obj)

    scenarios = [
        {"node_id": "reactor_1", "hazard_type": "FIRE", "intensity": 0.95},
        {"node_id": "tank_farm_a", "hazard_type": "GAS_RELEASE", "intensity": 0.95},
        {"node_id": "hazmat_basin", "hazard_type": "CHEMICAL_SPILL", "intensity": 0.95},
        {"node_id": "compressor_shed", "hazard_type": "EXPLOSION", "intensity": 0.95},
        {"node_id": "pipe_rack_junc_1", "hazard_type": "FIRE", "intensity": 0.95},
    ]

    results = {"static": [], "dijkstra": [], "st_tba_gat": []}

    for run_idx in range(min(num_runs, len(scenarios))):
        scen = scenarios[run_idx]
        seed = 100 + run_idx

        # ── 1. Static NFPA Baseline ───────────────────────────────────────────
        # Static signage has no environmental sensing; paths never redirect around fire/gas
        env.reset(seed=seed, options={**scen, "policy_mode": "static"})
        static_act = {a: np.zeros(env.MAX_CORRIDORS, dtype=np.int64) for a in env.agents}
        while True:
            _, _, terms, truncs, _ = env.step(static_act)
            if any(terms.values()) or any(truncs.values()):
                break
        m_stat = env.sim.metrics()
        results["static"].append(m_stat)

        # ── 2. Classical Centralized Dijkstra ─────────────────────────────────
        # Centralized SCADA host at control_room; subject to 25% communication link severance
        env.reset(seed=seed, options={**scen, "policy_mode": "dijkstra"})
        rng = random.Random(seed)
        edges = list(env.sim.router.G.edges())
        # In an industrial disaster, cables incident to the ignition node burn out,
        # plus 15% random infrastructure link failures across the facility:
        severed_links = {e for e in edges if scen["node_id"] in e}
        severed_links.update(rng.sample(edges, int(len(edges) * 0.15)))
        
        # Determine nodes with active connection to central host (control_room)
        import networkx as nx
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
            dijk_act = {}
            signs = env.sim.compute_signboards()
            for agent in env.agents:
                nbrs = env.agent_neighbors.get(agent, [])
                acts = np.zeros(env.MAX_CORRIDORS, dtype=np.int64)
                # Nodes partitioned from central server lose real-time updates and revert to static (0)
                if agent in central_connected:
                    agent_signs = {s["target"]: s["action"] for s in signs.get(agent, [])}
                    for k, nbr in enumerate(nbrs[:env.MAX_CORRIDORS]):
                        act_name = agent_signs.get(nbr, "NORMAL")
                        if act_name == "BLOCKED":
                            acts[k] = 2
                        elif act_name == "CAUTION":
                            acts[k] = 1
                        else:
                            acts[k] = 0
                dijk_act[agent] = acts

            _, _, terms, truncs, _ = env.step(dijk_act)
            if any(terms.values()) or any(truncs.values()):
                break
        m_dijk = env.sim.metrics()
        results["dijkstra"].append(m_dijk)

        # ── 3. ST-TBA-GAT (Ours - Decentralized Edge GAT + Reflexive Safety) ──
        # Edge routers execute local 1-hop GAT inference and 20ms hardware reflexive safety override
        obs, _ = env.reset(seed=seed, options={**scen, "policy_mode": "marl"})
        hs = torch.zeros(len(env.agents), 64, device=device_obj)
        while True:
            obs_array = np.stack([obs[agent] for agent in env.agents])
            nf = torch.tensor(obs_array, dtype=torch.float32, device=device_obj)
            with torch.no_grad():
                acts_t, _, hs, _ = policy.act(
                    node_features=nf,
                    edge_index=edge_index,
                    hidden_state=hs,
                    action_masks=action_masks,
                    deterministic=True,
                )
            acts_np = acts_t.cpu().numpy()
            act_dict = {a: acts_np[i] for i, a in enumerate(env.agents)}
            obs, _, terms, truncs, _ = env.step(act_dict)
            if any(terms.values()) or any(truncs.values()):
                break
        m_gat = env.sim.metrics()
        results["st_tba_gat"].append(m_gat)

    # Aggregate metrics
    def summarize(runs):
        surv = [r["evacuated"] / max(1, num_evacuees) * 100.0 for r in runs]
        cas = [r["casualties"] for r in runs]
        t = [r["sim_time"] for r in runs]
        return {
            "surv_mean": np.mean(surv),
            "surv_std": np.std(surv),
            "cas_mean": np.mean(cas),
            "cas_std": np.std(cas),
            "time_mean": np.mean(t),
            "time_std": np.std(t),
        }

    s_stat = summarize(results["static"])
    s_dijk = summarize(results["dijkstra"])
    s_gat = summarize(results["st_tba_gat"])

    print("\n" + "-" * 85)
    print(f"{'METRIC':<25} | {'STATIC NFPA':<18} | {'CENTRALIZED DIJK':<18} | {'ST-TBA-GAT (OURS)':<18}")
    print("-" * 85)
    print(f"{'Survival Rate (%)':<25} | {s_stat['surv_mean']:5.1f} ± {s_stat['surv_std']:4.1f}%    | {s_dijk['surv_mean']:5.1f} ± {s_dijk['surv_std']:4.1f}%     | {s_gat['surv_mean']:5.1f} ± {s_gat['surv_std']:4.1f}%")
    print(f"{'Casualties (count)':<25} | {s_stat['cas_mean']:5.1f} ± {s_stat['cas_std']:4.1f}      | {s_dijk['cas_mean']:5.1f} ± {s_dijk['cas_std']:4.1f}       | {s_gat['cas_mean']:5.1f} ± {s_gat['cas_std']:4.1f}")
    print(f"{'Avg Egress Time (s)':<25} | {s_stat['time_mean']:5.1f} ± {s_stat['time_std']:4.1f}s     | {s_dijk['time_mean']:5.1f} ± {s_dijk['time_std']:4.1f}s      | {s_gat['time_mean']:5.1f} ± {s_gat['time_std']:4.1f}s")
    print("-" * 85)
    red_static = max(0.0, (s_stat['cas_mean'] - s_gat['cas_mean']) / max(1e-6, s_stat['cas_mean']) * 100.0)
    red_central = max(0.0, (s_dijk['cas_mean'] - s_gat['cas_mean']) / max(1e-6, s_dijk['cas_mean']) * 100.0)
    print(f"🌟 Relative Casualty Reduction vs Static NFPA:    {red_static:.1f}%")
    print(f"🌟 Relative Casualty Reduction vs Centralized:    {red_central:.1f}%")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-TBA-GAT MAPPO Training & Benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--min-evacuees", type=int, default=20, help="Minimum headcount per episode (e.g. 20)")
    parser.add_argument("--max-evacuees", type=int, default=600, help="Maximum headcount per episode (e.g. 600)")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable progressive density scaling")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Run only benchmark evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(args.save_dir, "best_policy.pt")
        run_benchmark(policy_path=ckpt, num_runs=5, device=args.device)
    else:
        trainer = MAPPOTrainer(
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir,
            min_evacuees=args.min_evacuees,
            max_evacuees=args.max_evacuees,
            curriculum=not args.no_curriculum,
        )
        trainer.train(num_episodes=args.episodes)
        if args.eval:
            best_ckpt = os.path.join(args.save_dir, "best_policy.pt")
            run_benchmark(policy_path=best_ckpt, num_runs=5, device=args.device)
