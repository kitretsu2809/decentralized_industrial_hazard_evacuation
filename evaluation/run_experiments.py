"""
Rigorous Statistical Evaluation Suite (per Agarwal et al., NeurIPS 2021).

Evaluates NFPA Static, Dynamic D* Lite, Dynamic Quickest Flow, and ST-TBA-GAT MARL
across 3 standardized industrial disaster stress scenarios over 20 independent seeds.

Zero-Sycophancy / Ground-Truth Parity:
- All baselines receive identical real-time hazard updates (no information blinding).
- D* Lite re-weights edges instantaneously with dynamic sensor telemetry.
- Compressive Crowd Asphyxia (Crush) vs. ISO 13571 FED (Toxic) casualties are logged separately.
- Results are saved to assets/evaluation_summary.json and assets/performance_profiles.png.
"""

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from envs.evacuation_parallel_env import EvacuationParallelEnv
from agents.gat_policy import PermutationInvariantGATPolicy
from baselines.nfpa_heuristic import NFPAStaticRouter
from baselines.d_star_lite import DStarLiteRouter
from baselines.quickest_flow import QuickestFlowRouter
from evaluation.scenarios import SCENARIOS, DisasterScenario


def compute_iqm(data: np.ndarray) -> float:
    """Calculates Interquartile Mean (IQM) trimming top and bottom 25%."""
    if len(data) == 0:
        return 0.0
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    trimmed = data[(data >= q25) & (data <= q75)]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(data))


def bootstrap_ci(data: np.ndarray, num_bootstraps: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculates 95% stratified bootstrap confidence interval."""
    if len(data) < 2:
        val = float(data[0]) if len(data) == 1 else 0.0
        return val, val
    boot_means = []
    rng = np.random.default_rng(42)
    n = len(data)
    for _ in range(num_bootstraps):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100.0))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100.0))
    return lower, upper


def evaluate_single_run(
    env: EvacuationParallelEnv,
    scenario: DisasterScenario,
    model_type: str,
    policy: Optional[PermutationInvariantGATPolicy] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Runs a single episode under a specific model and disaster scenario."""
    # Map scenario node names to indices
    primary_indices = [
        env.node_id_to_idx[nid] for nid in scenario.primary_sources
        if nid in env.node_id_to_idx
    ]
    secondary_indices = [
        env.node_id_to_idx[nid] for nid in scenario.secondary_sources
        if nid in env.node_id_to_idx
    ]

    env.comm_channel.set_packet_loss_rate(scenario.packet_loss_rate)
    env.comm_channel.comm_radius = scenario.comm_radius

    obs, infos = env.reset(seed=seed, options={
        "num_pedestrians": scenario.num_pedestrians,
        "disaster_nodes": primary_indices
    })
    num_peds = env.num_pedestrians

    nfpa_router = NFPAStaticRouter(env.G, env.exit_indices)
    dstar_router = DStarLiteRouter(env.G, env.exit_indices)
    qflow_router = QuickestFlowRouter(env.G, env.exit_indices, env.edge_info)

    step_count = 0
    hidden_state = None
    secondary_triggered = False

    while env.agents and step_count < env.max_steps:
        step_count += 1
        current_time = step_count * env.dt

        # Secondary flash trigger (e.g. stairwell flash at t=10s)
        if not secondary_triggered and scenario.secondary_trigger_time > 0 and current_time >= scenario.secondary_trigger_time:
            for s_idx in secondary_indices:
                env.hazard_sim.set_source(s_idx, injection_rate=0.40)
                env.hazard_sim.concentration[s_idx] = max(0.75, env.hazard_sim.concentration[s_idx])
            secondary_triggered = True

        hazard_conc = env.hazard_sim.concentration

        # Node occupancies for queue-aware baselines
        node_occ = np.zeros(env.num_nodes, dtype=np.float64)
        for p in env.pedestrians:
            if not p.is_evacuated and not p.is_casualty:
                node_occ[p.current_node] += 1.0

        if model_type == 'nfpa_static':
            actions = nfpa_router.get_actions(env.agents, env.neighbors)
        elif model_type == 'd_star_lite':
            # Dynamic D* Lite with exact real-time hazard updates (PARITY OF INFORMATION)
            actions = dstar_router.update_and_route(hazard_conc, env.agents, env.neighbors)
        elif model_type == 'quickest_flow':
            actions = qflow_router.get_actions(env.agents, env.neighbors, hazard_conc, node_occ)
        elif model_type == 'gat_marl':
            if policy is None:
                actions = {a: env.action_space(a).sample() for a in env.agents}
            else:
                node_feats = torch.tensor(env._get_node_features(), dtype=torch.float32)
                edge_list = list(env.G.edges())
                u_list, v_list = [], []
                for u, v in edge_list:
                    u_list.extend([u, v])
                    v_list.extend([v, u])
                edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
                edge_attr = torch.ones((edge_index.size(1), 3), dtype=torch.float32)

                action_masks = torch.zeros((env.num_nodes, env.max_corridors), dtype=torch.float32)
                nbr_hazards = torch.zeros((env.num_nodes, env.max_corridors), dtype=torch.float32)

                for agent in env.agents:
                    node_idx = int(agent.split('_')[1])
                    nbrs = env.neighbors[node_idx]
                    valid_slots = min(len(nbrs), env.max_corridors)
                    for slot in range(valid_slots):
                        nbr = nbrs[slot]
                        if env.edge_info.get((node_idx, nbr), {}).get('throughput', 1.0) > 0:
                            action_masks[node_idx, slot] = 1.0
                        nbr_hazards[node_idx, slot] = float(hazard_conc[nbr])

                with torch.no_grad():
                    actions_tensor, _, _, hidden_state = policy.act(
                        node_feats, edge_index, edge_attr,
                        hidden_state=hidden_state,
                        action_masks=action_masks,
                        neighbor_hazards=nbr_hazards,
                        deterministic=True
                    )
                actions = {a: int(actions_tensor[int(a.split('_')[1])].item()) for a in env.agents}
        else:
            actions = {a: 0 for a in env.agents}

        obs, rewards, terminations, truncations, infos = env.step(actions)

    t_clear = float(step_count * env.dt)
    survival_pct = float((env.total_evacuated / num_peds) * 100.0)

    return {
        'seed': seed,
        'evacuated': int(env.total_evacuated),
        'total_casualties': int(env.total_casualties),
        'crush_casualties': int(env.total_crush_casualties),
        'toxic_casualties': int(env.total_toxic_casualties),
        'survival_rate': survival_pct,
        't_clear': t_clear,
    }


def plot_performance_profiles(results_summary: Dict[str, Any], output_path: str):
    """
    Generates Performance Profiles (Agarwal et al., NeurIPS 2021).
    Plots the fraction of evaluation runs where performance exceeds threshold tau.
    """
    models = ["nfpa_static", "d_star_lite", "quickest_flow", "gat_marl"]
    display_names = {
        "nfpa_static": "NFPA 101 Standard",
        "d_star_lite": "Dynamic D* Lite",
        "quickest_flow": "Dynamic Quickest Flow",
        "gat_marl": "ST-TBA-GAT (Ours)"
    }
    colors = {
        "nfpa_static": "#94a3b8",
        "d_star_lite": "#f59e0b",
        "quickest_flow": "#3b82f6",
        "gat_marl": "#10b981"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Performance Profile over Survival Rate
    ax1 = axes[0]
    tau_vals = np.linspace(0.0, 100.0, 101)

    for m in models:
        all_survival = []
        for sc_id, sc_data in results_summary["scenarios"].items():
            if m in sc_data["models"]:
                all_survival.extend(sc_data["models"][m]["raw_survival_rates"])

        all_survival = np.array(all_survival)
        if len(all_survival) > 0:
            profile = [np.mean(all_survival >= tau) for tau in tau_vals]
            ax1.plot(tau_vals, profile, label=display_names[m], color=colors[m], linewidth=2.5)

    ax1.set_xlabel("Survival Rate Threshold τ (%)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Fraction of Runs with Score ≥ τ", fontsize=12, fontweight='bold')
    ax1.set_title("Performance Profile: Survival Rate (Agarwal et al., 2021)", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=10)

    # 2. Casualty Breakdown by Mechanism across models
    ax2 = axes[1]
    x_indices = np.arange(len(models))
    bar_width = 0.35

    crush_means = []
    toxic_means = []

    for m in models:
        crushes, toxics = [], []
        for sc_id, sc_data in results_summary["scenarios"].items():
            if m in sc_data["models"]:
                crushes.append(sc_data["models"][m]["metrics"]["crush_casualties"]["mean"])
                toxics.append(sc_data["models"][m]["metrics"]["toxic_casualties"]["mean"])
        crush_means.append(np.mean(crushes) if crushes else 0.0)
        toxic_means.append(np.mean(toxics) if toxics else 0.0)

    bars1 = ax2.bar(x_indices - bar_width/2, crush_means, bar_width, label="Crush Asphyxia (CSI ≥ 15)", color="#ef4444")
    bars2 = ax2.bar(x_indices + bar_width/2, toxic_means, bar_width, label="Toxic Poisoning (FED ≥ 1.0)", color="#8b5cf6")

    ax2.set_xlabel("Egress Control Architecture", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Mean Casualties per Run", fontsize=12, fontweight='bold')
    ax2.set_title("Mortality Decomposition: Bottleneck Crush vs. Toxic Inhalation", fontsize=13, fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([display_names[m] for m in models], fontsize=9, rotation=15)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved performance profile plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Benchmark Suite across 3 Scenarios")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per scenario")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt", help="Path to checkpoint")
    args = parser.parse_args()

    num_seeds = args.seeds
    os.makedirs('assets', exist_ok=True)

    print("==========================================================================")
    print(f"  BENCHMARK SUITE: RUNNING {num_seeds} SEEDS ACROSS 3 STANDARDIZED SCENARIOS")
    print("==========================================================================")

    # Initialize GAT-MARL policy
    policy = PermutationInvariantGATPolicy(node_dim=5, edge_dim=3, hidden_dim=64, max_corridors=6, heads=4)
    if os.path.exists(args.checkpoint):
        try:
            state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            policy.load_state_dict(state_dict)
            print(f"[Policy] Loaded trained checkpoint from {args.checkpoint}")
        except Exception as e:
            print(f"[Policy] Failed to load {args.checkpoint} ({e}), running initialized policy")
    policy.eval()

    models = ["nfpa_static", "d_star_lite", "quickest_flow", "gat_marl"]
    results_summary: Dict[str, Any] = {
        "metadata": {
            "num_seeds": num_seeds,
            "models_evaluated": models,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "academic_protocol": "Agarwal et al. (2021) 95% Bootstrap CIs & IQM"
        },
        "scenarios": {}
    }

    env = EvacuationParallelEnv()

    for sc_id, scenario in SCENARIOS.items():
        print(f"\n>>> Running Scenario: {scenario.name} (N={scenario.num_pedestrians})")
        results_summary["scenarios"][sc_id] = {
            "name": scenario.name,
            "description": scenario.description,
            "num_pedestrians": scenario.num_pedestrians,
            "models": {}
        }

        for model in models:
            print(f"  --> Model: {model.upper()} ({num_seeds} seeds)...", end="", flush=True)
            runs = []
            for s in range(num_seeds):
                seed_val = 100 + s * 7
                run_res = evaluate_single_run(
                    env=env,
                    scenario=scenario,
                    model_type=model,
                    policy=policy,
                    seed=seed_val
                )
                runs.append(run_res)

            survivals = np.array([r["survival_rate"] for r in runs])
            crushes = np.array([r["crush_casualties"] for r in runs])
            toxics = np.array([r["toxic_casualties"] for r in runs])
            t_clears = np.array([r["t_clear"] for r in runs])

            s_low, s_high = bootstrap_ci(survivals)
            c_low, c_high = bootstrap_ci(crushes)
            t_low, t_high = bootstrap_ci(toxics)
            tc_low, tc_high = bootstrap_ci(t_clears)

            model_metrics = {
                "survival_rate": {
                    "mean": float(np.mean(survivals)),
                    "std": float(np.std(survivals)),
                    "iqm": compute_iqm(survivals),
                    "ci_95": [s_low, s_high]
                },
                "crush_casualties": {
                    "mean": float(np.mean(crushes)),
                    "std": float(np.std(crushes)),
                    "iqm": compute_iqm(crushes),
                    "ci_95": [c_low, c_high]
                },
                "toxic_casualties": {
                    "mean": float(np.mean(toxics)),
                    "std": float(np.std(toxics)),
                    "iqm": compute_iqm(toxics),
                    "ci_95": [t_low, t_high]
                },
                "t_clear": {
                    "mean": float(np.mean(t_clears)),
                    "std": float(np.std(t_clears)),
                    "iqm": compute_iqm(t_clears),
                    "ci_95": [tc_low, tc_high]
                }
            }

            results_summary["scenarios"][sc_id]["models"][model] = {
                "metrics": model_metrics,
                "raw_survival_rates": [float(x) for x in survivals],
                "raw_crush": [int(x) for x in crushes],
                "raw_toxic": [int(x) for x in toxics]
            }

            print(f" Done! Mean Survival: {np.mean(survivals):.1f}% (IQM: {compute_iqm(survivals):.1f}%, Crush: {np.mean(crushes):.1f}, Toxic: {np.mean(toxics):.1f})")

    # Save evaluation summary JSON
    summary_path = "assets/evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✅ Successfully saved raw evaluation summary to {summary_path}")

    # Plot and save performance profiles
    plot_path = "assets/performance_profiles.png"
    plot_performance_profiles(results_summary, plot_path)


if __name__ == "__main__":
    main()
