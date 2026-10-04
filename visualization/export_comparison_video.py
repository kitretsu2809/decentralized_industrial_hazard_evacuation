"""
Synchronized Split-Screen 1080p 60 FPS Video Exporter with Industrial Command HUD.

CLI Entrypoint:
    python -m visualization.export_comparison_video --seed 42 --output assets/demo_presentation.mp4
"""

import argparse
import math
import os
import subprocess
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from envs.evacuation_parallel_env import EvacuationParallelEnv
from agents.gat_policy import PermutationInvariantGATPolicy
from baselines.nfpa_heuristic import NFPAStaticRouter
from baselines.quickest_flow import QuickestFlowRouter
from visualization.render_engine import Isometric25DRenderer


def simulate_run(env, policy_type: str, policy=None, seed: int = 42, max_steps: int = 120):
    obs, infos = env.reset(seed=seed)
    nfpa_router = NFPAStaticRouter(env.G, env.exit_indices)
    qflow_router = QuickestFlowRouter(env.G, env.exit_indices, env.edge_info)

    history = []
    hazard_history = []
    hidden_state = None

    for step in range(max_steps):
        # Record agent states
        frame_agents = []
        for p in env.pedestrians:
            # Approximate current 3D position
            u = p.current_node
            pos_u = env.node_positions[u]
            if p.target_node is not None and p.edge_length > 0:
                pos_v = env.node_positions[p.target_node]
                frac = min(1.0, max(0.0, p.edge_progress / p.edge_length))
                x = (1.0 - frac) * pos_u[0] + frac * pos_v[0]
                y = (1.0 - frac) * pos_u[1] + frac * pos_v[1]
                z = (1.0 - frac) * pos_u[2] + frac * pos_v[2]
            else:
                x, y, z = pos_u

            frame_agents.append({
                'id': p.ped_id,
                'x': x,
                'y': y,
                'z': z,
                'fed': p.fed,
                'speed': p.speed,
                'is_casualty': p.is_casualty,
                'is_evacuated': p.is_evacuated
            })

        history.append(frame_agents)
        hazard_history.append(env.hazard_sim.concentration.copy())

        if not env.agents:
            break

        # Select actions
        hazard_conc = env.hazard_sim.concentration
        node_occ = np.zeros(env.num_nodes)
        for p in env.pedestrians:
            if not p.is_evacuated and not p.is_casualty:
                node_occ[p.current_node] += 1.0

        if policy_type == 'nfpa_static':
            actions = nfpa_router.get_actions(env.agents, env.neighbors)
        elif policy_type == 'gat_marl':
            if policy is not None:
                node_feats = torch.tensor(env._get_node_features(), dtype=torch.float32)
                edge_list = list(env.G.edges())
                u_list, v_list = [], []
                for u_e, v_e in edge_list:
                    u_list.extend([u_e, v_e])
                    v_list.extend([v_e, u_e])
                edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
                edge_attr = torch.ones((edge_index.size(1), 3), dtype=torch.float32)

                action_masks = torch.zeros((env.num_nodes, env.max_corridors), dtype=torch.float32)
                for agent in env.agents:
                    node_idx = int(agent.split('_')[1])
                    nbr_count = len(env.neighbors[node_idx])
                    action_masks[node_idx, :min(nbr_count, env.max_corridors)] = 1.0

                nbr_hazards = torch.zeros((env.num_nodes, env.max_corridors), dtype=torch.float32)
                for agent in env.agents:
                    n_idx = int(agent.split('_')[1])
                    for s, nbr in enumerate(env.neighbors[n_idx][:env.max_corridors]):
                        nbr_hazards[n_idx, s] = float(env.hazard_sim.concentration[nbr])

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
                actions = qflow_router.get_actions(env.agents, env.neighbors, hazard_conc, node_occ)
        else:
            actions = {a: 0 for a in env.agents}

        obs, rewards, terminations, truncations, infos = env.step(actions)

    return history, hazard_history, env.num_pedestrians


def draw_hud(draw, width, height, t_sec, left_stats, right_stats):
    # Header Banner
    draw.rectangle([(0, 0), (width, 80)], fill=(15, 23, 42))
    draw.line([(0, 80), (width, 80)], fill=(51, 65, 85), width=2)
    # Divider between left and right panes
    draw.line([(width // 2, 80), (width // 2, height - 120)], fill=(51, 65, 85), width=2)

    # Title & Clock
    draw.text((30, 25), "INDUSTRIAL DIGITAL TWIN: EVACUATION SIMULATOR", fill=(248, 250, 252))
    time_str = f"T = {t_sec:05.1f}s | 60 FPS ISO 13571"
    draw.text((width // 2 - 120, 25), time_str, fill=(56, 189, 248))
    draw.text((width - 320, 25), "STATUS: TOXIC RELEASE ACTIVE", fill=(239, 68, 68))

    # Left Pane Header
    draw.rectangle([(20, 95), (400, 135)], fill=(30, 41, 59))
    draw.text((35, 105), "BASELINE: NFPA 101 STATIC SIGNAGE", fill=(226, 232, 240))

    # Right Pane Header
    draw.rectangle([(width // 2 + 20, 95), (width // 2 + 450, 135)], fill=(30, 41, 59))
    draw.text((width // 2 + 35, 105), "PROPOSED: DECENTRALIZED GAT-MARL", fill=(52, 211, 153))

    # Bottom Telemetry Dashboard
    draw.rectangle([(0, height - 120), (width, height)], fill=(15, 23, 42))
    draw.line([(0, height - 120), (width, height - 120)], fill=(51, 65, 85), width=2)

    # Left Stats
    l_evac, l_cas, l_tot = left_stats['evac'], left_stats['cas'], left_stats['total']
    l_pct = (l_evac / max(1, l_tot)) * 100.0
    l_text = f"Evacuated: {l_evac:02d}/{l_tot:02d} ({l_pct:5.1f}%) | Casualties: {l_cas:02d} | Mean FED: {left_stats['fed']:.2f}"
    draw.text((40, height - 85), l_text, fill=(248, 250, 252))

    # Progress bar Left
    draw.rectangle([(40, height - 50), (40 + 350, height - 30)], fill=(51, 65, 85))
    draw.rectangle([(40, height - 50), (40 + int(350 * (l_evac / max(1, l_tot))), height - 30)], fill=(239, 68, 68) if l_cas > 5 else (234, 179, 8))

    # Right Stats
    r_evac, r_cas, r_tot = right_stats['evac'], right_stats['cas'], right_stats['total']
    r_pct = (r_evac / max(1, r_tot)) * 100.0
    r_text = f"Evacuated: {r_evac:02d}/{r_tot:02d} ({r_pct:5.1f}%) | Casualties: {r_cas:02d} | Mean FED: {right_stats['fed']:.2f}"
    draw.text((width // 2 + 40, height - 85), r_text, fill=(248, 250, 252))

    # Progress bar Right
    draw.rectangle([(width // 2 + 40, height - 50), (width // 2 + 40 + 350, height - 30)], fill=(51, 65, 85))
    draw.rectangle([(width // 2 + 40, height - 50), (width // 2 + 40 + int(350 * (r_evac / max(1, r_tot))), height - 30)], fill=(34, 197, 94))

    # Legend
    legend_x = width - 420
    draw.text((legend_x, height - 80), "LEGEND:", fill=(148, 163, 184))
    draw.ellipse([(legend_x + 60, height - 80), (legend_x + 72, height - 68)], fill=(34, 197, 94))
    draw.text((legend_x + 78, height - 80), "Nominal", fill=(226, 232, 240))

    draw.ellipse([(legend_x + 145, height - 80), (legend_x + 157, height - 68)], fill=(234, 179, 8))
    draw.text((legend_x + 163, height - 80), "Choked", fill=(226, 232, 240))

    draw.ellipse([(legend_x + 225, height - 80), (legend_x + 237, height - 68)], fill=(239, 68, 68))
    draw.text((legend_x + 243, height - 80), "Toxic FED", fill=(226, 232, 240))

    draw.ellipse([(legend_x + 315, height - 80), (legend_x + 327, height - 68)], fill=(100, 116, 139))
    draw.text((legend_x + 333, height - 80), "Casualty", fill=(226, 232, 240))


def render_pane(draw, renderer, env, agents, hazards):
    # 1. Draw Multi-Story Floor Slabs (Semi-transparent background polygons)
    # Floor 1 (z=0)
    slab1_corners = [(20.0, 0.0, 0.0), (220.0, 0.0, 0.0), (220.0, 120.0, 0.0), (20.0, 120.0, 0.0)]
    slab1_proj = [renderer.project(x, y, z) for x, y, z in slab1_corners]
    draw.polygon(slab1_proj, fill=(24, 32, 47), outline=(45, 60, 85))

    # Floor 2 (z=4.5)
    slab2_corners = [(75.0, 15.0, 4.5), (225.0, 15.0, 4.5), (225.0, 105.0, 4.5), (75.0, 105.0, 4.5)]
    slab2_proj = [renderer.project(x, y, z) for x, y, z in slab2_corners]
    draw.polygon(slab2_proj, fill=(20, 27, 45), outline=(59, 130, 246))

    # 2. Draw Vertical Shafts (Stairwells & Hoist) connecting Floor 1 and Floor 2
    for node_idx in env.stairwell_indices:
        pos = env.node_positions[node_idx]
        p_base = renderer.project(pos[0], pos[1], 0.0)
        p_top = renderer.project(pos[0], pos[1], 4.5)
        draw.line([p_base, p_top], fill=(96, 165, 250), width=2)

    # 3. Draw Corridor Edges
    for (u, v) in env.edge_info.keys():
        if u < v:
            p_u = renderer.project(*env.node_positions[u])
            p_v = renderer.project(*env.node_positions[v])
            draw.line([p_u, p_v], fill=(51, 65, 85), width=2)

    # 4. Draw Gas Dispersion Heatmap Plumes
    for node_idx, conc in enumerate(hazards):
        if conc > 0.05:
            pos = env.node_positions[node_idx]
            px, py = renderer.project(*pos)
            radius = int(min(60, 12 + conc * 45))
            # Orange-red expanding gas cloud
            draw.ellipse(
                [(px - radius, py - radius), (px + radius, py + radius)],
                fill=(220, 38, 38) if conc > 0.5 else (245, 158, 11),
                outline=(239, 68, 68)
            )

    # 5. Draw Exit Markers
    for ex in env.exit_indices:
        px, py = renderer.project(*env.node_positions[ex])
        draw.rectangle([(px - 8, py - 8), (px + 8, py + 8)], fill=(34, 197, 94), outline=(255, 255, 255), width=2)

    # 6. Draw Agents with Status Colors
    for ag in agents:
        if ag['is_evacuated']:
            continue
        px, py = renderer.project(ag['x'], ag['y'], ag['z'])
        color_hex = renderer.get_agent_color(ag)
        # Parse hex to tuple
        h = color_hex.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        r = 4 if not ag['is_casualty'] else 3
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=rgb, outline=(0, 0, 0))


def export_video(seed: int = 42, output_path: str = "assets/demo_presentation.mp4", max_steps: int = 80, checkpoint_path: str = "checkpoints/best_policy.pt", num_pedestrians: int = 150):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 1920, 1080

    print(f"[Video Exporter] Initializing simulation environment (Seed {seed}, Evacuees {num_pedestrians})...")
    env = EvacuationParallelEnv(num_pedestrians=num_pedestrians, max_steps=max_steps, dt=1.0)

    # Load GAT policy if available
    policy = PermutationInvariantGATPolicy(node_dim=5, edge_dim=3, hidden_dim=64, max_corridors=6, heads=4)
    if os.path.exists(checkpoint_path):
        try:
            policy.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"[Video Exporter] Loaded GAT policy from {checkpoint_path}")
        except Exception as e:
            print(f"[Video Exporter] Warning loading checkpoint: {e}")
    policy.eval()

    print("[Video Exporter] Running Left Pane (NFPA Static Baseline)...")
    left_hist, left_hazards, total_peds = simulate_run(env, 'nfpa_static', policy=None, seed=seed, max_steps=max_steps)

    print("[Video Exporter] Running Right Pane (GAT-MARL Decentralized Policy)...")
    right_hist, right_hazards, _ = simulate_run(env, 'gat_marl', policy=policy, seed=seed, max_steps=max_steps)

    renderer_left = Isometric25DRenderer(width=960, height=1080, x_offset=460, y_offset=640)
    renderer_right = Isometric25DRenderer(width=960, height=1080, x_offset=1420, y_offset=640)

    # Sub-step interpolation (60 FPS = 6 sub-steps per 1.0s dt)
    substeps = 6
    interp_left = renderer_left.interpolate_pedestrian_trajectories(left_hist, num_substeps=substeps)
    interp_right = renderer_right.interpolate_pedestrian_trajectories(right_hist, num_substeps=substeps)


    total_frames = min(len(interp_left), len(interp_right))
    print(f"[Video Exporter] Generating {total_frames} 1080p 60 FPS frames via ffmpeg pipe...")

    # Launch ffmpeg process
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f"{width}x{height}",
        '-pix_fmt', 'rgb24',
        '-r', '60',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '18',
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for f_idx in range(total_frames):
        t_sec = f_idx / 60.0
        discrete_t = min(len(left_hazards) - 1, f_idx // substeps)

        agents_l = interp_left[f_idx]
        agents_r = interp_right[f_idx]
        haz_l = left_hazards[discrete_t]
        haz_r = right_hazards[discrete_t]

        # Compute stats for HUD
        evac_l = sum(1 for a in agents_l if a['is_evacuated'])
        cas_l = sum(1 for a in agents_l if a['is_casualty'])
        fed_l = float(np.mean([a['fed'] for a in agents_l])) if agents_l else 0.0

        evac_r = sum(1 for a in agents_r if a['is_evacuated'])
        cas_r = sum(1 for a in agents_r if a['is_casualty'])
        fed_r = float(np.mean([a['fed'] for a in agents_r])) if agents_r else 0.0

        # Create canvas
        img = Image.new('RGB', (width, height), color=(10, 15, 30))
        draw = ImageDraw.Draw(img)

        # Render Left & Right isometric multi-story panes
        render_pane(draw, renderer_left, env, agents_l, haz_l)
        render_pane(draw, renderer_right, env, agents_r, haz_r)

        # Render Industrial Command HUD
        draw_hud(
            draw, width, height, t_sec,
            {'evac': evac_l, 'cas': cas_l, 'total': total_peds, 'fed': fed_l},
            {'evac': evac_r, 'cas': cas_r, 'total': total_peds, 'fed': fed_r}
        )

        # Pipe frame
        proc.stdin.write(img.tobytes())

        if (f_idx + 1) % 60 == 0:
            print(f"  -> Rendered {f_idx + 1}/{total_frames} frames ({((f_idx + 1)/total_frames)*100:.1f}%)")

    proc.stdin.close()
    proc.wait()
    print(f"[Video Exporter] Video successfully exported to {output_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-Screen 1080p 60 FPS Evacuation Comparison Video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="assets/demo_presentation.mp4", help="Output MP4 file path")
    parser.add_argument("--steps", type=int, default=70, help="Simulation duration in seconds")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt", help="Path to trained GAT model checkpoint")
    parser.add_argument("--pedestrians", type=int, default=150, help="Total crowd count N (e.g. 50, 150, 300, 600)")
    args = parser.parse_args()

    export_video(seed=args.seed, output_path=args.output, max_steps=args.steps, checkpoint_path=args.checkpoint, num_pedestrians=args.pedestrians)

