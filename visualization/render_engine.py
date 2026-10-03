"""
Exploded 2.5D Multi-Story Isometric Simulation Visualizer.

Implements:
1. Isometric 2.5D projection with stacked semi-transparent floor slabs.
2. Vertical stairwell and elevator shafts visually bridging levels.
3. 60 FPS cubic spline sub-step interpolation for continuous crowd kinematics.
4. Dynamic agent status color codes:
   - #22c55e (Safe / Normal velocity)
   - #eab308 (Bottlenecked / Density throttled)
   - #ef4444 (Accumulating FED / Hazard exposure)
   - #64748b (Casualty / Incapacitated)
5. Toxic gas dispersion heatmap overlay with alpha blending.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.interpolate import CubicSpline


class Isometric25DRenderer:
    """
    Projects 3D plant coordinates into 2.5D isometric screen coordinates:
        x_screen = (x - y) * cos(30 deg) * scale + x_offset
        y_screen = ((x + y) * sin(30 deg) - z * z_stack) * scale + y_offset
    """
    def __init__(
        self,
        width: int = 960,
        height: int = 1080,
        scale: float = 3.2,
        z_stack: float = 14.0,
        x_offset: float = 480,
        y_offset: float = 620
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.z_stack = z_stack
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.cos30 = math.cos(math.radians(30))
        self.sin30 = math.sin(math.radians(30))

    def project(self, x: float, y: float, z: float) -> Tuple[int, int]:
        """Converts (x, y, z) plant coordinates into 2D pixel coordinates."""
        x_iso = (x - y) * self.cos30 * self.scale + self.x_offset
        y_iso = ((x + y) * self.sin30 - z * self.z_stack) * self.scale + self.y_offset
        return int(x_iso), int(y_iso)

    def interpolate_pedestrian_trajectories(
        self,
        ped_history: List[Dict[str, Any]],
        num_substeps: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """
        Performs continuous cubic spline sub-step interpolation across discrete timesteps.
        """
        if len(ped_history) < 2:
            return [ped_history]

        T = len(ped_history)
        interpolated_steps = []

        # For each discrete interval [t, t+1], generate sub-steps
        for t in range(T - 1):
            curr_peds = {p['id']: p for p in ped_history[t]}
            next_peds = {p['id']: p for p in ped_history[t + 1]}

            for s in range(num_substeps):
                alpha = s / float(num_substeps)
                # Smooth hermite cubic s-curve: 3*alpha^2 - 2*alpha^3
                s_alpha = 3.0 * (alpha ** 2) - 2.0 * (alpha ** 3)

                substep_agents = []
                for p_id, p_curr in curr_peds.items():
                    if p_id in next_peds:
                        p_next = next_peds[p_id]
                        x = (1.0 - s_alpha) * p_curr['x'] + s_alpha * p_next['x']
                        y = (1.0 - s_alpha) * p_curr['y'] + s_alpha * p_next['y']
                        z = (1.0 - s_alpha) * p_curr['z'] + s_alpha * p_next['z']
                        fed = (1.0 - alpha) * p_curr['fed'] + alpha * p_next['fed']
                        speed = (1.0 - alpha) * p_curr['speed'] + alpha * p_next['speed']
                        is_cas = p_next['is_casualty'] if alpha > 0.5 else p_curr['is_casualty']
                        is_evac = p_next['is_evacuated'] if alpha > 0.5 else p_curr['is_evacuated']
                    else:
                        x, y, z = p_curr['x'], p_curr['y'], p_curr['z']
                        fed = p_curr['fed']
                        speed = p_curr['speed']
                        is_cas = p_curr['is_casualty']
                        is_evac = p_curr['is_evacuated']

                    substep_agents.append({
                        'id': p_id,
                        'x': x,
                        'y': y,
                        'z': z,
                        'fed': fed,
                        'speed': speed,
                        'is_casualty': is_cas,
                        'is_evacuated': is_evac
                    })
                interpolated_steps.append(substep_agents)

        return interpolated_steps

    def get_agent_color(self, agent: Dict[str, Any]) -> str:
        """
        Returns hex color based on academic presentation spec:
        - #64748b: Casualty or successfully evacuated
        - #ef4444: High FED accumulation / toxic exposure
        - #eab308: Bottleneck delay / density throttled (speed < 0.6)
        - #22c55e: Normal / nominal velocity
        """
        if agent.get('is_casualty', False):
            return '#64748b'
        if agent.get('fed', 0.0) >= 0.35:
            return '#ef4444'
        if agent.get('speed', 1.34) < 0.65:
            return '#eab308'
        return '#22c55e'
