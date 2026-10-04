"""
Unit Tests for Weidmann Fundamental Diagram & Bottleneck Capping.
"""

import numpy as np
from envs.fundamental_diagram import FundamentalDiagram, WeidmannParams


def test_weidmann_limits():
    fd = FundamentalDiagram()
    # 1. Zero density -> Free flow velocity v0 = 1.34
    assert np.isclose(fd.velocity(0.0), 1.34, atol=1e-3)

    # 2. Jam density (5.4 ped/m^2) -> 0.0 m/s
    assert np.isclose(fd.velocity(5.4), 0.0, atol=1e-3)
    assert np.isclose(fd.velocity(6.0), 0.0, atol=1e-3)

    # 3. Intermediate density
    v_mid = fd.velocity(2.0)
    assert 0.0 < v_mid < 1.34


def test_weidmann_monotonicity():
    fd = FundamentalDiagram()
    densities = np.linspace(0.1, 5.3, 50)
    velocities = fd.velocity(densities)

    # Speeds must strictly decrease as density increases
    diffs = np.diff(velocities)
    assert np.all(diffs <= 1e-5), "Velocity must decrease monotonically with density"


def test_bottleneck_capping():
    fd = FundamentalDiagram()
    # Doorway width = 1.0m, specific capacity = 1.33 ped/(m*s) -> doorway flow = 1.33 ped/s
    # Edge capacity = 5.0 ped/s -> Capped at 1.33 ped/s
    q_cap = fd.bottleneck_capacity(edge_capacity=5.0, doorway_width=1.0)
    assert np.isclose(q_cap, 1.33, atol=1e-2)

    # Edge capacity = 1.0 ped/s, doorway width = 2.0m (doorway flow = 2.66) -> Capped at 1.0 ped/s
    q_cap_edge = fd.bottleneck_capacity(edge_capacity=1.0, doorway_width=2.0)
    assert np.isclose(q_cap_edge, 1.0, atol=1e-2)


if __name__ == "__main__":
    test_weidmann_limits()
    test_weidmann_monotonicity()
    test_bottleneck_capping()
    print("All fundamental diagram tests passed!")
