"""
Unit Tests for Graph Laplacian Advection-Diffusion & ISO 13571 FED Toxicity.
"""

import numpy as np
from envs.hazard_simulator import HazardSimulator


def test_laplacian_mass_conservation():
    # 4-node ring graph
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=np.float64)

    # Set decay k=0 and zero external sources
    sim = HazardSimulator(num_nodes=4, adj_matrix=adj, diffusion_coeff=0.1, decay_rate=0.0, dt=1.0)
    sim.concentration[0] = 1.0 # 1.0 unit placed at node 0
    total_mass_init = np.sum(sim.concentration)

    for _ in range(20):
        sim.step()

    total_mass_end = np.sum(sim.concentration)
    assert np.isclose(total_mass_init, total_mass_end, atol=1e-5), f"Diffusion must conserve mass: {total_mass_init} vs {total_mass_end}"
    # Mass should spread evenly across all 4 nodes
    assert np.all(sim.concentration > 0.15)


def test_iso_13571_fed_accumulation():
    sim = HazardSimulator(num_nodes=2, fed_threshold=100.0, dt=1.0)

    # Constant exposure: C = 10.0 ppm for 10 seconds -> Delta FED = (10 / 100) * 1 = 0.1 per sec
    fed = 0.0
    for _ in range(10):
        fed += sim.compute_fed_delta(10.0)

    assert np.isclose(fed, 1.0, atol=1e-3), f"FED should reach 1.0 casualty threshold: {fed}"


if __name__ == "__main__":
    test_laplacian_mass_conservation()
    test_iso_13571_fed_accumulation()
    print("All hazard simulator and FED tests passed!")
