"""
Standardized Industrial Disaster Benchmark Scenarios (Phase 2 Mandate).

Defines 3 concrete, reproducible stress scenarios for peer-reviewed evaluation:
1. Scenario 1 (Reactor 2 Rupture + North Stairwell Thermal Flash):
   - N = 250 occupants.
   - Primary vertical egress (North Stairwell) severed at t = 10s.
   - Tests early dynamic diversion to prevent catastrophic doorway arching & crush stampede.

2. Scenario 2 (Multi-Source Chemical Cloud with Secondary Choke):
   - N = 300 occupants.
   - Dual chemical rupture: Chemical Storage + Loading Bay.
   - Severe corridor capacity choke on Floor 1.
   - Tests multi-exit load balancing and mixed-strategy flow splitting.

3. Scenario 3 (Ad-Hoc Mesh Disruption & Partial Comms Blackout):
   - N = 250 occupants.
   - 50% packet drop rate (p_loss = 0.50).
   - Direct thermal RF link cutoff within 10m of active thermal source.
   - Tests graceful degradation and local 1-hop autonomous resilience.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class DisasterScenario:
    name: str
    scenario_id: str
    num_pedestrians: int
    primary_sources: List[str]
    secondary_sources: List[str]
    secondary_trigger_time: float
    packet_loss_rate: float
    comm_radius: float
    thermal_cutoff_distance: float
    description: str


SCENARIOS: Dict[str, DisasterScenario] = {
    "scenario_1": DisasterScenario(
        name="Reactor 2 Rupture + North Stairwell Flash",
        scenario_id="scenario_1",
        num_pedestrians=250,
        primary_sources=["reactor_2", "corridor_f2_lab"],
        secondary_sources=["stair_north_f2", "stair_north_f1"],
        secondary_trigger_time=10.0,
        packet_loss_rate=0.10,
        comm_radius=18.0,
        thermal_cutoff_distance=8.0,
        description="Explosion at Reactor 2 spreads vertically. North Stairwell flashes at t=10s, severing primary egress.",
    ),
    "scenario_2": DisasterScenario(
        name="Multi-Source Chemical Cloud with Secondary Choke",
        scenario_id="scenario_2",
        num_pedestrians=300,
        primary_sources=["tank_farm_a", "loading_bay"],
        secondary_sources=["corridor_f1_central", "pipe_track_junc_1"],
        secondary_trigger_time=15.0,
        packet_loss_rate=0.15,
        comm_radius=18.0,
        thermal_cutoff_distance=10.0,
        description="Dual toxic releases at Tank Farm and Loading Bay create simultaneous converging plumes on Floor 1.",
    ),
    "scenario_3": DisasterScenario(
        name="Ad-Hoc Mesh Disruption & Partial Comms Blackout",
        scenario_id="scenario_3",
        num_pedestrians=250,
        primary_sources=["compressor_shed", "hazmat_basin"],
        secondary_sources=[],
        secondary_trigger_time=0.0,
        packet_loss_rate=0.50,
        comm_radius=12.0,
        thermal_cutoff_distance=15.0,
        description="Industrial fire triggers severe RF attenuation (50% packet drop) and complete mesh partition within 15m.",
    ),
}


def get_scenario(scenario_id: str) -> DisasterScenario:
    if scenario_id in SCENARIOS:
        return SCENARIOS[scenario_id]
    raise ValueError(f"Unknown scenario_id: {scenario_id}. Available: {list(SCENARIOS.keys())}")
