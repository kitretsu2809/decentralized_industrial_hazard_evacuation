import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from core.graph.building_graph import BuildingGraph
from core.graph.types import EdgeState
from core.hazard.cross_floor_propagation import compute_floor_hazard_map, PropagationConfig

class IndustrialHazardModel:
    """
    Industrial Multi-Disaster Hazard Engine.
    Models:
    - FIRE: Thermal combustion spread + vertical smoke stack propagation.
    - GAS / TOXIC_GAS: Atmospheric chemical dispersion (ammonia/chlorine) with wind advection & diffusion.
    - CHEMICAL_SPILL: Ground-level corrosive liquid spread along containment zones.
    - EXPLOSION / COLLAPSE: Instant blast shockwave, structural damage, and edge severing.
    - WATER: Deluge foam or flash liquid barrier.
    """
    def __init__(self, building_graph: BuildingGraph, multi_floor=None):
        self.building_graph = building_graph
        self.multi_floor = multi_floor
        self.mode = 'synthetic'  # or 'fds'
        self.fds_data: Optional[pd.DataFrame] = None
        self.current_timestep: float = 0.0
        
        # Threat registry: threat_id -> { 'threat_id': str, 'type': str, 'node_id': str, 'intensity': float, 'spread_rate': float, 'metadata': dict }
        self.active_threats: Dict[str, Dict[str, Any]] = {}
        self.node_threat_types: Dict[str, str] = {}
        
        # Environmental conditions (wind & atmosphere)
        self.wind_angle = 45.0            # deg (0 = East, 90 = North)
        self.wind_vector = (0.707, 0.707) # Northeast default
        self.wind_speed = 4.2             # m/s
        self.ambient_temp = 28.0          # deg C
        
    def load_fds_csv(self, csv_path: str):
        self.fds_data = pd.read_csv(csv_path)
        self.mode = 'fds'
        
    def set_wind(self, angle_deg: float, speed_mps: float = 4.2):
        """Update atmospheric wind vector dynamically (angle in degrees, 0 = East, 90 = North)."""
        rad = np.radians(angle_deg)
        self.wind_angle = float(angle_deg)
        self.wind_vector = (float(np.cos(rad)), float(np.sin(rad)))
        self.wind_speed = float(speed_mps)

    def inject_hazard(self, threat_type: str, node_id: str, intensity: float = 0.8, 
                      spread_rate: float = 0.1, metadata: dict = None):
        """Inject an industrial disaster. Supports multiple concurrent threats across the complex."""
        if self.mode == 'fds':
            self.mode = 'synthetic'
            
        t_type = threat_type.upper()
        threat_id = f"{t_type.lower()}_{node_id}"
        
        self.active_threats[threat_id] = {
            'threat_id': threat_id,
            'type': t_type,
            'node_id': node_id,
            'intensity': float(intensity),
            'spread_rate': float(spread_rate),
            'metadata': metadata or {}
        }
        self.node_threat_types[node_id] = t_type
        
        # Special handling for EXPLOSION / COLLAPSE:
        # Instant blast overpressure wave damages node capacity and severs edges
        if t_type in ('EXPLOSION', 'COLLAPSE'):
            node = self.building_graph.building.get_node(node_id)
            if node:
                node.capacity = max(1, node.capacity // 4)
            # Sever immediately adjacent edges within blast shockwave
            for nbr_id in self.building_graph.get_neighbors(node_id):
                edge = self.building_graph.get_edge_between(node_id, nbr_id)
                if edge:
                    edge.state = EdgeState.SEVERED
                    
        self.building_graph.update_hazard(node_id, intensity)

    def inject_scenario(self, scenario_name: str):
        """Inject preset multi-disaster industrial incident scenarios."""
        s = scenario_name.lower()
        if 'catastrophe' in s or 'major' in s:
            # Multi-hazard: Blast at Reactor 1, Flash fire at Tank Farm A, Toxic gas leak at Hazmat basin
            self.set_wind(45.0, 5.5)
            self.inject_hazard('EXPLOSION', 'reactor_1', intensity=0.95, spread_rate=0.0)
            self.inject_hazard('FIRE', 'tank_farm_a', intensity=0.9, spread_rate=0.25)
            self.inject_hazard('GAS', 'hazmat_basin', intensity=0.85, spread_rate=0.3)
        elif 'gas' in s or 'toxic' in s or 'plume' in s:
            # Toxic chemical plume release drifting North-East across catwalks
            self.set_wind(30.0, 6.0)
            self.inject_hazard('GAS', 'hazmat_basin', intensity=0.95, spread_rate=0.35)
            self.inject_hazard('GAS', 'pump_house', intensity=0.85, spread_rate=0.25)
        elif 'bleve' in s or 'tank' in s:
            # Tank Farm BLEVE with blast severing perimeter corridors and flash fire
            self.set_wind(135.0, 4.0)
            self.inject_hazard('EXPLOSION', 'tank_farm_b', intensity=0.95, spread_rate=0.0)
            self.inject_hazard('FIRE', 'compressor_shed', intensity=0.85, spread_rate=0.2)
        elif 'spill' in s or 'acid' in s or 'chemical' in s:
            # Severe corrosive acid spill with emergency washdown deluge
            self.inject_hazard('CHEMICAL_SPILL', 'hazmat_basin', intensity=0.9, spread_rate=0.15)
            self.inject_hazard('WATER', 'emergency_shower', intensity=0.7, spread_rate=0.1)
        elif 'collapse' in s:
            # Pipe rack structural collapse
            self.inject_hazard('COLLAPSE', 'pipe_rack_junc_1', intensity=0.9, spread_rate=0.0)
            self.inject_hazard('GAS', 'reactor_2', intensity=0.75, spread_rate=0.2)

    def inject_fire(self, node_id: str, intensity: float = 0.8, spread_rate: float = 0.1):
        """Backward-compatible wrapper for fire injection."""
        self.inject_hazard('FIRE', node_id, intensity, spread_rate)
        
    def remove_hazard(self, threat_id_or_node_id: str):
        """Remove a hazard from the active registry by threat_id or node_id."""
        if not threat_id_or_node_id:
            return
        target_str = str(threat_id_or_node_id).strip().lower()
        to_del = [
            k for k, v in self.active_threats.items() 
            if k.lower() == target_str or str(v.get('node_id', '')).lower() == target_str
        ]
        cleared_nodes = set()
        for k in to_del:
            node_id = self.active_threats[k]['node_id']
            del self.active_threats[k]
            cleared_nodes.add(node_id)
            
        for node_id in cleared_nodes:
            # If no more threats at this node, clear its hazard
            if not any(str(v.get('node_id', '')).lower() == node_id.lower() for v in self.active_threats.values()):
                self.building_graph.update_hazard(node_id, 0.0)
                if node_id in self.node_threat_types:
                    del self.node_threat_types[node_id]
                # Cool down un-sourced neighbors so residual hazard doesn't immediately re-infect this node
                for nbr_id in self.building_graph.get_neighbors(node_id):
                    if not any(str(v.get('node_id', '')).lower() == nbr_id.lower() for v in self.active_threats.values()):
                        nbr_node = self.building_graph.building.get_node(nbr_id)
                        if nbr_node:
                            nbr_node.hazard_score = min(nbr_node.hazard_score, 0.15)
                # Restore any severed edges attached to this neutralized node
                for nbr_id in self.building_graph.get_neighbors(node_id):
                    edge = self.building_graph.get_edge_between(node_id, nbr_id)
                    if edge and edge.state == EdgeState.SEVERED:
                        edge.state = EdgeState.OPEN
            
    def remove_fire(self, node_id: str):
        """Backward-compatible wrapper for removing threat."""
        self.remove_hazard(node_id)
        
    def step(self, dt: float = 1.0):
        self.current_timestep += dt
        
        if self.mode == 'fds' and self.fds_data is not None:
            closest_time = self.fds_data.iloc[(self.fds_data['timestep'] - self.current_timestep).abs().argsort()[:1]]['timestep'].values[0]
            current_data = self.fds_data[self.fds_data['timestep'] == closest_time]
            
            for _, row in current_data.iterrows():
                temp = row.get('temperature', 20)
                smoke = row.get('smoke_density', 0)
                temp_hazard = max(0, min(1.0, (temp - 50) / 500))
                smoke_hazard = max(0, min(1.0, smoke / 10.0))
                hazard = max(temp_hazard, smoke_hazard)
                if pd.notna(row['node_id']):
                    self.building_graph.update_hazard(row['node_id'], hazard)
                    
        elif self.mode == 'synthetic':
            new_hazards = {}
            for node_id in self.building_graph.adj.keys():
                node = self.building_graph.building.get_node(node_id)
                if not node:
                    continue
                    
                current_hazard = node.hazard_score
                delta = 0.0
                
                # Active threat source maintains its intensity
                node_threats = [t for t in self.active_threats.values() if t['node_id'] == node_id]
                if node_threats:
                    target_intensity = max(t['intensity'] for t in node_threats)
                    delta = max(0.0, target_intensity - current_hazard)
                else:
                    # Natural decay
                    delta -= current_hazard * 0.015 * dt
                    
                    # Spread from neighbors
                    for neighbor_id, edge in self.building_graph.adj.get(node_id, {}).items():
                        neighbor = self.building_graph.building.get_node(neighbor_id)
                        if not neighbor or edge.state == EdgeState.SEVERED:
                            continue
                            
                        if neighbor.hazard_score > current_hazard:
                            nbr_threats = [t for t in self.active_threats.values() if t['node_id'] == neighbor_id]
                            threat_type = nbr_threats[0]['type'] if nbr_threats else self.node_threat_types.get(neighbor_id, 'FIRE')
                            spread_rate = nbr_threats[0]['spread_rate'] if nbr_threats else 0.1
                            
                            # Physics modeling per disaster type:
                            if threat_type in ('GAS', 'TOXIC_GAS'):
                                # Gas disperses rapidly with wind advection
                                dx = node.position[0] - neighbor.position[0]
                                dy = node.position[1] - neighbor.position[1]
                                norm = max(1e-3, float(np.sqrt(dx**2 + dy**2)))
                                cos_wind = (dx * self.wind_vector[0] + dy * self.wind_vector[1]) / (norm * max(1e-3, float(np.sqrt(self.wind_vector[0]**2 + self.wind_vector[1]**2))))
                                wind_factor = max(0.4, 1.0 + 1.2 * cos_wind)
                                spread_rate *= 2.5 * wind_factor
                            elif threat_type == 'FIRE':
                                # Thermal radiation and rising smoke stack effect
                                if edge.is_vertical:
                                    spread_rate *= 3.0
                                if edge.state == EdgeState.LOCKED:
                                    spread_rate *= 0.2
                            elif threat_type == 'CHEMICAL_SPILL':
                                # Liquid spills follow floor surface, cannot traverse upward
                                if edge.is_vertical:
                                    spread_rate = 0.0
                                else:
                                    spread_rate *= 0.7
                            elif threat_type in ('EXPLOSION', 'COLLAPSE'):
                                # Blast shockwave decays rapidly after detonation
                                spread_rate *= 0.1
                                
                            eff_dist = max(5.0, min(25.0, edge.distance))
                            spread_amount = (neighbor.hazard_score - current_hazard) * spread_rate * (18.0 / eff_dist) * dt
                            delta += spread_amount
                            if spread_amount > 0:
                                self.node_threat_types[node_id] = threat_type
                            
                new_hazards[node_id] = max(0.0, min(1.0, current_hazard + delta))
                
            # Cross-floor propagation based on active sources (airborne smoke & gas plumes only)
            hazard_sources = []
            for tid, data in self.active_threats.items():
                t_type = data.get('type', 'FIRE')
                if t_type in ('FIRE', 'GAS', 'TOXIC_GAS'):
                    node = self.building_graph.building.get_node(data['node_id'])
                    if node:
                        hazard_sources.append((data['node_id'], node.floor, data['intensity'], 2))
            
            if hazard_sources:
                building_floors = list(self.building_graph.building.floors.keys())
                config = PropagationConfig()
                floor_hazards = compute_floor_hazard_map(hazard_sources, building_floors, config)
                for node_id in new_hazards:
                    node = self.building_graph.building.get_node(node_id)
                    if node and node.is_cross_floor:
                        floor_h = floor_hazards.get(node.floor, 0.0)
                        new_hazards[node_id] = max(new_hazards[node_id], floor_h)
            
            # Apply updated hazard scores
            for node_id, hazard in new_hazards.items():
                self.building_graph.update_hazard(node_id, hazard)
                
    def get_hazard_at(self, node_id: str) -> float:
        node = self.building_graph.building.get_node(node_id)
        return node.hazard_score if node else 0.0
        
    def get_all_hazards(self) -> Dict[str, float]:
        return {n.id: n.hazard_score for n in self.building_graph.building.all_nodes.values()}
        
    def get_active_threats(self) -> List[Dict]:
        threats = []
        for tid, data in self.active_threats.items():
            node = self.building_graph.building.get_node(data['node_id'])
            threats.append({
                'threat_id': tid,
                'node_id': data['node_id'],
                'type': data.get('type', 'FIRE'),
                'intensity': data.get('intensity', 0.8),
                'spread_rate': data.get('spread_rate', 0.1),
                'floor': node.floor if node else 1,
                'metadata': data.get('metadata', {})
            })
        return threats
        
    def get_fire_sources(self) -> List[Dict]:
        """Backward compatibility for existing callers."""
        return self.get_active_threats()
        
    def clear_all(self):
        """Clear all disasters, reset hazard scores, and restore severed corridors."""
        self.active_threats.clear()
        self.node_threat_types.clear()
        for node in self.building_graph.building.all_nodes.values():
            node.hazard_score = 0.0
        for edge in self.building_graph.building.all_edges:
            if edge.state == EdgeState.SEVERED:
                edge.state = EdgeState.OPEN


# Backward compatibility alias
FireModel = IndustrialHazardModel

