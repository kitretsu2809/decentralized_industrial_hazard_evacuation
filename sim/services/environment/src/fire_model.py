import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.graph.building_graph import BuildingGraph
from core.graph.types import EdgeState
from core.hazard.cross_floor_propagation import compute_floor_hazard_map, PropagationConfig

class FireModel:
    def __init__(self, building_graph: BuildingGraph, multi_floor=None):
        self.building_graph = building_graph
        self.multi_floor = multi_floor
        self.mode = 'synthetic'  # or 'fds'
        self.fds_data: Optional[pd.DataFrame] = None
        self.current_timestep: float = 0.0
        
        # synthetic fires: node_id -> { 'intensity': float, 'spread_rate': float }
        self.active_fires: Dict[str, Dict] = {}
        
    def load_fds_csv(self, csv_path: str):
        self.fds_data = pd.read_csv(csv_path)
        self.mode = 'fds'
        
    def inject_fire(self, node_id: str, intensity: float = 0.8, spread_rate: float = 0.1):
        if self.mode == 'fds':
            self.mode = 'synthetic'
        self.active_fires[node_id] = {
            'intensity': intensity,
            'spread_rate': spread_rate
        }
        self.building_graph.update_hazard(node_id, intensity)
        
    def step(self, dt: float = 1.0):
        self.current_timestep += dt
        
        if self.mode == 'fds' and self.fds_data is not None:
            # simple closest timestep match
            closest_time = self.fds_data.iloc[(self.fds_data['timestep'] - self.current_timestep).abs().argsort()[:1]]['timestep'].values[0]
            current_data = self.fds_data[self.fds_data['timestep'] == closest_time]
            
            for _, row in current_data.iterrows():
                # Normalized hazard score based on temperature/smoke
                # Assuming simple mapping for this implementation
                temp = row.get('temperature', 20)
                smoke = row.get('smoke_density', 0)
                
                # very crude normalization for example purposes
                temp_hazard = max(0, min(1.0, (temp - 50) / 500))
                smoke_hazard = max(0, min(1.0, smoke / 10.0))
                hazard = max(temp_hazard, smoke_hazard)
                
                if pd.notna(row['node_id']):
                    self.building_graph.update_hazard(row['node_id'], hazard)
                    
        elif self.mode == 'synthetic':
            # Spread logic
            new_hazards = {}
            for node_id in self.building_graph.adj.keys():
                node = self.building_graph.building.get_node(node_id)
                if not node:
                    continue
                    
                current_hazard = node.hazard_score
                delta = 0.0
                
                # Active fire source maintains its intensity
                if node_id in self.active_fires:
                    delta = max(0, self.active_fires[node_id]['intensity'] - current_hazard)
                else:
                    # Decay
                    delta -= current_hazard * 0.02 * dt
                    
                    # Spread from neighbors
                    for neighbor_id, edge in self.building_graph.adj.get(node_id, {}).items():
                        neighbor = self.building_graph.building.get_node(neighbor_id)
                        if not neighbor or edge.state == EdgeState.SEVERED:
                            continue
                            
                        if neighbor.hazard_score > current_hazard:
                            spread_rate = 0.1 # default
                            if neighbor_id in self.active_fires:
                                spread_rate = self.active_fires[neighbor_id]['spread_rate']
                                
                            # Vertical is 3x faster
                            if edge.is_vertical:
                                spread_rate *= 3.0
                                
                            # Fire doors reduce by 80%
                            if edge.state == EdgeState.LOCKED: # simplistic rep of closed fire door
                                spread_rate *= 0.2
                                
                            spread_amount = (neighbor.hazard_score - current_hazard) * spread_rate * dt / max(1.0, edge.distance)
                            delta += spread_amount
                            
                new_hazards[node_id] = max(0.0, min(1.0, current_hazard + delta))
                
            # Cross-floor propagation based on sources
            hazard_sources = []
            for n_id, data in self.active_fires.items():
                node = self.building_graph.building.get_node(n_id)
                if node:
                    hazard_sources.append((n_id, node.floor, data['intensity'], 2)) # threat type 2 = smoke
            
            if hazard_sources:
                building_floors = list(self.building_graph.building.floors.keys())
                config = PropagationConfig()
                floor_hazards = compute_floor_hazard_map(hazard_sources, building_floors, config)
                
                # Apply cross-floor hazards to nodes on those floors
                for node_id in new_hazards:
                    node = self.building_graph.building.get_node(node_id)
                    if node and node.is_cross_floor:
                        floor_h = floor_hazards.get(node.floor, 0.0)
                        # We apply some factor of the floor hazard
                        new_hazards[node_id] = max(new_hazards[node_id], floor_h)
            
            # Update all
            for node_id, hazard in new_hazards.items():
                self.building_graph.update_hazard(node_id, hazard)
                
    def get_hazard_at(self, node_id: str) -> float:
        node = self.building_graph.building.get_node(node_id)
        return node.hazard_score if node else 0.0
        
    def get_all_hazards(self) -> Dict[str, float]:
        return {n.id: n.hazard_score for n in self.building_graph.building.all_nodes.values()}
        
    def get_fire_sources(self) -> List[Dict]:
        sources = []
        for n_id, data in self.active_fires.items():
            sources.append({
                'node_id': n_id,
                'intensity': data['intensity'],
                'spread_rate': data['spread_rate']
            })
        return sources
        
    def clear_all(self):
        self.active_fires.clear()
        for node in self.building_graph.building.all_nodes.values():
            node.hazard_score = 0.0
            
    def remove_fire(self, node_id: str):
        if node_id in self.active_fires:
            del self.active_fires[node_id]
