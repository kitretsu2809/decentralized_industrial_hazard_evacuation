import math
from typing import List, Tuple, Dict, Optional
from core.graph.types import NodeType, EdgeState, Edge
from core.graph.building_graph import BuildingGraph

class MultiFloorManager:
    def __init__(self, graph: BuildingGraph):
        self.graph = graph

    def get_vertical_connections(self, floor: int) -> List[Edge]:
        connections = []
        for edge in self.graph.building.cross_floor_edges:
            if edge.floor_span and (edge.floor_span[0] <= floor <= edge.floor_span[1]):
                src_node = self.graph.building.get_node(edge.source)
                dst_node = self.graph.building.get_node(edge.target)
                if src_node and dst_node and (src_node.floor == floor or dst_node.floor == floor):
                    connections.append(edge)
            else:
                # Fallback check via nodes
                src_node = self.graph.building.get_node(edge.source)
                dst_node = self.graph.building.get_node(edge.target)
                if src_node and dst_node and (src_node.floor == floor or dst_node.floor == floor):
                    if edge not in connections:
                        connections.append(edge)
        return connections

    def get_stairwell_nodes(self, floor: int) -> List[str]:
        stairwells = []
        nodes = self.graph.get_floor_subgraph_nodes(floor)
        for node_id in nodes:
            node = self.graph.building.get_node(node_id)
            if node and node.type == NodeType.STAIRWELL:
                stairwells.append(node_id)
        return stairwells

    def get_elevator_nodes(self, floor: int) -> List[str]:
        elevators = []
        nodes = self.graph.get_floor_subgraph_nodes(floor)
        for node_id in nodes:
            node = self.graph.building.get_node(node_id)
            if node and node.type == NodeType.ELEVATOR:
                elevators.append(node_id)
        return elevators

    def disable_elevators(self, floor: int = None):
        edges_to_disable = []
        if floor is not None:
            edges_to_disable = self.get_vertical_connections(floor)
        else:
            edges_to_disable = self.graph.building.cross_floor_edges
            
        for edge in edges_to_disable:
            src_node = self.graph.building.get_node(edge.source)
            dst_node = self.graph.building.get_node(edge.target)
            if src_node and dst_node and (src_node.type == NodeType.ELEVATOR or dst_node.type == NodeType.ELEVATOR):
                self.graph.sever_edge(edge.id)

    def enable_elevators(self, floor: int = None):
        edges_to_enable = []
        if floor is not None:
            edges_to_enable = self.get_vertical_connections(floor)
        else:
            edges_to_enable = self.graph.building.cross_floor_edges
            
        for edge in edges_to_enable:
            src_node = self.graph.building.get_node(edge.source)
            dst_node = self.graph.building.get_node(edge.target)
            if src_node and dst_node and (src_node.type == NodeType.ELEVATOR or dst_node.type == NodeType.ELEVATOR):
                self.graph.restore_edge(edge.id)

    def get_stairwell_congestion(self) -> Dict[str, float]:
        congestion = {}
        for node_id, node in self.graph.building.all_nodes.items():
            if node.type == NodeType.STAIRWELL:
                if node.capacity > 0:
                    congestion[node_id] = node.current_occupancy / float(node.capacity)
                else:
                    congestion[node_id] = float('inf') if node.current_occupancy > 0 else 0.0
        return congestion

    def compute_optimal_exit_floor(self, node_id: str) -> Tuple[int, List[str], float]:
        paths = self.graph.get_all_exit_paths(node_id)
        if not paths:
            return -1, [], float('inf')
        
        best_path, best_cost = paths[0]
        exit_node = self.graph.building.get_node(best_path[-1])
        exit_floor = exit_node.floor if exit_node else -1
        
        return exit_floor, best_path, best_cost

    def get_cross_floor_neighbors(self, node_id: str) -> List[str]:
        neighbors = self.graph.get_neighbors(node_id)
        cross_neighbors = []
        node = self.graph.building.get_node(node_id)
        if not node:
            return []
            
        for neighbor in neighbors:
            neighbor_node = self.graph.building.get_node(neighbor)
            if neighbor_node and neighbor_node.floor != node.floor:
                cross_neighbors.append(neighbor)
        return cross_neighbors

    def isolate_floor(self, floor: int):
        for edge in self.get_vertical_connections(floor):
            self.graph.sever_edge(edge.id)

    def restore_floor(self, floor: int):
        for edge in self.get_vertical_connections(floor):
            self.graph.restore_edge(edge.id)

    def get_vertical_flow_direction(self, stairwell_id: str) -> str:
        stair_node = self.graph.building.get_node(stairwell_id)
        if not stair_node:
            return "blocked"
            
        active_edges = self.graph.get_active_edges(stairwell_id)
        if not active_edges:
            return "blocked"
            
        flow_up = False
        flow_down = False
        
        for edge in active_edges:
            neighbor_id = edge.target if edge.source == stairwell_id else edge.source
            neighbor_node = self.graph.building.get_node(neighbor_id)
            if neighbor_node and neighbor_node.floor > stair_node.floor:
                flow_up = True
            elif neighbor_node and neighbor_node.floor < stair_node.floor:
                flow_down = True
                
        if flow_up and flow_down:
            return "both"
        elif flow_up:
            return "up"
        elif flow_down:
            return "down"
        return "blocked"
