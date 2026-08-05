import numpy as np
import gymnasium
from gymnasium.spaces import Box, MultiDiscrete, Dict as GymDict, MultiBinary
from pettingzoo import ParallelEnv
from core.graph.types import Building, NodeType, EdgeState
from core.graph.building_graph import BuildingGraph
from sim.services.environment.src.fire_model import FireModel

class EvacuationEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "lbp_evac_v0"}

    def __init__(self, building: Building, max_steps: int = 500, num_evacuees: int = 200):
        super().__init__()
        self.building = building
        self.building_graph = BuildingGraph(building)
        self.fire_model = FireModel(self.building_graph)
        self.max_steps = max_steps
        self.total_evacuees = num_evacuees
        self.current_step = 0
        
        # Router Nodes only (INTERSECTION, STAIRWELL)
        self.possible_agents = [
            n_id for n_id, n in self.building.all_nodes.items() 
            if n.type in (NodeType.INTERSECTION, NodeType.STAIRWELL)
        ]
        self.agents = self.possible_agents.copy()
        
        # Determine max neighbors across all agents for padding
        self.max_neighbors = max((len(self.building_graph.get_neighbors(ag)) for ag in self.possible_agents), default=1)
        
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for agent in self.possible_agents:
            num_neighbors = len(self.building_graph.get_neighbors(agent))
            has_door = any(edge.has_door for edge in self.building_graph.adj.get(agent, {}).values())
            
            # Action space: [ALLOW=0, REDIRECT=1, BLOCK=2] for each edge + door action
            # To handle variable neighbors, we just use max_neighbors and ignore extra actions
            # Door action: 0=NO_CHANGE, 1=LOCK
            actions = [3] * self.max_neighbors + ([2] if has_door else [1])
            self.action_spaces[agent] = MultiDiscrete(actions)
            
            self.observation_spaces[agent] = GymDict({
                'hazard': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'crowd': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'neighbor_hazards': Box(low=0.0, high=1.0, shape=(self.max_neighbors,), dtype=np.float32),
                'neighbor_crowds': Box(low=0.0, high=1.0, shape=(self.max_neighbors,), dtype=np.float32),
                'edge_states': MultiBinary(self.max_neighbors),
                'floor': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                'is_stairwell': MultiBinary(1),
                'time_remaining': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })
            
        self.evacuees_at_node = {n_id: 0 for n_id in self.building.all_nodes.keys()}
        self.casualties = 0
        self.evacuated = 0

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.casualties = 0
        self.evacuated = 0
        self.fire_model.clear_all()
        
        # Reset edge states
        for edge in self.building.all_edges:
            edge.state = EdgeState.OPEN
            
        # Distribute evacuees randomly in ROOMs and CORRIDORs
        self.evacuees_at_node = {n_id: 0 for n_id in self.building.all_nodes.keys()}
        eligible_nodes = [n_id for n_id, n in self.building.all_nodes.items() if n.type in (NodeType.ROOM, NodeType.CORRIDOR)]
        
        if eligible_nodes:
            # Simple uniform distribution
            counts = np.random.multinomial(self.total_evacuees, [1/len(eligible_nodes)]*len(eligible_nodes))
            for i, n_id in enumerate(eligible_nodes):
                self.evacuees_at_node[n_id] = counts[i]
                
        # Inject random fire
        if options and options.get('fire_node'):
            fire_node = options['fire_node']
        else:
            fire_node = np.random.choice(eligible_nodes) if eligible_nodes else None
        
        if fire_node:
            self.fire_model.inject_fire(fire_node, intensity=0.8, spread_rate=0.1)

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            node = self.building.get_node(agent)
            neighbors = self.building_graph.get_neighbors(agent)
            
            n_hazards = np.zeros(self.max_neighbors, dtype=np.float32)
            n_crowds = np.zeros(self.max_neighbors, dtype=np.float32)
            e_states = np.zeros(self.max_neighbors, dtype=np.int8)
            
            for i, n_id in enumerate(neighbors):
                if i >= self.max_neighbors:
                    break
                n_node = self.building.get_node(n_id)
                n_hazards[i] = n_node.hazard_score
                n_cap = max(1, n_node.capacity)
                n_crowds[i] = min(1.0, self.evacuees_at_node.get(n_id, 0) / n_cap)
                
                edge = self.building_graph.get_edge_between(agent, n_id)
                e_states[i] = 1 if edge and edge.state == EdgeState.OPEN else 0
                
            floor_norm = np.array([node.floor / max(1, len(self.building.floors))], dtype=np.float32)
            is_stair = np.array([1 if node.type == NodeType.STAIRWELL else 0], dtype=np.int8)
            
            obs[agent] = {
                'hazard': np.array([node.hazard_score], dtype=np.float32),
                'crowd': np.array([min(1.0, self.evacuees_at_node.get(agent, 0) / max(1, node.capacity))], dtype=np.float32),
                'neighbor_hazards': n_hazards,
                'neighbor_crowds': n_crowds,
                'edge_states': e_states,
                'floor': floor_norm,
                'is_stairwell': is_stair,
                'time_remaining': np.array([1.0 - (self.current_step / self.max_steps)], dtype=np.float32)
            }
        return obs

    def step(self, actions):
        self.current_step += 1
        
        # 1. Apply actions
        for agent, action in actions.items():
            neighbors = self.building_graph.get_neighbors(agent)
            edge_actions = action[:-1]
            door_action = action[-1]
            
            for i, n_id in enumerate(neighbors):
                if i >= self.max_neighbors:
                    break
                edge = self.building_graph.get_edge_between(agent, n_id)
                if not edge:
                    continue
                    
                e_act = edge_actions[i]
                if e_act == 2: # BLOCK
                    # Simplify: mark as closed (in real MARL might use emergency seal)
                    # We just use locked for now
                    edge.state = EdgeState.LOCKED
                elif e_act == 0: # ALLOW
                    if edge.state == EdgeState.LOCKED:
                        edge.state = EdgeState.OPEN
                        
            # Handle doors
            if door_action == 1:
                # Lock all doors at this node
                for n_id in neighbors:
                    edge = self.building_graph.get_edge_between(agent, n_id)
                    if edge and edge.has_door:
                        edge.state = EdgeState.LOCKED

        # 2. Step fire
        self.fire_model.step(dt=1.0)
        
        # 3. Move evacuees
        new_evacuees = {n_id: 0 for n_id in self.building.all_nodes.keys()}
        evacuees_moved_through = {agent: 0 for agent in self.agents}
        
        for n_id, count in self.evacuees_at_node.items():
            if count == 0:
                continue
                
            node = self.building.get_node(n_id)
            if node.type == NodeType.EXIT:
                # They are evacuated
                self.evacuated += count
                continue
                
            if node.hazard_score > 0.95:
                # Become casualties
                self.casualties += count
                continue
                
            # Find best neighbor to move to based on shortest path to exit
            # We cache paths or compute on fly. For simplicity in Env, we compute basic routing
            paths = self.building_graph.get_all_exit_paths(n_id)
            if not paths:
                # Stuck
                new_evacuees[n_id] += count
                continue
                
            best_path, _ = paths[0]
            if len(best_path) > 1:
                next_node = best_path[1]
                edge = self.building_graph.get_edge_between(n_id, next_node)
                
                if edge and edge.state == EdgeState.OPEN:
                    # Move bounded by throughput and capacity
                    tgt_node = self.building.get_node(next_node)
                    tgt_capacity = tgt_node.capacity - new_evacuees.get(next_node, 0)
                    
                    can_move = min(count, edge.max_throughput, tgt_capacity)
                    can_move = max(0, can_move)
                    
                    new_evacuees[next_node] += can_move
                    new_evacuees[n_id] += (count - can_move)
                    
                    if next_node in evacuees_moved_through:
                        evacuees_moved_through[next_node] += can_move
                else:
                    # Blocked, stay
                    new_evacuees[n_id] += count
            else:
                new_evacuees[n_id] += count
                
        self.evacuees_at_node = new_evacuees
        
        # 4 & 5. Casualties & Evacuated are handled in the movement loop
        
        # 6. Rewards
        rewards = {}
        for agent in self.agents:
            node = self.building.get_node(agent)
            
            moved = evacuees_moved_through[agent]
            casualties_here = self.evacuees_at_node[agent] if node.hazard_score > 0.95 else 0
            
            cap = max(1, node.capacity)
            congestion = min(1.0, self.evacuees_at_node[agent] / cap)
            
            alpha = 1.0
            beta = 10.0
            gamma = 0.5
            delta = 0.1
            
            rewards[agent] = (alpha * moved) - (beta * casualties_here) - (gamma * congestion) - delta
            
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        total_remaining = sum(self.evacuees_at_node.values())
        done = (total_remaining == 0) or (self.current_step >= self.max_steps)
        
        if done:
            for agent in self.agents:
                if self.current_step >= self.max_steps:
                    truncations[agent] = True
                else:
                    terminations[agent] = True
            self.agents = []
            
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
