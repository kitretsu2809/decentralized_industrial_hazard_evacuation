import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from .gat_encoder import create_encoder

class Actor(nn.Module):
    def __init__(self, embedding_dim: int = 64, max_actions: int = 6, hidden_dims: List[int] = [128, 64]):
        super(Actor, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_actions = max_actions
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Action head per edge: [ALLOW, REDIRECT, BLOCK]
        self.edge_head = nn.Linear(hidden_dims[-1], max_actions * 3)
        
        # Door head: [NO_CHANGE, LOCK] (could be scaled to max_actions * 2 if per door)
        # Taking simplified interpretation: 1 door per node or aggregated
        # Assuming simplified representation per edge for now or a general node door state
        # The spec says: Door head: Linear(hidden_dims[-1], 2)
        self.door_head = nn.Linear(hidden_dims[-1], 2)
        
        # Vertical head: [ALLOW_BOTH, UP, DOWN, BLOCK_VERTICAL]
        self.vertical_head = nn.Linear(hidden_dims[-1], 4)
        
    def forward(self, embedding: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.mlp(embedding)
        
        # Edge actions
        edge_logits_flat = self.edge_head(features) # (N, max_actions * 3)
        edge_logits = edge_logits_flat.view(-1, self.max_actions, 3) # (N, max_actions, 3)
        
        # Apply action mask if provided (set invalid actions to -inf)
        # assuming action_mask is (N, max_actions) boolean or float
        if action_mask is not None:
            mask = action_mask.unsqueeze(-1).expand_as(edge_logits)
            # Mask format depends on implementation, assuming boolean:
            edge_logits = edge_logits.masked_fill(~mask, float('-inf'))
            
        door_logits = self.door_head(features) # (N, 2)
        vertical_logits = self.vertical_head(features) # (N, 4)
        
        return edge_logits, door_logits, vertical_logits

class Critic(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_agents: int = 20, hidden_dims: List[int] = [256, 128]):
        super(Critic, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * num_agents, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
    def forward(self, all_embeddings: torch.Tensor) -> torch.Tensor:
        # all_embeddings: (Batch, num_agents, embedding_dim) or (num_agents, embedding_dim)
        if all_embeddings.dim() == 2:
            flat_embeddings = all_embeddings.view(-1) # (num_agents * embedding_dim)
        else:
            flat_embeddings = all_embeddings.view(all_embeddings.size(0), -1) # (B, num_agents * embedding_dim)
            
        value = self.mlp(flat_embeddings)
        return value

class ActorCritic(nn.Module):
    def __init__(self, node_feature_dim: int = 8, embedding_dim: int = 64, max_actions: int = 6, num_agents: int = 20, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1, use_recurrent: bool = True, actor_hidden_dims: List[int] = [128, 64], critic_hidden_dims: List[int] = [256, 128]):
        super(ActorCritic, self).__init__()
        self.use_recurrent = use_recurrent
        
        self.encoder = create_encoder(
            node_feature_dim=node_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_recurrent=use_recurrent
        )
        
        self.actor = Actor(
            embedding_dim=embedding_dim,
            max_actions=max_actions,
            hidden_dims=actor_hidden_dims
        )
        
        self.critic = Critic(
            embedding_dim=embedding_dim,
            num_agents=num_agents,
            hidden_dims=critic_hidden_dims
        )
        
    def forward_encoder(self, node_features: torch.Tensor, edge_index: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_recurrent:
            embeddings, next_h = self.encoder(node_features, edge_index, hidden_state)
            return embeddings, next_h
        else:
            embeddings = self.encoder(node_features, edge_index)
            return embeddings, None

    def get_action(self, node_features: torch.Tensor, edge_index: torch.Tensor, agent_idx: int, action_mask: torch.Tensor, hidden_state: Optional[torch.Tensor] = None, deterministic: bool = False):
        embeddings, next_h = self.forward_encoder(node_features, edge_index, hidden_state)
        
        # Get specific agent's embedding
        agent_embedding = embeddings[agent_idx].unsqueeze(0) if embeddings.dim() == 2 else embeddings[:, agent_idx]
        
        # Expand mask if necessary
        mask = action_mask.unsqueeze(0) if action_mask.dim() == 1 else action_mask
        
        edge_logits, door_logits, vertical_logits = self.actor(agent_embedding, mask)
        
        if deterministic:
            edge_action = torch.argmax(edge_logits, dim=-1)
            door_action = torch.argmax(door_logits, dim=-1)
            vertical_action = torch.argmax(vertical_logits, dim=-1)
            log_prob = None # Not needed for deterministic
        else:
            # Distribution logic
            edge_dist = torch.distributions.Categorical(logits=edge_logits)
            door_dist = torch.distributions.Categorical(logits=door_logits)
            vertical_dist = torch.distributions.Categorical(logits=vertical_logits)
            
            edge_action = edge_dist.sample()
            door_action = door_dist.sample()
            vertical_action = vertical_dist.sample()
            
            log_prob = edge_dist.log_prob(edge_action).sum(dim=-1) + door_dist.log_prob(door_action) + vertical_dist.log_prob(vertical_action)
            
        value = self.critic(embeddings.unsqueeze(0) if embeddings.dim() == 2 else embeddings)
        
        return (edge_action, door_action, vertical_action), log_prob, value, next_h
        
    def evaluate_action(self, node_features: torch.Tensor, edge_index: torch.Tensor, agent_idx: int, action: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], action_mask: torch.Tensor, hidden_state: Optional[torch.Tensor] = None):
        embeddings, next_h = self.forward_encoder(node_features, edge_index, hidden_state)
        
        agent_embedding = embeddings[agent_idx].unsqueeze(0) if embeddings.dim() == 2 else embeddings[:, agent_idx]
        mask = action_mask.unsqueeze(0) if action_mask.dim() == 1 else action_mask
        
        edge_logits, door_logits, vertical_logits = self.actor(agent_embedding, mask)
        
        edge_dist = torch.distributions.Categorical(logits=edge_logits)
        door_dist = torch.distributions.Categorical(logits=door_logits)
        vertical_dist = torch.distributions.Categorical(logits=vertical_logits)
        
        edge_action, door_action, vertical_action = action
        
        log_prob = edge_dist.log_prob(edge_action).sum(dim=-1) + door_dist.log_prob(door_action) + vertical_dist.log_prob(vertical_action)
        entropy = edge_dist.entropy().sum(dim=-1) + door_dist.entropy() + vertical_dist.entropy()
        
        value = self.critic(embeddings.unsqueeze(0) if embeddings.dim() == 2 else embeddings)
        
        return log_prob, entropy, value, next_h
        
    def get_value(self, node_features: torch.Tensor, edge_index: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings, _ = self.forward_encoder(node_features, edge_index, hidden_state)
        return self.critic(embeddings.unsqueeze(0) if embeddings.dim() == 2 else embeddings)

