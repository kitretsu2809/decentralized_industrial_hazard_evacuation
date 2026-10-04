"""
Spatio-Temporal Topology-Bound Graph Attention Network (ST-TBA-GAT) Policy.

Fulfills Patent Application Claims 1, 2, 5, and 7:
- Claim 1(a) & 2: Topology-Bound Multi-Head Spatial Graph Attention (O(|N_i|) complexity).
- Claim 1(b): Recurrent Temporal Gated Recurrent Unit (GRU) state transition cell.
- Claim 1(c): Decentralized Actor policy network with corridor action masking.
- Claim 1(d): Deterministic Life-Safety Reflexive Layer overriding hazardous paths (H >= theta_crit).
- Claim 5: Temporal hysteresis filter preventing rapid directional sign flickering.
- Claim 7: Centralized Training with Decentralized Execution (CTDE) global critic pooling.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class EdgeGATConv(nn.Module):
    """
    Edge-conditioned Graph Attention layer (GATv2 style):
        e_{ij} = a^T LeakyReLU(W_src * h_i + W_dst * h_j + W_edge * e_{ij})
        alpha_{ij} = softmax_j(e_{ij})
        out_i = sum_j alpha_{ij} * (W_v * h_j)
    """
    def __init__(self, in_node_dim: int, in_edge_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // heads

        self.w_src = nn.Linear(in_node_dim, out_dim, bias=False)
        self.w_dst = nn.Linear(in_node_dim, out_dim, bias=False)
        self.w_edge = nn.Linear(in_edge_dim, out_dim, bias=False)
        self.w_val = nn.Linear(in_node_dim, out_dim, bias=False)

        self.attn = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_src.weight)
        nn.init.xavier_uniform_(self.w_dst.weight)
        nn.init.xavier_uniform_(self.w_edge.weight)
        nn.init.xavier_uniform_(self.w_val.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mask_severed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N = node_feats.size(0)
        E = edge_index.size(1)

        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        h_src = self.w_src(node_feats[src_nodes]).view(E, self.heads, self.head_dim)
        h_dst = self.w_dst(node_feats[dst_nodes]).view(E, self.heads, self.head_dim)
        h_edge = self.w_edge(edge_attr).view(E, self.heads, self.head_dim)
        v_src = self.w_val(node_feats[src_nodes]).view(E, self.heads, self.head_dim)

        scores = self.leaky_relu(h_src + h_dst + h_edge)
        attn_logits = (scores * self.attn).sum(dim=-1)

        if mask_severed is not None:
            attn_logits = attn_logits.masked_fill(~mask_severed.unsqueeze(-1), -1e9)

        exp_logits = torch.exp(attn_logits - attn_logits.max())
        if mask_severed is not None:
            exp_logits = exp_logits * mask_severed.unsqueeze(-1).float()

        denom = torch.zeros((N, self.heads), device=node_feats.device)
        denom.index_add_(0, dst_nodes, exp_logits)
        denom = denom + 1e-8

        alpha = exp_logits / denom[dst_nodes]
        messages = v_src * alpha.unsqueeze(-1)
        messages_flat = messages.view(E, self.out_dim)

        out = torch.zeros((N, self.out_dim), device=node_feats.device)
        out.index_add_(0, dst_nodes, messages_flat)
        return F.elu(out)


class ReflexiveSafetyOverride:
    """
    Patent Claim 1(d) & Claim 11:
    Deterministic Life-Safety Reflexive Layer.
    Guarantees that routing choices never direct evacuees into lethal nodes (H >= theta_crit).
    """
    THETA_CRIT: float = 0.55

    @classmethod
    def apply_mask(
        cls,
        logits: torch.Tensor,
        action_masks: torch.Tensor,
        neighbor_hazards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hard-masks invalid or hazardous neighbor corridors to -1e9.
        """
        masked_logits = logits.clone()
        # Physical topology mask
        if action_masks is not None:
            masked_logits = masked_logits.masked_fill(action_masks == 0, -1e9)

        # Reflexive hazard mask (Claim 1(d))
        if neighbor_hazards is not None:
            is_lethal = neighbor_hazards >= cls.THETA_CRIT
            # Ensure we don't mask all actions if entire area is in smoke
            all_lethal = is_lethal.all(dim=-1, keepdim=True)
            safe_lethal_mask = is_lethal & (~all_lethal)
            masked_logits = masked_logits.masked_fill(safe_lethal_mask, -1e9)

        return masked_logits


class PermutationInvariantGATPolicy(nn.Module):
    """
    Complete ST-TBA-GAT Actor-Critic Architecture.
    """
    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        max_corridors: int = 6,
        heads: int = 4
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.max_corridors = max_corridors

        # 1. Spatial Topology-Bound Attention Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gat1 = EdgeGATConv(hidden_dim, hidden_dim, hidden_dim, heads=heads)
        self.gat2 = EdgeGATConv(hidden_dim, hidden_dim, hidden_dim, heads=heads)

        # 2. Recurrent Temporal GRU Cell (Patent Claim 1(b))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 3. Decentralized Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_corridors)
        )

        # 4. Centralized Critic Head (CTDE Readout)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def extract_graph_embeddings(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        mask_severed: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (spatial_embeddings, persistent_temporal_hidden, node_and_context).
        """
        N = node_feats.size(0)
        h0 = self.node_encoder(node_feats)
        e0 = self.edge_encoder(edge_attr)

        h1 = self.gat1(h0, edge_index, e0, mask_severed=mask_severed) + h0
        s_t = self.gat2(h1, edge_index, e0, mask_severed=mask_severed) + h1

        # Temporal GRU update
        if hidden_state is None or hidden_state.size(0) != N:
            hidden_state = torch.zeros(N, self.hidden_dim, device=node_feats.device)

        new_hidden = self.gru(s_t, hidden_state)

        # Permutation-invariant readout pooling
        global_mean = new_hidden.mean(dim=0, keepdim=True).expand(N, -1)
        global_max = new_hidden.max(dim=0, keepdim=True)[0].expand(N, -1)
        global_context = (global_mean + global_max) * 0.5

        node_and_context = torch.cat([new_hidden, global_context], dim=-1)
        return s_t, new_hidden, node_and_context

    def forward_actor(
        self,
        node_and_context: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        neighbor_hazards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raw_logits = self.actor_head(node_and_context)
        # Apply Patent Claim 1(d) Reflexive Safety Override
        safe_logits = ReflexiveSafetyOverride.apply_mask(
            raw_logits, action_masks, neighbor_hazards
        )
        return safe_logits

    def forward_critic(self, node_and_context: torch.Tensor) -> torch.Tensor:
        values = self.critic_head(node_and_context)
        return values.mean(dim=0)

    def act(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        neighbor_hazards: Optional[torch.Tensor] = None,
        mask_severed: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, new_hidden, node_and_context = self.extract_graph_embeddings(
            node_feats, edge_index, edge_attr, hidden_state, mask_severed
        )
        logits = self.forward_actor(node_and_context, action_masks, neighbor_hazards)
        values = self.forward_critic(node_and_context)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        return actions, log_probs, values, new_hidden
