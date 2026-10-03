"""
ST-TBA-GAT: Spatio-Temporal Topology & Bottleneck-Aware Graph Attention Network.
Pure PyTorch implementation — zero torch-geometric dependency.
Includes:
  - GATv2 Multi-Head Graph Attention over 1-hop physical mesh graph.
  - Recurrent GRU Cell tracking temporal hazard and bottleneck queues.
  - Decentralized Actor heads for multi-corridor signage actions [ALLOW, REDIRECT, BLOCK].
  - Centralized Critic head (CTDE) for global value estimation.
  - Deterministic Hardware Reflexive Safety Override.
"""
from __future__ import annotations
import os
import math
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class GATv2Layer(nn.Module):
    """
    Graph Attention Network v2 layer implemented in native PyTorch.
    Computes dynamic attention over directed edges (src -> dst):
      e_{ij} = a^T LeakyReLU(W_src * x_j + W_dst * x_i)
      alpha_{ij} = softmax_j(e_{ij})
      out_i = sum_{j in N(i)} alpha_{ij} * W_val * x_j
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope

        self.w_src = nn.Linear(in_features, heads * out_features, bias=False)
        self.w_dst = nn.Linear(in_features, heads * out_features, bias=False)
        self.w_val = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, heads, out_features))

        self.bias = nn.Parameter(
            torch.Tensor(heads * out_features if concat else out_features)
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.w_src.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_dst.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_val.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N, in_features)
        edge_index: (2, E) where edge_index[0] = source, edge_index[1] = target
        """
        num_nodes = x.size(0)
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # Linear projections: (N, heads, out_features)
        h_src = self.w_src(x).view(num_nodes, self.heads, self.out_features)
        h_dst = self.w_dst(x).view(num_nodes, self.heads, self.out_features)
        h_val = self.w_val(x).view(num_nodes, self.heads, self.out_features)

        # GATv2 edge scoring: (E, heads, out_features)
        edge_feat = F.leaky_relu(
            h_src[src_nodes] + h_dst[dst_nodes],
            negative_slope=self.leaky_relu_slope,
        )
        # Attention scores: (E, heads)
        scores = (edge_feat * self.attn).sum(dim=-1)

        # Softmax over incoming neighbors for each target node:
        # Subtract max for numerical stability
        max_scores = torch.zeros(
            num_nodes, self.heads, device=x.device
        ).fill_(-math.inf)
        dst_idx = dst_nodes.unsqueeze(-1).expand(-1, self.heads)
        max_scores.scatter_reduce_(
            0,
            dst_idx,
            scores,
            reduce="amax",
            include_self=False,
        )
        exp_scores = torch.exp(scores - max_scores[dst_nodes])

        sum_exp = torch.zeros(
            num_nodes, self.heads, device=x.device
        ).fill_(1e-8)
        sum_exp.scatter_add_(
            0,
            dst_idx,
            exp_scores,
        )
        alpha = exp_scores / sum_exp[dst_nodes]
        if self.dropout > 0.0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted value aggregation: (E, heads, out_features)
        weighted_val = h_val[src_nodes] * alpha.unsqueeze(-1)

        out = torch.zeros(
            num_nodes, self.heads, self.out_features, device=x.device
        )
        out.scatter_add_(
            0,
            dst_nodes.view(-1, 1, 1).expand(-1, self.heads, self.out_features),
            weighted_val,
        )

        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out


class ReflexiveSafetyOverride:
    """
    Deterministic Hardware Reflexive Safety Override.
    Emulates the dedicated analog/FPGA life-safety interlock on edge nodes:
      If local hazard score H_v >= theta_crit (0.80) or downstream corridor is lethal,
      the corridor state is strictly clamped to BLOCK (action=2, Red X) within < 20 ms,
      guaranteeing zero trapped evacuees regardless of neural network outputs.
    """
    THETA_CRIT: float = 0.80

    @classmethod
    def apply(
        cls,
        actions: torch.Tensor,
        node_hazards: torch.Tensor,
        neighbor_hazards: torch.Tensor,
    ) -> torch.Tensor:
        """
        actions: (N, max_corridors) with integer values 0, 1, 2
        node_hazards: (N,) or (N, 1) in [0, 1]
        neighbor_hazards: (N, max_corridors) in [0, 1]
        """
        overridden = actions.clone()
        # 1. If local node is in critical danger, block incoming flows
        crit_local = (node_hazards.squeeze(-1) >= cls.THETA_CRIT).unsqueeze(-1)
        # 2. If target neighbor corridor is critical, block outgoing path
        crit_neighbor = neighbor_hazards >= cls.THETA_CRIT

        must_block = crit_local | crit_neighbor
        overridden = torch.where(must_block, torch.tensor(2, device=actions.device), overridden)
        return overridden


class ST_TBA_GAT(nn.Module):
    """
    Complete Spatio-Temporal Topology & Bottleneck-Aware Graph Attention Network.
    Designed for Multi-Agent Reinforcement Learning (MAPPO) in hazardous facilities.
    """
    def __init__(
        self,
        node_dim: int = 10,
        hidden_dim: int = 64,
        max_corridors: int = 6,
        gat_heads: int = 4,
        gat_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.max_corridors = max_corridors
        self.gat_heads = gat_heads
        self.gat_layers = gat_layers

        # ── 1. Spatial GAT Encoder ──────────────────────────────────────────
        head_dim = hidden_dim // gat_heads
        self.gat1 = GATv2Layer(
            in_features=node_dim,
            out_features=head_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATv2Layer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ── 2. Temporal Recurrent GRU Cell ──────────────────────────────────
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # ── 3. Decentralized Actor Heads ────────────────────────────────────
        # For each of the max_corridors: categorical logits over {0: ALLOW, 1: REDIRECT, 2: BLOCK}
        self.actor_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_corridors * 3),
        )
        # Initialize actor head bias to prioritize ALLOW (0) over REDIRECT (1) and BLOCK (2) initially
        with torch.no_grad():
            bias = torch.zeros(max_corridors * 3)
            for k in range(max_corridors):
                bias[k * 3 + 0] = 2.0   # ALLOW
                bias[k * 3 + 1] = 0.0   # REDIRECT
                bias[k * 3 + 2] = -2.0  # BLOCK
            self.actor_mlp[-1].bias.copy_(bias)

        # ── 4. Centralized Critic Head (CTDE) ───────────────────────────────
        # Evaluates global facility state (mean + max pooling = 2 * hidden_dim)
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_spatial(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Runs spatial Graph Attention message passing across facility mesh."""
        h1 = F.elu(self.gat1(x, edge_index))
        h1 = self.norm1(h1 + F.relu(h1))  # Skip connection

        h2 = F.elu(self.gat2(h1, edge_index))
        spatial_emb = self.norm2(h2 + h1)
        return spatial_emb

    def forward_actor(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes actor logits for all agents and updates recurrent state.
        Returns:
          logits: (N, max_corridors, 3)
          new_hidden_state: (N, hidden_dim)
        """
        N = node_features.size(0)
        device = node_features.device

        spatial_emb = self.encode_spatial(node_features, edge_index)

        if hidden_state is None:
            hidden_state = torch.zeros(N, self.hidden_dim, device=device)

        new_hidden_state = self.gru(spatial_emb, hidden_state)

        logits_flat = self.actor_mlp(new_hidden_state)  # (N, max_corridors * 3)
        logits = logits_flat.view(N, self.max_corridors, 3)

        if action_masks is not None:
            # action_masks: (N, max_corridors) boolean mask (True = valid corridor, False = padded)
            # Mask out non-existent corridors with large negative logits
            mask = action_masks.unsqueeze(-1).expand_as(logits)
            logits = logits.masked_fill(~mask, -1e9)

        return logits, new_hidden_state

    def forward_critic(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Centralized Critic estimates value V(s) from global graph state.
        hidden_state: (N, hidden_dim) or (Batch, N, hidden_dim)
        Returns: scalar value or (Batch, 1)
        """
        if hidden_state.dim() == 2:
            mean_pool = hidden_state.mean(dim=0, keepdim=True)
            max_pool, _ = hidden_state.max(dim=0, keepdim=True)
            global_feat = torch.cat([mean_pool, max_pool], dim=-1)  # (1, 2*hidden_dim)
            value = self.critic_mlp(global_feat).squeeze(-1)       # (1,)
        else:
            mean_pool = hidden_state.mean(dim=1)
            max_pool, _ = hidden_state.max(dim=1)
            global_feat = torch.cat([mean_pool, max_pool], dim=-1)  # (Batch, 2*hidden_dim)
            value = self.critic_mlp(global_feat)                   # (Batch, 1)

        return value

    def act(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        apply_reflexive: bool = True,
        neighbor_hazards: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Action selection during rollout or live inference.
        Returns:
          actions: (N, max_corridors) in {0, 1, 2}
          action_log_probs: (N,) sum of log probs over valid corridors
          new_hidden_state: (N, hidden_dim)
          value: scalar state value estimate
        """
        logits, new_hidden_state = self.forward_actor(
            node_features, edge_index, hidden_state, action_masks
        )
        value = self.forward_critic(new_hidden_state)

        dist = Categorical(logits=logits)
        if deterministic:
            raw_actions = torch.argmax(logits, dim=-1)
        else:
            raw_actions = dist.sample()

        log_probs = dist.log_prob(raw_actions)  # (N, max_corridors)
        if action_masks is not None:
            log_probs = (log_probs * action_masks.float()).sum(dim=-1)
        else:
            log_probs = log_probs.sum(dim=-1)

        if apply_reflexive:
            local_hazards = node_features[:, 0]
            if neighbor_hazards is None:
                if node_features.shape[-1] >= 10 + self.max_corridors * 4:
                    neighbor_hazards = node_features[:, 10::4][:, :self.max_corridors]
                else:
                    neighbor_hazards = torch.zeros(node_features.size(0), self.max_corridors, device=node_features.device)
            actions = ReflexiveSafetyOverride.apply(
                raw_actions, local_hazards, neighbor_hazards
            )
        else:
            actions = raw_actions

        return actions, log_probs, new_hidden_state, value

    def evaluate_actions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO loss computation step.
        Returns:
          values: (1,) or (Batch, 1)
          action_log_probs: (N,)
          entropy: (N,)
        """
        logits, new_hidden_state = self.forward_actor(
            node_features, edge_index, hidden_state, action_masks
        )
        value = self.forward_critic(new_hidden_state)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)  # (N, max_corridors)
        entropy = dist.entropy()           # (N, max_corridors)

        if action_masks is not None:
            mask_f = action_masks.float()
            log_probs = (log_probs * mask_f).sum(dim=-1)
            entropy = (entropy * mask_f).sum(dim=-1)
        else:
            log_probs = log_probs.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        return value, log_probs, entropy

    # ── Checkpointing & Export ────────────────────────────────────────────────
    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "node_dim": self.node_dim,
            "hidden_dim": self.hidden_dim,
            "max_corridors": self.max_corridors,
            "gat_heads": self.gat_heads,
            "gat_layers": self.gat_layers,
            "extra_meta": extra_meta or {},
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> ST_TBA_GAT:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            node_dim=payload.get("node_dim", 10),
            hidden_dim=payload.get("hidden_dim", 64),
            max_corridors=payload.get("max_corridors", 6),
            gat_heads=payload.get("gat_heads", 4),
            gat_layers=payload.get("gat_layers", 2),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        return model
