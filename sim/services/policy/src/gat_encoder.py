import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

class GATEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 8,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(GATEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is not available but GATEncoder was requested.")
            
        self.conv1 = GATConv(
            in_channels=node_feature_dim,
            out_channels=embedding_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        
        self.conv2 = GATConv(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.layer_norm(x)
        return x


class MLPEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 8,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(node_feature_dim, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Ignore edge_index as this is an MLP fallback
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x


class SpatioTemporalGATEncoder(nn.Module):
    """
    Spatio-Temporal Graph Neural Network Encoder (ST-GNN).
    Combines Spatial GAT/MLP corridor message passing with a Gated Recurrent Unit (GRU)
    to model temporal dynamics of crowd velocity, rate-of-hazard-rise, and smoke movement.
    """
    def __init__(
        self,
        node_feature_dim: int = 8,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(SpatioTemporalGATEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 1. Spatial Graph Encoder (PyG GAT or MLP fallback)
        if PYG_AVAILABLE:
            self.spatial_encoder = GATEncoder(
                node_feature_dim=node_feature_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            self.spatial_encoder = MLPEncoder(
                node_feature_dim=node_feature_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
            
        # 2. Temporal Recurrent Cell (GRU)
        self.gru = nn.GRUCell(input_size=embedding_dim, hidden_size=embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Spatio-Temporal aggregation.
        
        Args:
            x: Node feature tensor [N, feature_dim] or [B, N, feature_dim]
            edge_index: Graph edge index tensor [2, E]
            hidden_state: Previous GRU hidden state [N, embedding_dim] or None
            
        Returns:
            Tuple of (st_embeddings, next_hidden_state)
        """
        # Spatial Graph Convolution
        spatial_emb = self.spatial_encoder(x, edge_index)
        
        # Temporal Recurrent Step (GRU)
        if spatial_emb.dim() == 2:
            if hidden_state is None:
                hidden_state = torch.zeros_like(spatial_emb)
            next_h = self.gru(spatial_emb, hidden_state)
        else:
            # Batched processing [B, N, D]
            B, N, D = spatial_emb.shape
            flat_spatial = spatial_emb.view(-1, D)
            if hidden_state is None:
                hidden_state = torch.zeros_like(flat_spatial)
            else:
                hidden_state = hidden_state.view(-1, D)
            next_h = self.gru(flat_spatial, hidden_state)
            next_h = next_h.view(B, N, D)
            
        st_embeddings = self.layer_norm(next_h)
        return st_embeddings, next_h


def create_encoder(
    node_feature_dim: int = 8,
    embedding_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    use_recurrent: bool = True,
    **kwargs
) -> nn.Module:
    if use_recurrent:
        return SpatioTemporalGATEncoder(
            node_feature_dim=node_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif PYG_AVAILABLE:
        return GATEncoder(
            node_feature_dim=node_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        return MLPEncoder(
            node_feature_dim=node_feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

