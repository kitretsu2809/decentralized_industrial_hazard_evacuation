import torch
import torch.nn as nn
import torch.nn.functional as F

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


def create_encoder(
    node_feature_dim: int = 8,
    embedding_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    if PYG_AVAILABLE:
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
