"""
Unit Tests for Permutation Invariance of Graph Attention Network Policy.
"""

import torch
import numpy as np
from agents.gat_policy import PermutationInvariantGATPolicy


def test_gat_permutation_invariance():
    torch.manual_seed(42)
    policy = PermutationInvariantGATPolicy(node_dim=5, edge_dim=3, hidden_dim=32, max_corridors=4, heads=2)
    policy.eval()

    num_nodes = 5
    # Random node features
    x = torch.randn(num_nodes, 5)

    # Simple undirected chain 0-1-2-3-4
    edges = [
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ]
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), 3)

    # 1. Forward original graph
    with torch.no_grad():
        _, h_orig, ctx_orig = policy.extract_graph_embeddings(x, edge_index, edge_attr)
        val_orig = policy.forward_critic(ctx_orig)

    # 2. Apply arbitrary permutation of nodes: p = [3, 0, 4, 1, 2]
    perm = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
    inv_perm = torch.zeros_like(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i

    x_perm = x[perm]

    # Remap edge indices
    edge_index_perm = inv_perm[edge_index]

    with torch.no_grad():
        _, h_perm, ctx_perm = policy.extract_graph_embeddings(x_perm, edge_index_perm, edge_attr)
        val_perm = policy.forward_critic(ctx_perm)

    # Permuted graph must yield identical scalar value V(s) = V(pi(s))
    assert torch.isclose(val_orig, val_perm, atol=1e-4), f"Critic value must be strictly permutation-invariant: {val_orig.item()} vs {val_perm.item()}"

    # Node embeddings of permuted graph must match permuted original node embeddings
    h_orig_permuted = h_orig[perm]
    assert torch.allclose(h_perm, h_orig_permuted, atol=1e-4), "Node embeddings must be permutation-equivariant"


if __name__ == "__main__":
    test_gat_permutation_invariance()
    print("GAT permutation invariance test passed!")
