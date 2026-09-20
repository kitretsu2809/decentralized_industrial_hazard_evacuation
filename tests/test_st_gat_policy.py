import unittest
import os
import torch
import numpy as np

from sim.services.policy.src.gat_encoder import SpatioTemporalGATEncoder, create_encoder
from sim.services.policy.src.actor_critic import ActorCritic, Actor
from sim.services.training.src.train import MAPPOTrainer

class TestSpatioTemporalGATPolicy(unittest.TestCase):
    def setUp(self):
        self.node_feature_dim = 8
        self.embedding_dim = 64
        self.max_actions = 6
        self.num_nodes = 10
        self.num_agents = 10
        
        # Create dummy ring graph
        edges = []
        for i in range(self.num_nodes):
            edges.append([i, (i + 1) % self.num_nodes])
            edges.append([(i + 1) % self.num_nodes, i])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
    def test_st_gat_encoder_recurrent_evolution(self):
        """Test that GRU hidden state evolves across consecutive time steps."""
        encoder = SpatioTemporalGATEncoder(
            node_feature_dim=self.node_feature_dim,
            embedding_dim=self.embedding_dim
        )
        encoder.eval()
        
        # Step 1: Initial state
        x_t1 = torch.randn(self.num_nodes, self.node_feature_dim)
        emb_t1, h_t1 = encoder(x_t1, self.edge_index, hidden_state=None)
        
        self.assertEqual(emb_t1.shape, (self.num_nodes, self.embedding_dim))
        self.assertEqual(h_t1.shape, (self.num_nodes, self.embedding_dim))
        
        # Step 2: Next time step with previous hidden state
        x_t2 = torch.randn(self.num_nodes, self.node_feature_dim)
        emb_t2, h_t2 = encoder(x_t2, self.edge_index, hidden_state=h_t1)
        
        self.assertEqual(emb_t2.shape, (self.num_nodes, self.embedding_dim))
        self.assertEqual(h_t2.shape, (self.num_nodes, self.embedding_dim))
        
        # Verify that hidden state evolved and is not identical to initial
        self.assertFalse(torch.allclose(h_t1, h_t2, atol=1e-4))
        print("  [PASSED] ST-GAT-GRU Recurrent Hidden State Evolution Verified")

    def test_actor_critic_forward_pass(self):
        """Test ActorCritic with recurrent encoder outputting valid action logits and values."""
        ac = ActorCritic(
            node_feature_dim=self.node_feature_dim,
            embedding_dim=self.embedding_dim,
            max_actions=self.max_actions,
            num_agents=self.num_agents,
            use_recurrent=True
        )
        ac.eval()
        
        x = torch.randn(self.num_nodes, self.node_feature_dim)
        mask = torch.ones(self.max_actions, dtype=torch.bool)
        
        actions, log_prob, value, next_h = ac.get_action(
            node_features=x,
            edge_index=self.edge_index,
            agent_idx=0,
            action_mask=mask,
            hidden_state=None,
            deterministic=True
        )
        
        edge_act, door_act, vert_act = actions
        
        self.assertEqual(edge_act.shape[-1], self.max_actions)
        self.assertIsNotNone(value)
        self.assertEqual(next_h.shape, (self.num_nodes, self.embedding_dim))
        print("  [PASSED] ActorCritic GAT-GRU Forward Pass Verified")

    def test_onnx_model_export(self):
        """Test exporting the Actor network to standard ONNX format."""
        output_path = "data/models/policy.onnx"
        trainer = MAPPOTrainer()
        exported_path = trainer.export_onnx(output_path=output_path)
        
        self.assertTrue(os.path.exists(exported_path))
        size_kb = os.path.getsize(exported_path) / 1024
        self.assertGreater(size_kb, 5)  # At least 5 KB
        print(f"  [PASSED] ONNX Model Export Verified ({size_kb:.2f} KB at {exported_path})")

if __name__ == '__main__':
    unittest.main()
