import os
import unittest
import torch
from simulator.policy.st_tba_gat import ST_TBA_GAT, GATv2Layer, ReflexiveSafetyOverride

class TestSTTBAGATPolicy(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 36
        self.node_dim = 41
        self.hidden_dim = 64
        self.max_corridors = 6

        self.model = ST_TBA_GAT(
            node_dim=self.node_dim,
            hidden_dim=self.hidden_dim,
            max_corridors=self.max_corridors,
            gat_heads=4,
            gat_layers=2,
        )

        edges = []
        for i in range(self.num_nodes):
            edges.append([i, (i + 1) % self.num_nodes])
            edges.append([(i + 1) % self.num_nodes, i])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t()

    def test_forward_and_act(self):
        x = torch.randn(self.num_nodes, self.node_dim)
        actions, log_probs, h_new, val = self.model.act(
            node_features=x,
            edge_index=self.edge_index,
            deterministic=True,
        )

        self.assertEqual(actions.shape, (self.num_nodes,))
        self.assertTrue(torch.all((actions >= 0) & (actions < 7)))
        self.assertEqual(log_probs.shape, (self.num_nodes,))
        self.assertEqual(h_new.shape, (self.num_nodes, self.hidden_dim))
        self.assertEqual(val.shape, (1,))

    def test_reflexive_override(self):
        # Create scenario with high hazard (>= 0.80)
        actions = torch.zeros(self.num_nodes, self.max_corridors, dtype=torch.long)
        node_hazards = torch.zeros(self.num_nodes)
        node_hazards[5] = 0.90  # Critical node

        neighbor_hazards = torch.zeros(self.num_nodes, self.max_corridors)
        neighbor_hazards[10, 2] = 0.85  # Critical corridor

        overridden = ReflexiveSafetyOverride.apply(actions, node_hazards, neighbor_hazards)

        # Node 5 all corridors must be forced to BLOCK (2)
        self.assertTrue(torch.all(overridden[5] == 2))
        # Node 10 corridor 2 must be forced to BLOCK (2)
        self.assertEqual(overridden[10, 2].item(), 2)
        # Node 0 corridor 0 should remain ALLOW (0)
        self.assertEqual(overridden[0, 0].item(), 0)

    def test_evaluate_actions_and_backprop(self):
        x = torch.randn(self.num_nodes, self.node_dim)
        h = torch.zeros(self.num_nodes, self.hidden_dim)
        actions = torch.randint(0, 7, (self.num_nodes,))

        values, log_probs, entropy = self.model.evaluate_actions(
            node_features=x,
            edge_index=self.edge_index,
            hidden_state=h,
            actions=actions,
        )

        loss = -log_probs.mean() + values.mean() - 0.01 * entropy.mean()
        self.model.zero_grad()
        loss.backward()

        # Check gradients exist on GAT weights
        self.assertIsNotNone(self.model.gat1.w_src.weight.grad)
        self.assertIsNotNone(self.model.critic_mlp[0].weight.grad)

    def test_checkpoint_save_and_load(self):
        save_path = "/tmp/test_st_tba_gat_ckpt.pt"
        self.model.save_checkpoint(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded = ST_TBA_GAT.load_checkpoint(save_path)
        self.assertEqual(loaded.node_dim, self.node_dim)
        self.assertEqual(loaded.hidden_dim, self.hidden_dim)

        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == "__main__":
    unittest.main()
