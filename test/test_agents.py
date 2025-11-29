import unittest
import torch
from omegaconf import OmegaConf
from src.agents.ppo_agent import create_ppo_actor_critic
from src.envs.env import FormationEnv
from tensordict import TensorDict


class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test."""
        self.device = torch.device("cpu")
        # Create minimal config for testing
        self.cfg = OmegaConf.create(
            {
                "env": {
                    "num_agents": 5,
                    "arena_size": 10,
                    "max_steps": 100,
                    "shape_type": "circle",
                    "circle": {"center": [0.0, 0.0], "radius": 2.0},
                    "agent_size": 0.05,
                    "agent_accel": 3.0,
                    "agent_max_speed": 0.6,
                    "obs_keys_for_actor": ["observation"],
                },
                "algo": {
                    "hidden_dims": [64, 64],
                },
                "base": {
                    "device": "cpu",
                    "seed": 42,
                },
            }
        )
        self.env = FormationEnv(cfg=self.cfg, device=self.device)

    def test_agent_initialization(self):
        """Test that PPO actor-critic networks initialize correctly."""
        actor, critic = create_ppo_actor_critic(self.cfg, self.env)
        self.assertIsNotNone(actor)
        self.assertIsNotNone(critic)

    def test_select_action(self):
        """Test that actor network produces valid actions."""
        actor, _ = create_ppo_actor_critic(self.cfg, self.env)

        dummy_obs = TensorDict(
            {
                "observation": torch.randn(
                    self.cfg.env.num_agents,
                    self.env.observation_spec["observation"].shape[-1],
                )
            },
            batch_size=[self.cfg.env.num_agents],
        )

        output_td = actor(dummy_obs)
        self.assertIsNotNone(output_td)
        self.assertIn("action", output_td.keys())
        self.assertEqual(output_td["action"].shape[0], self.cfg.env.num_agents)


if __name__ == "__main__":
    unittest.main()
