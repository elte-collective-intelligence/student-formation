import unittest
import torch
from omegaconf import OmegaConf
from src.envs.env import FormationEnv
from tensordict import TensorDict


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test."""
        self.device = torch.device("cpu")
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
                }
            }
        )

    def test_env_initialization(self):
        """Test that environment initializes correctly."""
        env = FormationEnv(cfg=self.cfg, device=self.device)
        self.assertIsNotNone(env)
        self.assertTrue(hasattr(env, "reset"))
        self.assertTrue(hasattr(env, "step"))

    def test_env_reset(self):
        """Test that environment reset works."""
        env = FormationEnv(cfg=self.cfg, device=self.device)
        td = env.reset()
        self.assertIsNotNone(td)
        self.assertIn("observation", td.keys())

    def test_env_step(self):
        """Test that environment step works."""
        env = FormationEnv(cfg=self.cfg, device=self.device)
        env.reset()

        action_td = TensorDict(
            {"action": env.action_spec.rand()}, batch_size=env.batch_size
        )
        result_td = env.step(action_td)
        self.assertIsNotNone(result_td)


if __name__ == "__main__":
    unittest.main()
