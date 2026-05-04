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

    def test_sdf_circle_known_points(self):
        """Checks if the circle distance calculation is correct."""

        env = FormationEnv(cfg=self.cfg, device=self.device)
        center = torch.tensor(self.cfg.env.circle.center, dtype=torch.float32)
        radius = float(self.cfg.env.circle.radius)

        sdf_center = env.target_shape.signed_distance(center.unsqueeze(0))[0].item()

        on_boundary = center + torch.tensor([radius, 0.0], dtype=torch.float32)
        sdf_boundary = env.target_shape.signed_distance(on_boundary.unsqueeze(0))[
            0
        ].item()

        self.assertAlmostEqual(sdf_center, -radius, places=4)
        self.assertAlmostEqual(sdf_boundary, 0.0, places=4)

    def test_assignment_hungarian_is_one_to_one(self):
        """Hungarian assignment should make one-to-one assignments. Greedy will prefer closer targets."""

        cfg_greedy = OmegaConf.merge(
            self.cfg, {"env": {"num_agents": 2, "assignment_method": "greedy"}}
        )
        env_greedy = FormationEnv(cfg=cfg_greedy, device=self.device)
        target0 = env_greedy.shape_boundary_points[0].detach().cpu()
        env_greedy.agent_positions = torch.stack(
            [
                target0 + torch.tensor([0.01, 0.0]),
                target0 + torch.tensor([0.02, 0.0]),
            ],
            dim=0,
        ).to(self.device)
        env_greedy._FormationEnv__update_assignments()
        unique_greedy = torch.unique(
            env_greedy.assigned_target_positions.cpu(), dim=0
        ).shape[0]

        cfg_hungarian = OmegaConf.merge(
            self.cfg, {"env": {"num_agents": 2, "assignment_method": "hungarian"}}
        )
        env_hungarian = FormationEnv(cfg=cfg_hungarian, device=self.device)
        target0_h = env_hungarian.shape_boundary_points[0].detach().cpu()
        env_hungarian.agent_positions = torch.stack(
            [
                target0_h + torch.tensor([0.01, 0.0]),
                target0_h + torch.tensor([0.02, 0.0]),
            ],
            dim=0,
        ).to(self.device)
        env_hungarian._FormationEnv__update_assignments()
        unique_hungarian = torch.unique(
            env_hungarian.assigned_target_positions.cpu(), dim=0
        ).shape[0]

        self.assertEqual(unique_hungarian, 2)
        self.assertEqual(unique_greedy, 1)

    def test_observation_dimension_sdf_on(self):
        env = FormationEnv(cfg=self.cfg, device=self.device)
        td = env.reset()
        self.assertEqual(
            td["observation"].shape[-1], env.observation_spec["observation"].shape[-1]
        )
        self.assertEqual(td["observation"].shape[-1], env._obs_feature_dim())

    def test_observation_dimension_sdf_off(self):
        cfg = OmegaConf.merge(self.cfg, {"env": {"use_sdf_obs": False, "knn_k": 2}})
        env = FormationEnv(cfg=cfg, device=self.device)
        td = env.reset()
        self.assertEqual(env._obs_feature_dim(), 2 + 2 + 2)
        self.assertEqual(td["observation"].shape[-1], 6)

    def test_circle_boundary_frame(self):
        env = FormationEnv(cfg=self.cfg, device=self.device)
        pts = torch.tensor(
            [[3.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device=self.device
        )
        c, n, t = env.target_shape.boundary_frame(pts)
        self.assertEqual(c.shape, (2, 2))
        norms = torch.norm(n, dim=-1)
        self.assertTrue((norms > 0.99).all())


if __name__ == "__main__":
    unittest.main()
