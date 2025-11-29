import unittest
from src.envs.env import make_env


class TestEnvironment(unittest.TestCase):
    def test_make_env(self):
        env = make_env()
        self.assertIsNotNone(env)
        self.assertTrue(hasattr(env, "reset"))
        self.assertTrue(hasattr(env, "step"))
