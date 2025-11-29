import unittest
from src.agents.ppo_agent import PPOAgent


class TestPPOAgent(unittest.TestCase):
    def test_agent_initialization(self):
        agent = PPOAgent()
        self.assertIsNotNone(agent)

    def test_select_action(self):
        agent = PPOAgent()
        dummy_state = [0.0] * agent.state_dim  # Assumes agent has 'state_dim'
        action = agent.select_action(dummy_state)
        self.assertIsNotNone(action)
