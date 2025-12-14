import torch
import numpy as np
from tensordict import TensorDict
from typing import Dict, List, Optional

class FormationMetrics:

    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.num_agents = env.num_agents
        self.arena_size = env.arena_size
        
    def _unwrap_next(self, tensordict):
        next_td = tensordict.get("next", None)
        return next_td if next_td is not None else tensordict
        
    def compute_boundary_error(
        self, agent_positions: torch.Tensor, target_shape=None
    ) -> Dict[str, float]:
        
        # Placeholder for boundary error computation
        return "Dummy"
        
    def evaluate_episode(
        self, policy, num_episodes: int = 3, render: bool = False
    ) -> Dict[str, any]:
        
        episode_metrics = []

        for ep in range(num_episodes):
            metrics_per_step = {
                "boundary_errors": [],
            }

            td = self.env.reset()

            steps_limit = max(1, int(self.env.max_steps))
            for step in range(steps_limit):
                # Get action from policy
                with torch.no_grad():
                    td_policy = td.select(*policy.in_keys)
                    print("td_policy", td_policy)
                    policy(td_policy)

                td_step = TensorDict(
                    {"action": td_policy["action"]},
                    batch_size=[self.num_agents],
                    device=self.device,
                )

                transition_td = self.env.step(td_step)
                next_td = self._unwrap_next(transition_td)

                # Collect metrics at this step
                positions = self.env.agent_positions.clone()
                print("positions", positions)
            
                be_metrics = self.compute_boundary_error(positions)

                metrics_per_step["boundary_errors"].append(be_metrics)
                # print("boundary_errors", metrics_per_step["boundary_errors"])
            

            if render:
                self.env.render(mode="human")

            # Check if over
            done_tensor = next_td.get("done", None)
            done = bool(done_tensor.any().item()) if done_tensor is not None else False
            
            if done:
                break

            td = next_td

            episode_metrics.append(metrics_per_step)

        # Aggregate across episodes
        return episode_metrics