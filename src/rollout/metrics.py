import torch
import numpy as np
from tensordict import TensorDict
from typing import Dict, List, Optional

class FormationMetrics:

    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.num_agents = env.num_agents
        self.agent_size_world_units = env.agent_size_world_units
        self.arena_size = env.arena_size
        
    def _unwrap_next(self, tensordict):
        next_td = tensordict.get("next", None)
        return next_td if next_td is not None else tensordict
        
    def compute_boundary_error(
        self, agent_positions: torch.Tensor, target_shape=None
    ) -> Dict[str, float]:
        
        # Check if shape was not provided
        if target_shape is None:
            target_shape = self.env.target_shape

        # Compute distance to boundary
        sdf = target_shape.signed_distance(agent_positions)
        boundary_errors = torch.abs(sdf)

        # Consider agents right on the boundary
        threshold = self.agent_size_world_units # * 2.0 modify threshold if needed
        on_boundary = (boundary_errors < threshold).float()

        metrics = {
            "boundary_error_mean": boundary_errors.mean().item(),
            "boundary_error_max": boundary_errors.max().item(),
            "agents_on_boundary_pct": (on_boundary.mean() * 100.0).item(),
        }
        # print("Boundary Error Metrics:", metrics)
        return metrics
        
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
            for step in range(3):
                # Get action from policy
                with torch.no_grad():
                    td_policy = td.select(*policy.in_keys)
                    policy(td_policy)

                td_step = TensorDict(
                    {"action": td_policy["action"]},
                    batch_size=[self.num_agents],
                    device=self.device,
                )

                transition_td = self.env.step(td_step)
                next_td = self._unwrap_next(transition_td)

                # Collect metrics at this step from position of agents
                positions = self.env.agent_positions.clone()
                # print("positions", positions)
            
                be_metrics = self.compute_boundary_error(positions, target_shape=self.env.target_shape)

                metrics_per_step["boundary_errors"].append(be_metrics)
                # print("boundary_errors", metrics_per_step["boundary_errors"])
            
                # Check if over
                done_tensor = next_td.get("done", None)
                done = bool(done_tensor.any().item()) if done_tensor is not None else False
                
                if done:
                    break

                td = next_td

            episode_metrics.append(metrics_per_step)

        # Aggregate across episodes
        # print("Episode Metrics Collected:", episode_metrics)
        aggregated = self._aggregate_metrics(episode_metrics)
        return aggregated
    
    def _aggregate_metrics(
        self, episode_metrics: List
    ) -> Dict[str, any]:
        aggregated = {}

        final_step_metrics = [
            ep["boundary_errors"][-1]
            for ep in episode_metrics
            if ep.get("boundary_errors")
        ]

        if not final_step_metrics:
            return {
                "boundary_error_mean": 0.0,
                "boundary_error_max": 0.0,
                "agents_on_boundary_pct": 0.0,
            }

        aggregated["boundary_error_mean"] = float(np.mean([m["boundary_error_mean"] for m in final_step_metrics]))
        aggregated["boundary_error_max"] = float(np.mean([m["boundary_error_max"] for m in final_step_metrics]))
        aggregated["agents_on_boundary_pct"] = float(np.mean([m["agents_on_boundary_pct"] for m in final_step_metrics]))

        return aggregated