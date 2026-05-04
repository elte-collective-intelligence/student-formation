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

        if target_shape is None:
            target_shape = self.env.target_shape

        sdf = target_shape.signed_distance(agent_positions)
        boundary_errors = torch.abs(sdf)

        threshold = self.agent_size_world_units
        on_boundary = (boundary_errors < threshold).float()

        metrics = {
            "boundary_error_mean": boundary_errors.mean().item(),
            "boundary_error_max": boundary_errors.max().item(),
            "agents_on_boundary_pct": (on_boundary.mean() * 100.0).item(),
        }
        return metrics

    def compute_uniformity(self, agent_positions: torch.Tensor) -> Dict[str, float]:

        if self.num_agents < 2:
            return {
                "uniformity_nn_distance_mean": 0.0,
                "uniformity_nn_distance_std": 0.0,
                "uniformity_coefficient": 0.0,
            }

        dist_matrix = torch.cdist(agent_positions, agent_positions)
        dist_matrix.fill_diagonal_(float("inf"))

        nn_distances, _ = torch.min(dist_matrix, dim=1)

        mean_nn = nn_distances.mean().item()
        std_nn = nn_distances.std().item() if len(nn_distances) > 1 else 0.0

        cv = std_nn / (mean_nn + 0.000001)

        metrics = {
            "uniformity_nn_distance_mean": mean_nn,
            "uniformity_nn_distance_std": std_nn,
            "uniformity_coefficient": cv,
        }
        return metrics

    def compute_collisions(
        self, agent_positions: torch.Tensor, collision_threshold: Optional[float] = None
    ) -> Dict[str, float]:

        if collision_threshold is None:
            collision_threshold = 1.5 * self.agent_size_world_units

        if self.num_agents < 2:
            return {
                "collision_count": 0,
                "collision_rate_pct": 0.0,
            }

        dist_matrix = torch.cdist(agent_positions, agent_positions)
        dist_matrix.fill_diagonal_(float("inf"))

        colliding_pairs = (dist_matrix < collision_threshold).sum().item() // 2
        total_pairs = (self.num_agents * (self.num_agents - 1)) // 2
        collision_rate = (colliding_pairs / max(total_pairs, 1)) * 100.0

        metrics = {
            "collision_count": int(colliding_pairs),
            "collision_rate_pct": collision_rate,
        }
        return metrics

    def evaluate_episode(
        self, policy, num_episodes: int = 3, render: bool = False
    ) -> Dict[str, any]:

        episode_metrics = []
        reconfig_recovery_times = []
        track_reconfig = self.env.reconfig_step is not None
        thr = float(self.env.cfg.env.get("reconfig_success_threshold", 0.12))
        hold_n = int(self.env.cfg.env.get("reconfig_success_hold", 5))

        for ep in range(num_episodes):
            metrics_per_step = {
                "boundary_errors": [],
                "uniformities": [],
                "collisions": [],
            }
            rec_time = None
            hold = 0

            td = self.env.reset()
            steps_limit = min(500, int(self.env.max_steps))
            for step in range(steps_limit):
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

                positions = self.env.agent_positions.clone()

                be_metrics = self.compute_boundary_error(
                    positions, target_shape=self.env.target_shape
                )
                metrics_per_step["boundary_errors"].append(be_metrics)

                un_metrics = self.compute_uniformity(positions)
                metrics_per_step["uniformities"].append(un_metrics)

                col_metrics = self.compute_collisions(positions)
                metrics_per_step["collisions"].append(col_metrics)

                if track_reconfig and rec_time is None:
                    if (
                        self.env.has_reconfigured
                        and self.env.current_step > self.env.reconfig_step
                    ):
                        be_mean = torch.abs(
                            self.env.target_shape.signed_distance(
                                self.env.agent_positions
                            )
                        ).mean().item()
                        if be_mean < thr:
                            hold += 1
                            if hold >= hold_n:
                                rec_time = float(
                                    self.env.current_step - self.env.reconfig_step
                                )
                        else:
                            hold = 0

                done_tensor = next_td.get("done", None)
                done = (
                    bool(done_tensor.any().item()) if done_tensor is not None else False
                )

                if done:
                    break

                td = next_td

            episode_metrics.append(metrics_per_step)
            if track_reconfig and rec_time is not None:
                reconfig_recovery_times.append(rec_time)

        aggregated = self._aggregate_metrics(episode_metrics)
        if track_reconfig and reconfig_recovery_times:
            aggregated["reconfiguration_time_mean"] = float(
                np.mean(reconfig_recovery_times)
            )
        elif track_reconfig:
            aggregated["reconfiguration_time_mean"] = float("nan")
        return aggregated

    def _aggregate_metrics(self, episode_metrics: List) -> Dict[str, any]:
        aggregated = {}

        boundary_final_step_metrics = [
            ep["boundary_errors"][-1]
            for ep in episode_metrics
            if ep.get("boundary_errors")
        ]

        if not boundary_final_step_metrics:
            return {
                "boundary_error_mean": 0.0,
                "boundary_error_max": 0.0,
                "agents_on_boundary_pct": 0.0,
            }

        aggregated["boundary_error_mean"] = float(
            np.mean([m["boundary_error_mean"] for m in boundary_final_step_metrics])
        )
        aggregated["boundary_error_max"] = float(
            np.mean([m["boundary_error_max"] for m in boundary_final_step_metrics])
        )
        aggregated["agents_on_boundary_pct"] = float(
            np.mean([m["agents_on_boundary_pct"] for m in boundary_final_step_metrics])
        )

        uniformity_final_step_metrics = [
            ep["uniformities"][-1] for ep in episode_metrics if ep.get("uniformities")
        ]

        if not uniformity_final_step_metrics:
            aggregated.update(
                {
                    "uniformity_nn_distance_mean": 0.0,
                    "uniformity_nn_distance_std": 0.0,
                    "uniformity_coefficient": 0.0,
                }
            )
        else:
            aggregated["uniformity_nn_distance_mean"] = float(
                np.mean(
                    [
                        m["uniformity_nn_distance_mean"]
                        for m in uniformity_final_step_metrics
                    ]
                )
            )
            aggregated["uniformity_nn_distance_std"] = float(
                np.mean(
                    [
                        m["uniformity_nn_distance_std"]
                        for m in uniformity_final_step_metrics
                    ]
                )
            )
            aggregated["uniformity_coefficient"] = float(
                np.mean(
                    [m["uniformity_coefficient"] for m in uniformity_final_step_metrics]
                )
            )

        collision_all_steps_metrics = [
            ep["collisions"] for ep in episode_metrics if ep.get("collisions")
        ]

        if not collision_all_steps_metrics:
            aggregated.update(
                {
                    "collision_count": 0,
                    "collision_rate_pct": 0.0,
                }
            )
        else:
            episode_collisions_total = []
            episode_collisions_rates = []

            for ep_metrics in episode_metrics:
                collisions = ep_metrics.get("collisions", [])
                ep_total_collisions = sum(m["collision_count"] for m in collisions)
                ep_total_collisions_rate = sum(
                    m["collision_rate_pct"] for m in collisions
                )
                episode_collisions_total.append(ep_total_collisions)
                episode_collisions_rates.append(
                    ep_total_collisions_rate / max(len(collisions), 1)
                )

            aggregated["collision_count"] = (
                float(np.mean(episode_collisions_total))
                if episode_collisions_total
                else 0.0
            )

            aggregated["collision_rate_pct"] = (
                float(np.mean(episode_collisions_rates))
                if episode_collisions_rates
                else 0.0
            )

        return aggregated
