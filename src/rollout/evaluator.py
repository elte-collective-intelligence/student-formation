import torch
from tensordict import TensorDict
from .metrics import FormationMetrics


def evaluate_policy(env, policy, num_episodes=5, max_steps_per_episode=None):
    total_rewards = 0
    if max_steps_per_episode is None:
        max_steps_per_episode = env.max_steps

    for _ in range(num_episodes):
        episode_reward = 0
        td = env.reset()

        for step in range(max_steps_per_episode):
            td_policy = td.select(*policy.in_keys)

            with torch.no_grad():
                policy(td_policy)

            td_step = TensorDict(
                {"action": td_policy["action"]},
                batch_size=[env.num_agents],
                device=env.device,
            )

            td = env.step(td_step)

            episode_reward += td["reward"].sum().item()

            if td["done"].any():
                break
        total_rewards += episode_reward

    return total_rewards / num_episodes


def evaluate_with_metrics(env, policy, num_episodes=3, render=False):
    metrics_evaluator = FormationMetrics(env, device=env.device)
    aggregated = metrics_evaluator.evaluate_episode(
        policy, num_episodes=num_episodes, render=render
    )

    return aggregated
