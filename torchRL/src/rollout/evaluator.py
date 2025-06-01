import torch
from tensordict import TensorDict

def evaluate_policy(env, policy, num_episodes=5, max_steps_per_episode=None):
    """
    Basic evaluation function.
    Runs the policy in the environment for a few episodes and returns average reward.
    For CTDE, ensure policy only uses local observations during execution.
    The policy module created in ppo_agent should already be set up for this
    if its `in_keys` are restricted to local agent observations.
    """
    total_rewards = 0
    if max_steps_per_episode is None:
        max_steps_per_episode = env.max_steps # From env config

    for _ in range(num_episodes):
        episode_reward = 0
        # Reset returns a TensorDict with shape [num_agents, ...]
        # For a shared policy, we expect observations for each agent.
        td = env.reset()

        for step in range(max_steps_per_episode):
            # Policy expects batch_size leading dims. Here, num_agents is the "batch".
            # If policy.in_keys are ["obs_self", "obs_target_vector"], it will pick these up.
            td_policy = td.select(*policy.in_keys) # Make sure policy gets only what it needs
            
            with torch.no_grad():
                policy(td_policy) # Populates td_policy with "action", "log_prob" etc.
            
            # The action from policy might be for all agents [num_agents, action_dim]
            # The environment step expects a tensordict with "action" key
            td_step = TensorDict({"action": td_policy["action"]}, batch_size=[env.num_agents], device=env.device)
            
            td = env.step(td_step)
            
            # Sum rewards over all agents for this step
            episode_reward += td["reward"].sum().item() # Sum agent rewards
            
            if td["done"].any(): # If any agent is done, or if global done is true
                break
        total_rewards += episode_reward
    
    return total_rewards / num_episodes