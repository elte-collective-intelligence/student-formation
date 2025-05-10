def evaluate_policy(policy, env, episodes=5):
    returns = []
    for _ in range(episodes):
        td = env.reset()
        done = torch.tensor([False])
        total_reward = 0.0
        while not done.item():
            action_td = policy(td)
            td = env.step(action_td)
            total_reward += td["reward"].item()
            done = td["done"]
        returns.append(total_reward)
    return sum(returns) / len(returns)
