from stable_baselines3 import PPO

import environment
from environment import env

env = env(num_agents=10, max_cycles=1000, render_mode='human')
env.reset()

model_circle = PPO.load("circle_model")
model_triangle = PPO.load("mountains_model")

step = 0
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        if step < 5000:
            environment.MOUNTAINS = False
            action = model_circle.predict(observation, deterministic=True)[0]
        else:
            environment.MOUNTAINS = True
            action = model_triangle.predict(observation, deterministic=True)[0]
    env.step(action)
    step += 1
env.close()
