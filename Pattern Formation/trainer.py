import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

import environment


env = environment.parallel_env(num_agents=10, max_cycles=1000)

env.reset()

env = ss.multiagent_wrappers.pad_observations_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
env = VecMonitor(env, "logs/")

model = PPO("MlpPolicy", env, verbose=1, batch_size=256, tensorboard_log="logs/tensorboard")

for iters in range(300):
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
    model.save(f"models/{iters}")

env.close()
