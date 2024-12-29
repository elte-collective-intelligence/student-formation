import os
import sys
import wandb
from wandb.integration.sb3 import WandbCallback
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from pettingzoo.utils.conversions import aec_to_parallel
import matplotlib.pyplot as plt
import numpy as np

# Dynamically add the project root to the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.environment import env  # Import after updating sys.path

# Define Early Stopping Callback
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, reward_threshold, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self):
        if self.locals["rewards"].mean() >= self.reward_threshold:
            print("Stopping training early as reward threshold reached.")
            return False  # Stop training
        return True

# Reward Tracking Callback
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Check if the episode is done and log rewards
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True

# Add the project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Define paths for logs and outputs
log_path = os.path.join(ROOT_DIR, 'logs', 'tensorboard')
model_save_dir = os.path.join(ROOT_DIR, 'models')
output_dir = os.path.join(ROOT_DIR, 'outputs')

# Ensure directories exist
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Initialize W&B
wandb.init(
    project="multi-agent-rl-simulation",  # Replace with your project name
    config={
        "num_agents": 10,
        "max_cycles": 1000,
        "algorithm": "PPO",
        "batch_size": 256,
        "learning_rate": 3e-4,
        "total_timesteps": 50000,
        "iterations": 10
    },
    sync_tensorboard=True  # Sync TensorBoard logs with W&B
)

# Initialize the environment
env = env(num_agents=wandb.config.num_agents, max_cycles=wandb.config.max_cycles)  # AEC environment
env = aec_to_parallel(env)  # Convert to ParallelEnv
env = ss.multiagent_wrappers.pad_observations_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
env = VecMonitor(env, log_path)

# Define the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=wandb.config.batch_size,
    learning_rate=wandb.config.learning_rate,
    tensorboard_log=log_path
)

# Define callbacks
wandb_callback = WandbCallback(
    gradient_save_freq=100,  # Log gradients every 100 steps
    model_save_path=model_save_dir,  # Save models to W&B
    verbose=1
)
early_stopping = EarlyStoppingCallback(reward_threshold=100)

# Track metrics for plotting
rewards = []
iterations = []

# Training loop
for iters in range(wandb.config.iterations):
    print(f"Starting iteration {iters}...")
    reward_callback = RewardTrackingCallback()
    model.learn(
        total_timesteps=wandb.config.total_timesteps,
        callback=[wandb_callback, early_stopping, reward_callback],  # Include reward tracking callback
        reset_num_timesteps=False
    )

    # Calculate mean reward after each iteration
    if reward_callback.rewards:
        mean_reward = np.mean(reward_callback.rewards)
        rewards.append(mean_reward)
        wandb.log({"mean_reward": mean_reward})

    iterations.append(iters)

    # Save the model
    model_save_path = os.path.join(model_save_dir, f"model_{iters}.zip")
    model.save(model_save_path)
    print(f"Iteration {iters} complete. Model saved at {model_save_path}. Mean reward: {mean_reward}")

# Plot and log rewards
plt.figure(figsize=(10, 6))
plt.plot(iterations, rewards, label="Mean Reward", color="blue")
plt.fill_between(
    iterations,
    np.array(rewards) - np.std(rewards),
    np.array(rewards) + np.std(rewards),
    color="blue",
    alpha=0.3,  # Transparency
    label="Confidence Interval"
)
plt.title("Training Rewards Over Iterations", fontsize=16)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plot_path = os.path.join(output_dir, "reward_plot.png")
plt.savefig(plot_path)
plt.show()

wandb.log({"Reward Plot": wandb.Image(plot_path)})  # Log plot to W&B

# Finish W&B run
wandb.finish()

# Close the environment
env.close()
