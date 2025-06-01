
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.utils.tensorboard import SummaryWriter
import tqdm

# Imports assuming main.py is in the root, and src is a package in the root
from src.envs.env import FormationEnv
from src.agents.ppo_agent import create_ppo_actor_critic
from src.rollout.evaluator import evaluate_policy

def generate_formation_gif(agent_trajectories, filename="formation.gif"):
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=100)

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return scat,

    def update(frame):
        coords = agent_trajectories[frame]  # List of (x, y)
        scat.set_offsets(coords)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(agent_trajectories),
                                  init_func=init, blit=True)
    ani.save(filename, writer='pillow', fps=60)
    plt.close(fig)

@hydra.main(version_base=None, config_path="./configs", config_name="experiment/default_exp")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())

    #print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.base.device)
    torch.manual_seed(cfg.base.seed)

    # Create environment
    env = FormationEnv(cfg, device=device)

    # Create PPO model
    actor, critic = create_ppo_actor_critic(cfg, env)

    # Load a trained policy (assumes you have some loading mechanism)
    # Optionally load checkpoint here

    # Evaluation + Logging positions
    positions_over_time = []
    # Reset environment
    td = env.reset()
    print("Initial agent positions:", env.agent_positions)

    # Run loop until any agent is done, or max_steps reached
    positions_over_time = []
    iter = 0
    while not td["done"].any() and iter < 2000:
        obs = td.select(*env.actor_obs_keys)

        with torch.no_grad():
            action = actor(obs)["action"]  # extract tensor from tensordict

        td.update({"action": action})
        td = env.step(td)

        # Log agent positions
        positions_over_time.append(env.agent_positions.cpu().numpy().tolist())
        iter += 1

    # Save GIF
    output_path = os.path.join(os.getcwd(), "formation.gif")
    print(f"Collected {len(positions_over_time)} frames")
    generate_formation_gif(positions_over_time, filename=output_path)
    print(f"GIF saved to: {output_path}")

if __name__ == "__main__":
    main()
