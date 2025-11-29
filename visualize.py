import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensordict.nn import TensorDictSequential
from torchrl.envs.utils import step_mdp
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


def generate_formation_gif(agent_trajectories, cfg, filename="formation.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw target shape
    if cfg.env.shape_type == "circle":
        center = cfg.env.circle.center  # [0.0, 0.0]
        radius = cfg.env.circle.radius
        shape_patch = plt.Circle(
            tuple(center), radius, color="r", fill=False, linestyle="--", alpha=0.3
        )
        ax.add_artist(shape_patch)
    elif cfg.env.shape_type == "star":
        # Implementation soon
        pass
    elif cfg.env.shape_type == "polygon":
        verts = cfg.env.star.vertices
        shape_patch = plt.Polygon(
            verts, closed=True, color="r", fill=False, linestyle="--", alpha=0.3
        )
        ax.add_artist(shape_patch)
    else:
        raise ValueError("Unknown shape type for visualization")

    scat = ax.scatter([], [], s=100)

    def init():
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        return (scat,)

    def update(frame):
        coords = agent_trajectories[frame]  # List of (x, y)
        scat.set_offsets(coords)
        return (scat,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(agent_trajectories), init_func=init, blit=True
    )
    ani.save(filename, writer="pillow", fps=30)
    plt.close(fig)


@hydra.main(
    version_base=None, config_path="./configs", config_name="experiment/default_exp"
)
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())

    # print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.base.device)
    torch.manual_seed(cfg.base.seed)

    # Create environment
    env = FormationEnv(cfg, device=device)

    # Create PPO model
    actor, critic = create_ppo_actor_critic(cfg, env)

    # Load a trained policy (assumes you have some loading mechanism)
    model_path = "./wandb/latest-run/files/models/actor_network.pt"
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        actor.load_state_dict(state_dict)
    else:
        print("WARNING: Model not found. Running random weights.")

    # Evaluation + Logging positions
    positions_over_time = []
    # Reset environment
    td = env.reset()
    print("Initial agent positions:", env.agent_positions)

    # Run loop until any agent is done, or max_steps reached
    positions_over_time = []
    with torch.no_grad():
        frames = 400  # increase if needed
        for i in range(frames):
            actor(td)
            td["action"] = td["loc"]
            td = env.step(td)

            # Log both agents and their assigned targets
            positions_over_time.append(env.agent_positions.cpu().numpy().tolist())

            # Normalize step (this solves visualization for now)
            td = step_mdp(td)

            if td["done"].any():
                break

    # Save GIF
    output_path = os.path.join(os.getcwd(), "formation.gif")
    print(f"Collected {len(positions_over_time)} frames")
    generate_formation_gif(positions_over_time, cfg, filename=output_path)
    print(f"GIF saved to: {output_path}")


if __name__ == "__main__":
    main()
