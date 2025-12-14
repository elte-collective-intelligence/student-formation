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
import tensordict
import glob

# Imports assuming main.py is in the root, and src is a package in the root
from src.envs.env import FormationEnv
from src.agents.ppo_agent import create_ppo_actor_critic
from src.envs.shapes import make_star_vertices


def generate_formation_gif(
    agent_trajectories, target_trajectories, cfg, filename="formation.gif"
):
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
        verts = make_star_vertices(
            center=cfg.env.star.center,
            r1=cfg.env.star.r1,
            r2=cfg.env.star.r2,
            n_points=cfg.env.star.n_points,
        )
        shape_patch = plt.Polygon(
            verts, closed=True, color="r", fill=False, linestyle="--", alpha=0.3
        )
        ax.add_artist(shape_patch)
    elif cfg.env.shape_type == "polygon":
        verts = cfg.env.polygon.vertices
        shape_patch = plt.Polygon(
            verts, closed=True, color="r", fill=False, linestyle="--", alpha=0.3
        )
        ax.add_artist(shape_patch)
    else:
        raise ValueError("Unknown shape type for visualization")

    # Agent Scatter (Blue)
    scat_agents = ax.scatter([], [], s=100, c="blue", label="Agents", zorder=5)

    # Target Scatter (Green X)
    scat_targets = ax.scatter(
        [], [], s=50, c="green", marker="x", label="Targets", zorder=4
    )

    def init():
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        return (scat_agents, scat_targets)

    def update(frame):
        coords = agent_trajectories[frame]  # List of (x, y)
        scat_agents.set_offsets(coords)
        target_coords = target_trajectories[frame]
        scat_targets.set_offsets(target_coords)

        return (scat_agents, scat_targets)

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
    
    # Select the model from wandb
    model_path = "./wandb/latest-run/files/models/actor_network.pt"
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        actor.load_state_dict(state_dict)
    else:
        # Load a trained model from wandb, if latest-run does not exist
        run_dirs = glob.glob(os.path.join("wandb", "run-*"))
        latest_run = max(run_dirs, key=os.path.getmtime, default="42")
        if latest_run == "42":
            print("WARNING: No trained model found. Running random weights.")
        else:
            model_path = os.path.join(latest_run, "files", "models", "actor_network.pt")
            print(f"Loading model from: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            actor.load_state_dict(state_dict)

    # Evaluation + Logging positions
    positions_over_time = []
    # Reset environment
    td = env.reset()
    print("Initial agent positions:", env.agent_positions)

    # Run loop until any agent is done, or max_steps reached
    positions_over_time = []
    target_pos_log = []
    with torch.no_grad():
        frames = 400  # increase if needed
        for i in range(frames):
            actor(td)
            td["action"] = td["loc"]
            td = env.step(td)

            # Log both agents and their assigned targets
            positions_over_time.append(env.agent_positions.cpu().numpy().tolist())
            target_pos_log.append(env.assigned_target_positions.cpu().numpy().tolist())

            # Normalize step (this solves visualization for now)
            td = step_mdp(td)

            if td["done"].any():
                break

    # Save GIF
    output_path = os.path.join(os.getcwd(), "formation.gif")
    print(f"Collected {len(positions_over_time)} frames")
    generate_formation_gif(
        positions_over_time, target_pos_log, cfg, filename=output_path
    )
    print(f"GIF saved to: {output_path}")


if __name__ == "__main__":
    main()
