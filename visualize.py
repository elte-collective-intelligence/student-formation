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

# Imports assuming main.py is in the root, and src is a package in the root
from src.envs.env import FormationEnv
from src.agents.ppo_agent import create_ppo_actor_critic
from src.envs.shapes import make_star_vertices


def generate_formation_gif(
    agent_trajectories, target_trajectories, cfg, filename="formation.gif"
):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Store active patches to remove them later
    current_patches = []

    def get_shape_patches(config_source):
        """Generates a list of Matplotlib artists from a config block."""
        patches = []

        # Normalize: if it's a single shape, wrap it in a list so we can iterate
        if config_source.shape_type == "multishape":
            shape_configs = config_source.multishape
        else:
            shape_configs = [{"type": config_source.shape_type, **config_source}]
            if config_source.shape_type == "circle":
                shape_configs = [config_source.circle]
                shape_configs[0]["type"] = "circle"
            elif config_source.shape_type == "polygon":
                shape_configs = [config_source.polygon]
                shape_configs[0]["type"] = "polygon"
            elif config_source.shape_type == "star":
                shape_configs = [config_source.star]
                shape_configs[0]["type"] = "star"

        for s in shape_configs:
            if s.type == "circle":
                p = plt.Circle(
                    tuple(s.center),
                    s.radius,
                    color="r",
                    fill=False,
                    linestyle="--",
                    alpha=0.3,
                )
                patches.append(p)
            elif s.type == "polygon":
                p = plt.Polygon(
                    s.vertices,
                    closed=True,
                    color="r",
                    fill=False,
                    linestyle="--",
                    alpha=0.3,
                )
                patches.append(p)
            elif s.type == "star":
                verts = make_star_vertices(s.center, s.r1, s.r2, s.n_points)
                p = plt.Polygon(
                    verts, closed=True, color="r", fill=False, linestyle="--", alpha=0.3
                )
                patches.append(p)
        return patches

    # Setup Scatters
    scat_agents = ax.scatter([], [], s=100, c="blue", label="Agents", zorder=5)
    scat_targets = ax.scatter(
        [], [], s=50, c="green", marker="x", label="Targets", zorder=4
    )

    def init():
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        return scat_agents, scat_targets

    def update(frame):
        scat_agents.set_offsets(agent_trajectories[frame])
        scat_targets.set_offsets(target_trajectories[frame])

        reconfig_step = cfg.env.get("reconfig_step", None)

        needs_redraw = (frame == 0) or (
            reconfig_step is not None and frame == reconfig_step
        )

        if needs_redraw:
            for p in current_patches:
                p.remove()
            current_patches.clear()

            if (
                reconfig_step is not None
                and frame >= reconfig_step
                and "reconfig_shape" in cfg.env
            ):
                source = cfg.env.reconfig_shape
            else:
                source = cfg.env  # Initial state

            new_patches = get_shape_patches(source)
            for p in new_patches:
                ax.add_artist(p)
                current_patches.append(p)

        return scat_agents, scat_targets, *current_patches

    ani = animation.FuncAnimation(
        fig, update, frames=len(agent_trajectories), init_func=init, blit=False
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
