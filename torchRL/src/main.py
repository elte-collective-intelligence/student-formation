import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
import torch

from src.envs.env import Env
from src.agents.ppo_agents import policy, critic
from src.rollout.evaulator import evaluate_policy
from src.rollout.visualizer import record_episode

@hydra.main(version_base=None, config_path="configs/experiment", config_name="exp.yaml")
def main(cfg: DictConfig):
    writer = SummaryWriter(log_dir=cfg.logging.log_dir)
    env = Env()
    collector = SyncDataCollector(env, policy, frames_per_batch=1000, total_frames=cfg.experiment.total_frames)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.algo.learning_rate)
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=critic,
        clip_epsilon=cfg.algo.clip_epsilon,
        entropy_coef=cfg.algo.entropy_coef
    )

    for batch in collector:
        for _ in range(cfg.algo.ppo_epochs):
            loss = loss_module(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_return = evaluate_policy(policy, env)
        writer.add_scalar("reward/mean", avg_return, collector.state["frames"])

    # Save models
    torch.save(policy.module.state_dict(), "models/ppo/policy.pt")
    torch.save(critic.module.state_dict(), "models/ppo/critic.pt")
    torch.save(optimizer.state_dict(), "models/ppo/optimizer.pt")

    # Generate GIF
    record_episode(env, policy)

if __name__ == "__main__":
    main()
