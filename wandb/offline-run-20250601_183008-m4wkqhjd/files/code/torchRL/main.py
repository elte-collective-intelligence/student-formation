import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import time
from pathlib import Path # For model saving
import wandb # For W&B integration

from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.utils.tensorboard import SummaryWriter
import tqdm

from src.envs.env import FormationEnv # Assuming this is our translated TorchRL env
from src.agents.ppo_agent import create_ppo_actor_critic

@hydra.main(version_base=None, config_path="configs", config_name="experiment/default_exp")
def main(cfg: DictConfig) -> None:
    # Initialize W&B
    wandb.init(
        project=cfg.base.project_name + "-torchrl", # Distinguish from SB3 project
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        # sync_tensorboard=True, # Can auto-sync tensorboard logs
        name=f"run_{time.strftime('%Y%m%d-%H%M%S')}", # Optional: custom run name
        save_code=True # Save main script to W&B
    )

    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.base.device)
    torch.manual_seed(cfg.base.seed)
    if device == torch.device("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.base.seed)

    # Environment
    # Create one instance for spec checking, agent creation
    proof_env_instance = FormationEnv(cfg=cfg, device=device)
    proof_env_instance.reset() 
    print(f"Proof env batch_size after reset: {proof_env_instance.batch_size}")

    # Factory function for collector
    def create_env_fn_for_collector():
        return FormationEnv(cfg=cfg, device=device)

    # Agent
    actor_network, value_network = create_ppo_actor_critic(cfg, proof_env_instance)
    actor_network = actor_network.to(device)
    value_network = value_network.to(device)
    # W&B: Watch models (optional, can be verbose)
    # wandb.watch(actor_network, log="all", log_freq=100)
    # wandb.watch(value_network, log="all", log_freq=100)


    # Collector
    collector = SyncDataCollector(
        create_env_fn=create_env_fn_for_collector,
        policy=actor_network,
        frames_per_batch=cfg.algo.frames_per_batch,
        total_frames=cfg.algo.total_frames,
        device=device,
    )

    # Loss Module
    loss_module = ClipPPOLoss(
        actor=actor_network,
        critic=value_network,
        clip_epsilon=cfg.algo.clip_epsilon,
        entropy_coef=cfg.algo.entropy_coef,
        value_loss_coef=cfg.algo.value_loss_coef,
    )
    loss_module = loss_module.to(device)

    # Advantage Module
    adv_module = GAE(
        gamma=cfg.algo.gamma,
        lmbda=cfg.algo.gae_lambda,
        value_network=value_network,
        average_gae=True 
    )
    adv_module = adv_module.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.algo.lr
    )

    # TensorBoard Logging
    # W&B can sync this if sync_tensorboard=True, or we can log directly to W&B
    tb_log_dir_path = Path(wandb.run.dir) / "tensorboard" # Save TB logs within W&B run dir
    # tb_log_dir_path = f"logs/{cfg.base.project_name}_tb/{time.strftime('%Y%m%d-%H%M%S')}" # Alternative
    tb_log_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Tensorboard logs will be saved to: {tb_log_dir_path}")
    writer = SummaryWriter(log_dir=str(tb_log_dir_path))
    
    pbar = tqdm.tqdm(total=cfg.algo.total_frames)
    collected_frames = 0
    # For more stable early stopping reward tracking
    recent_rewards = []
    early_stop_patience = cfg.algo.get("early_stop_patience", 10) # Number of intervals to average over

    for i, data_batch_from_collector in enumerate(collector):
        current_frames_collected_this_iter = data_batch_from_collector.numel()
        pbar.update(current_frames_collected_this_iter)
        collected_frames += current_frames_collected_this_iter
        
        data_batch_from_collector = data_batch_from_collector.to(device)
        batch_for_update = data_batch_from_collector.reshape(-1)

        with torch.no_grad():
            adv_module(batch_for_update)

        avg_actor_loss_epoch = 0
        avg_critic_loss_epoch = 0
        avg_entropy_loss_epoch = 0
        avg_total_loss_epoch = 0

        for ppo_epoch_num in range(cfg.algo.ppo_epochs):
            loss_td = loss_module(batch_for_update) 
            
            actor_objective_loss = loss_td["loss_objective"]
            critic_loss = loss_td["loss_critic"]
            entropy_loss = loss_td["loss_entropy"]
            total_loss = actor_objective_loss + critic_loss + entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            avg_actor_loss_epoch += actor_objective_loss.item()
            avg_critic_loss_epoch += critic_loss.item()
            avg_entropy_loss_epoch += entropy_loss.item()
            avg_total_loss_epoch += total_loss.item()
        
        avg_actor_loss_epoch /= cfg.algo.ppo_epochs
        avg_critic_loss_epoch /= cfg.algo.ppo_epochs
        avg_entropy_loss_epoch /= cfg.algo.ppo_epochs
        avg_total_loss_epoch /= cfg.algo.ppo_epochs


        # Logging
        if i % cfg.log_interval == 0:
            mean_batch_reward_per_agent_step = data_batch_from_collector[("next", "reward")].mean().item()
            
            log_payload = {
                "Loss/Actor_Objective": avg_actor_loss_epoch,
                "Loss/Critic": avg_critic_loss_epoch,
                "Loss/Entropy": avg_entropy_loss_epoch,
                "Loss/Total": avg_total_loss_epoch,
                "Reward/MeanBatchRewardPerAgentStep": mean_batch_reward_per_agent_step,
                "Progress/Iteration": i,
            }
            # TensorBoard
            writer.add_scalar("Loss/Actor_Objective", avg_actor_loss_epoch, collected_frames)
            writer.add_scalar("Loss/Critic", avg_critic_loss_epoch, collected_frames)
            writer.add_scalar("Loss/Entropy", avg_entropy_loss_epoch, collected_frames)
            writer.add_scalar("Loss/Total", avg_total_loss_epoch, collected_frames)
            writer.add_scalar("Reward/MeanBatchRewardPerAgentStep", mean_batch_reward_per_agent_step, collected_frames)
            
            # W&B
            wandb.log(log_payload, step=collected_frames) # Use collected_frames as the global step for W&B

            print(f"Iter {i}: Frames {collected_frames}, Mean Reward: {mean_batch_reward_per_agent_step:.3f}, Total Loss: {avg_total_loss_epoch:.3f}")

            # Early Stopping Logic
            early_stop_reward_thresh = cfg.algo.get("early_stop_reward_threshold")
            if early_stop_reward_thresh is not None:
                recent_rewards.append(mean_batch_reward_per_agent_step)
                if len(recent_rewards) > early_stop_patience:
                    recent_rewards.pop(0)
                if len(recent_rewards) == early_stop_patience and \
                   (sum(recent_rewards) / early_stop_patience) >= early_stop_reward_thresh:
                    print(f"Early stopping: Avg reward {sum(recent_rewards)/early_stop_patience:.3f} >= threshold {early_stop_reward_thresh}")
                    break 

        if collected_frames >= cfg.algo.total_frames:
            break
            
    pbar.close()
    collector.shutdown()
    writer.close()
    proof_env_instance.close() # Close the proof env instance
    
    print("Training finished.")

    # Save models to W&B run directory (or a configured path)
    model_save_dir = Path(wandb.run.dir) / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    actor_path = model_save_dir / "actor_network.pt"
    critic_path = model_save_dir / "value_network.pt"
    
    torch.save(actor_network.state_dict(), actor_path)
    torch.save(value_network.state_dict(), critic_path)
    print(f"Models saved: {actor_path}, {critic_path}")

    # Log models as W&B Artifacts (optional)
    # model_artifact = wandb.Artifact(f'{wandb.run.name}-models', type='model')
    # model_artifact.add_file(str(actor_path))
    # model_artifact.add_file(str(critic_path))
    # wandb.log_artifact(model_artifact)

    wandb.finish()


if __name__ == "__main__":
    main()