import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import time
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.utils.tensorboard import SummaryWriter
import tqdm
# from pathlib import Path # For model saving example

from src.envs.env import FormationEnv
from src.agents.ppo_agent import create_ppo_actor_critic

@hydra.main(version_base=None, config_path="configs", config_name="experiment/default_exp") # Adjusted path assuming execution from project root
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.base.device)
    torch.manual_seed(cfg.base.seed)
    if device == torch.device("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.base.seed)

    proof_env_instance = FormationEnv(cfg=cfg, device=device)
    # Resetting proof_env_instance here is mainly for checking batch_size and specs.
    # The collector will create and manage its own env instances.
    proof_env_instance.reset() 
    print(f"Proof env batch_size after reset: {proof_env_instance.batch_size}")

    def create_env_fn_for_collector():
        return FormationEnv(cfg=cfg, device=device)

    actor_network, value_network = create_ppo_actor_critic(cfg, proof_env_instance)
    actor_network = actor_network.to(device)
    value_network = value_network.to(device)

    collector = SyncDataCollector(
        create_env_fn=create_env_fn_for_collector,
        policy=actor_network,
        frames_per_batch=cfg.algo.frames_per_batch,
        total_frames=cfg.algo.total_frames,
        device=device,
        # max_frames_per_traj can be set if you want to limit trajectory length,
        # otherwise, it uses env.max_steps or runs until done.
        # max_frames_per_traj=proof_env_instance.max_steps 
    )

    loss_module = ClipPPOLoss(
        actor=actor_network,
        critic=value_network,
        clip_epsilon=cfg.algo.clip_epsilon,
        entropy_coef=cfg.algo.entropy_coef,
        value_loss_coef=cfg.algo.value_loss_coef,
    )
    loss_module = loss_module.to(device)

    adv_module = GAE(
        gamma=cfg.algo.gamma,
        lmbda=cfg.algo.gae_lambda,
        value_network=value_network,
        average_gae=True 
    )
    adv_module = adv_module.to(device)

    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.algo.lr
    )

    log_dir_path = f"logs/{cfg.base.project_name}/{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Tensorboard logs will be saved to: {log_dir_path}")
    writer = SummaryWriter(log_dir=log_dir_path)
    
    pbar = tqdm.tqdm(total=cfg.algo.total_frames)
    collected_frames = 0

    for i, data_batch_from_collector in enumerate(collector):
        current_frames_collected_this_iter = data_batch_from_collector.numel()
        pbar.update(current_frames_collected_this_iter)
        collected_frames += current_frames_collected_this_iter
        
        data_batch_from_collector = data_batch_from_collector.to(device)
        # `data_batch_from_collector` has leading dims [Time, N_agents]
        # Reshape to [Time * N_agents, Features] for batch processing by loss/GAE
        batch_for_update = data_batch_from_collector.reshape(-1)

        # GAE computation needs next_observation and next_done, which are in
        # data_batch_from_collector[("next", "observation_...")]
        # data_batch_from_collector[("next", "done")]
        # The adv_module will look for these ("next", key) entries.
        with torch.no_grad():
            adv_module(batch_for_update) # Modifies batch_for_update in-place with "advantage" & "value_target"

        for ppo_epoch_num in range(cfg.algo.ppo_epochs):
            loss_td = loss_module(batch_for_update) 
            
            actor_objective_loss = loss_td["loss_objective"]
            critic_loss = loss_td["loss_critic"]
            entropy_loss = loss_td["loss_entropy"]
            total_loss = actor_objective_loss + critic_loss + entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if i % cfg.log_interval == 0:
            writer.add_scalar("Loss/Actor_Objective", actor_objective_loss.item(), collected_frames)
            writer.add_scalar("Loss/Critic", critic_loss.item(), collected_frames)
            writer.add_scalar("Loss/Entropy", entropy_loss.item(), collected_frames)
            writer.add_scalar("Loss/Total", total_loss.item(), collected_frames)
            
            # Reward is associated with the *next* state in the collected batch
            # data_batch_from_collector has shape [T, N_agents, ...]
            # data_batch_from_collector[("next", "reward")] will have shape [T, N_agents, 1]
            mean_batch_reward_per_agent_step = data_batch_from_collector[("next", "reward")].mean().item() # CORRECTED KEY
            writer.add_scalar("Reward/MeanBatchRewardPerAgentStep", mean_batch_reward_per_agent_step, collected_frames)
            
            print(f"Iter {i}: Frames {collected_frames}, Mean Batch Reward: {mean_batch_reward_per_agent_step:.3f}, Total Loss: {total_loss.item():.3f}")

        if collected_frames >= cfg.algo.total_frames:
            break
            
    pbar.close()
    collector.shutdown()
    writer.close()
    proof_env_instance.close()
    print("Training finished.")

    # Example of saving models:
    # from pathlib import Path # Add to imports
    # model_save_dir = Path(log_dir_path) / "models"
    # model_save_dir.mkdir(parents=True, exist_ok=True)
    # torch.save(actor_network.state_dict(), model_save_dir / "actor_network.pt")
    # torch.save(value_network.state_dict(), model_save_dir / "value_network.pt")
    # print(f"Models saved to {model_save_dir}")

if __name__ == "__main__":
    main()