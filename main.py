import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import time
from pathlib import Path  # For model saving
import wandb  # For W&B integration

from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np  # For reward averaging in early stopping

from src.envs.env import FormationEnv
from src.agents.ppo_agent import create_ppo_actor_critic
from src.rollout.evaluator import evaluate_with_metrics


@hydra.main(
    version_base=None, config_path="configs", config_name="experiment/default_exp"
)
def main(cfg: DictConfig) -> None:
    # Initialize W&B
    run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}_{cfg.env.shape_type}"
    if cfg.env.shape_type == "circle":
        run_name += f"_r{cfg.env.circle.radius}"

    # When running Hydra MULTIRUN, save the Hydra sweep_id and job_num into W&B config from the Hydra runtime metadata.
    sweep_id = None
    sweep_job_num = None
    mode_value = HydraConfig.get().mode if HydraConfig.initialized() else None
    mode_name = (
        getattr(mode_value, "name", str(mode_value)).upper() if mode_value is not None else ""
    )
    if HydraConfig.initialized() and mode_name == "MULTIRUN":
        sweep_id = str(HydraConfig.get().sweep.dir)
        sweep_job_num = HydraConfig.get().job.num

    if sweep_job_num is not None:
        run_name += f"_job{sweep_job_num}"

    # Save the config to W&B
    wandb_config = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=True)
    if sweep_id is not None:
        wandb_config.setdefault("base", {})
        if isinstance(wandb_config["base"], dict):
            wandb_config["base"]["sweep_id"] = sweep_id
            wandb_config["base"]["sweep_job_num"] = sweep_job_num

    wandb.init(
        project=cfg.base.project_name + "-torchrl_formations",
        config=wandb_config,
        name=run_name,
        group=sweep_id,
        save_code=True,
    )

    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.base.device)
    torch.manual_seed(cfg.base.seed)
    if device == torch.device("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.base.seed)

    # Environment
    proof_env_instance = FormationEnv(cfg=cfg, device=device)
    proof_env_td = proof_env_instance.reset()
    print(f"Proof env batch_size after reset: {proof_env_instance.batch_size}")
    # print(f"Initial observation from proof_env:
    # {proof_env_td['observation']}") # For debugging

    # --- Quick Test of Rendering (Optional - uncomment to test) ---
    # if cfg.get("test_render_on_start", False):
    #     print("Testing rendering...")
    #     for _ in range(5): # Render a few frames of initial random state
    #         proof_env_instance.render(mode="human")
    #         time.sleep(0.1)
    #     # To render a few steps:
    #     # for _ in range(20):
    #     #     actions = proof_env_instance.action_spec.rand() # Random actions
    #     #     action_td = TensorDict({"action": actions}, batch_size=proof_env_instance.batch_size)
    #     #     proof_env_instance.step(action_td)
    #     #     proof_env_instance.render(mode="human")
    #     # proof_env_instance.close() # Close after test if you want to exit
    #     # print("Render test finished.")

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
        # Ensure trajectories don't exceed max_steps
        max_frames_per_traj=proof_env_instance.max_steps,
    )

    loss_module = ClipPPOLoss(
        actor=actor_network,
        critic=value_network,
        clip_epsilon=cfg.algo.clip_epsilon,
        entropy_coef=cfg.algo.entropy_coef,
        value_loss_coef=cfg.algo.value_loss_coef,
        # normalize_advantage=True, # Consider adding this if advantages are
        # unstable
    )
    loss_module = loss_module.to(device)

    adv_module = GAE(
        gamma=cfg.algo.gamma,
        lmbda=cfg.algo.gae_lambda,
        value_network=value_network,
        average_gae=True,
    )
    adv_module = adv_module.to(device)

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.algo.lr)

    tb_log_dir_path = Path(wandb.run.dir) / "tensorboard"
    tb_log_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Tensorboard logs will be saved to: {tb_log_dir_path}")
    writer = SummaryWriter(log_dir=str(tb_log_dir_path))

    pbar = tqdm.tqdm(total=cfg.algo.total_frames)
    collected_frames = 0

    recent_rewards_for_early_stop = []
    early_stop_patience = cfg.algo.get("early_stop_patience", 20)
    early_stop_reward_thresh = cfg.algo.get("early_stop_reward_threshold", None)

    for i, data_batch_from_collector in enumerate(collector):
        current_frames_collected_this_iter = data_batch_from_collector.numel()
        pbar.update(current_frames_collected_this_iter)
        collected_frames += current_frames_collected_this_iter

        data_batch_from_collector = data_batch_from_collector.to(device)
        batch_for_update = data_batch_from_collector.reshape(-1)

        with torch.no_grad():
            adv_module(batch_for_update)

        avg_actor_loss_iter = 0
        avg_critic_loss_iter = 0
        avg_entropy_loss_iter = 0
        avg_total_loss_iter = 0

        for ppo_epoch_num in range(cfg.algo.ppo_epochs):
            loss_td = loss_module(batch_for_update)

            actor_objective_loss = loss_td["loss_objective"]
            critic_loss = loss_td["loss_critic"]
            entropy_loss = loss_td["loss_entropy"]
            total_loss = actor_objective_loss + critic_loss + entropy_loss

            # Safety check for NaN or Inf losses
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(
                    f"WARNING: Skipped batch with NaN loss! Actor: {actor_objective_loss.item()}, Critic: {critic_loss.item()}"
                )
                continue  # Skip backward pass

            optimizer.zero_grad()
            total_loss.backward()
            # Optional: Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_norm=cfg.algo.get("max_grad_norm", 0.5)
            )
            optimizer.step()

            # Safety check for NaN gradients
            if torch.isnan(grad_norm):
                print("WARNING: Skipped update with NaN gradients!")
                continue

            avg_actor_loss_iter += actor_objective_loss.item()
            avg_critic_loss_iter += critic_loss.item()
            avg_entropy_loss_iter += entropy_loss.item()
            avg_total_loss_iter += total_loss.item()

        avg_actor_loss_iter /= cfg.algo.ppo_epochs
        avg_critic_loss_iter /= cfg.algo.ppo_epochs
        avg_entropy_loss_iter /= cfg.algo.ppo_epochs
        avg_total_loss_iter /= cfg.algo.ppo_epochs

        if i % cfg.log_interval == 0:
            mean_reward_in_batch = (
                data_batch_from_collector[("next", "reward")].mean().item()
            )

            log_payload = {
                "Loss/Actor_Objective": avg_actor_loss_iter,
                "Loss/Critic": avg_critic_loss_iter,
                "Loss/Entropy": avg_entropy_loss_iter,
                "Loss/Total": avg_total_loss_iter,
                "Reward/MeanRewardInBatch": mean_reward_in_batch,  # Log this mean
                "Progress/Iteration": i,
                # "LearningRate": optimizer.param_groups[0]['lr'], # If using LR scheduler
            }
            # TensorBoard
            for key, value in log_payload.items():
                if (
                    "Loss/" in key or "Reward/" in key
                ):  # Only log numericals to TB scalar
                    writer.add_scalar(key, value, collected_frames)

            wandb.log(log_payload, step=collected_frames)

            print(
                f"Iter {i}: Frames {collected_frames}, Mean Reward in Batch: {mean_reward_in_batch:.4f}, Total Loss: {avg_total_loss_iter:.4f}"
            )

            if early_stop_reward_thresh is not None:
                recent_rewards_for_early_stop.append(mean_reward_in_batch)
                if len(recent_rewards_for_early_stop) > early_stop_patience:
                    recent_rewards_for_early_stop.pop(0)

                if len(recent_rewards_for_early_stop) == early_stop_patience:
                    avg_recent_reward = np.mean(recent_rewards_for_early_stop)
                    wandb.log(
                        {"Reward/AvgRecentRewardForEarlyStop": avg_recent_reward},
                        step=collected_frames,
                    )
                    if avg_recent_reward >= early_stop_reward_thresh:
                        print(
                            f"Early stopping: Avg recent reward {avg_recent_reward:.4f} >= threshold {early_stop_reward_thresh}"
                        )
                        pbar.set_description(
                            f"Early stopping: Avg Reward {avg_recent_reward:.3f}"
                        )
                        break

        if collected_frames >= cfg.algo.total_frames:
            break

    pbar.close()
    collector.shutdown()
    writer.close()

    # Call evaluation with metrics
    print("Evaluate trained policy with formation metrics")

    try:
        aggregated_metrics = evaluate_with_metrics(
            proof_env_instance, actor_network, num_episodes=10, render=False
        )

        wandb.log(
            {
                "Evaluation/Boundary_Error_Mean": aggregated_metrics.get(
                    "boundary_error_mean", 0
                ),
                "Evaluation/Boundary_Error_Max": aggregated_metrics.get(
                    "boundary_error_max", 0
                ),
                "Evaluation/Agents_On_Boundary_Pct": str(
                    aggregated_metrics.get("agents_on_boundary_pct", 0)
                )
                + "%",
                "Evaluation/Uniformity_Mean": aggregated_metrics.get(
                    "uniformity_nn_distance_mean", 0
                ),
                "Evaluation/Uniformity_Std": aggregated_metrics.get(
                    "uniformity_nn_distance_std", 0
                ),
                "Evaluation/Uniformity_Coefficient": aggregated_metrics.get(
                    "uniformity_coefficient", 0
                ),
                "Evaluation/Collision_Count_Mean": str(
                    aggregated_metrics.get("collision_count", 0)
                )
                + " collisions",
                "Evaluation/Collision_Rate_Pct": aggregated_metrics.get(
                    "collision_rate_pct", 0
                ),
            }
        )

    except Exception as e:
        print(f"Evaluation with metrics failed: {e}")

    if proof_env_instance is not None:
        # Check if the render test might have already closed it and nulled pygame
        # This depends on how render test close is handled; for now, just call
        # close.
        try:
            proof_env_instance.close()
        except Exception as e:
            print(f"Error closing proof_env_instance: {e}")

    print("Training finished.")

    model_save_dir = Path(wandb.run.dir) / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    actor_path = model_save_dir / "actor_network.pt"
    critic_path = model_save_dir / "value_network.pt"

    torch.save(actor_network.state_dict(), actor_path)
    torch.save(value_network.state_dict(), critic_path)
    print(f"Models saved: {actor_path}, {critic_path}")

    model_artifact = wandb.Artifact(f"{wandb.run.name}-models", type="model")
    model_artifact.add_file(str(actor_path))
    model_artifact.add_file(str(critic_path))
    wandb.log_artifact(model_artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
