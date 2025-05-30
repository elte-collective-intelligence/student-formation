import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

class ActorMlpWithProcessedScale(nn.Module):
    def __init__(self, mlp, action_dim):
        super().__init__()
        self.mlp = mlp
        self.action_dim = action_dim

    def forward(self, *features): # Expects concatenated features
        x = torch.cat(features, dim=-1)
        mixed_output = self.mlp(x)
        loc = mixed_output[..., :self.action_dim]
        raw_scale = mixed_output[..., self.action_dim:]
        scale = torch.nn.functional.softplus(raw_scale) + 1e-5 # ensure positivity and non-zero
        return loc, scale


def create_ppo_actor_critic(cfg, proof_environment):
    device = torch.device(cfg.base.device)
    env_cfg = cfg.env
    algo_cfg = cfg.algo
    
    # The action_spec for ProbabilisticActor should correspond to the action space it's modeling.
    # If the environment's action_spec is batched (e.g., [N_agents, ActionDim]),
    # and the policy network (MLP) is designed to output loc/scale for each of these N_agents,
    # then the full batched action_spec can be used.
    action_spec_for_policy = proof_environment.action_spec 
    
    # The MLP should output loc and scale for each action dimension for each agent.
    # If action_spec_for_policy.shape is [N, A], then single_agent_action_dim is A.
    # The MLP output will be [Batch (from collector), N, 2*A] if it processes all agents at once,
    # or [Batch (from collector) * N, 2*A] if agents are flattened into the batch.
    # Let's assume the td_module_actor_mlp will receive a batch of observations,
    # where each item in the batch corresponds to one agent's observation,
    # and it outputs loc/scale for that one agent.
    # So, the MLP out_features should be 2 * number_of_action_dimensions_per_agent.
    
    # Infer single_agent_action_dim from the potentially batched action_spec
    # proof_environment.action_spec.shape is (num_agents, action_dim_per_agent)
    single_agent_action_dim = proof_environment.action_spec.shape[-1]

    # --- Actor Network ---
    actor_input_features_concat = 0
    actor_mlp_in_keys_from_env = []
    # Obs specs are [N_agents, ObsDim_per_agent]. We need ObsDim_per_agent for MLP input.
    if "observation_self" in env_cfg.obs_keys:
        actor_input_features_concat += proof_environment.observation_spec["observation_self"].shape[-1]
        actor_mlp_in_keys_from_env.append("observation_self")
    if "observation_target_vector" in env_cfg.obs_keys:
        actor_input_features_concat += proof_environment.observation_spec["observation_target_vector"].shape[-1]
        actor_mlp_in_keys_from_env.append("observation_target_vector")

    actor_mlp_raw = MLP(
        in_features=actor_input_features_concat, # Input for a single agent's concatenated obs
        out_features=2 * single_agent_action_dim, # Output loc and scale for that agent
        num_cells=list(algo_cfg.hidden_dims),
        activation_class=nn.Tanh,
        device=device,
    )
    
    # Custom wrapper for MLP to output loc and processed scale
    actor_processed_mlp = ActorMlpWithProcessedScale(actor_mlp_raw, single_agent_action_dim)

    td_module_actor_mlp = TensorDictModule(
        module=actor_processed_mlp, # Use the new custom module
        in_keys=actor_mlp_in_keys_from_env, 
        out_keys=["loc", "scale"], # The custom module now directly outputs these
    )

    
    # ProbabilisticActor will use the provided 'spec' (action_spec_for_policy)
    # to potentially scale/transform the output of TanhNormal.
    # TanhNormal itself (without extra args) outputs in approx [-1, 1].
    actor_network = ProbabilisticActor(
        module=td_module_actor_mlp,
        spec=action_spec_for_policy, # Crucial: This tells the actor the target action space
        in_keys=["loc", "scale"], 
        distribution_class=TanhNormal,
        # No distribution_kwargs related to min/max/upscale.
        # Let TanhNormal use its defaults (outputs near [-1,1]) and let
        # ProbabilisticActor handle scaling based on 'spec'.
        distribution_kwargs={}, # Empty or remove if not needed
        return_log_prob=True,
        # By default, ProbabilisticActor applies necessary transforms (like ActionScale)
        # if the default range of the distribution (e.g. [-1,1] for TanhNormal)
        # does not match `spec`.
    )

    # --- Critic Network ---
    critic_input_features_concat = 0
    critic_mlp_in_keys_from_env = []
    if "observation_self" in env_cfg.obs_keys:
        critic_input_features_concat += proof_environment.observation_spec["observation_self"].shape[-1]
        critic_mlp_in_keys_from_env.append("observation_self")
    if "observation_target_vector" in env_cfg.obs_keys:
        critic_input_features_concat += proof_environment.observation_spec["observation_target_vector"].shape[-1]
        critic_mlp_in_keys_from_env.append("observation_target_vector")

    critic_mlp_raw = MLP(
        in_features=critic_input_features_concat,
        out_features=1,
        num_cells=list(algo_cfg.hidden_dims),
        activation_class=nn.Tanh,
        device=device,
    )
    
    value_network = ValueOperator(
        module=critic_mlp_raw,
        in_keys=critic_mlp_in_keys_from_env,
    )

    return actor_network, value_network