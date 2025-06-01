import torch
import torch.nn as nn
import torch.nn.functional as F # For F.softplus
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

# Custom module to process MLP output into loc and positive scale
class ActorMlpWithProcessedScale(nn.Module):
    def __init__(self, mlp_module: nn.Module, action_dim: int):
        super().__init__()
        self.mlp_module = mlp_module
        self.action_dim = action_dim

    def forward(self, concatenated_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed_output = self.mlp_module(concatenated_features)
        
        loc = mixed_output[..., :self.action_dim]
        raw_scale_params = mixed_output[..., self.action_dim:]
        scale = F.softplus(raw_scale_params) + 1e-5 # Ensure scale is positive and non-zero
        return loc, scale

def create_ppo_actor_critic(cfg, proof_environment):
    device = torch.device(cfg.base.device)
    env_cfg = cfg.env # cfg.env (e.g., from formation.yaml)
    algo_cfg = cfg.algo # cfg.algo (e.g., from ppo.yaml)

    action_spec_for_policy = proof_environment.action_spec 
    
    # The MLP should output loc and scale for each action dimension for each agent.
    # single_agent_action_dim is the number of action dimensions for one agent.
    single_agent_action_dim = proof_environment.action_spec.shape[-1]

    # --- Actor Network ---
    actor_input_features_concat = 0
    actor_tdm_in_keys = [] 
    

    for obs_key in env_cfg.obs_keys_for_actor:
        actor_input_features_concat += proof_environment.observation_spec[obs_key].shape[-1]
        actor_tdm_in_keys.append(obs_key)

    # This is the core MLP that outputs parameters for loc and scale
    core_actor_mlp = MLP(
        in_features=actor_input_features_concat, 
        out_features=2 * single_agent_action_dim, 
        num_cells=list(algo_cfg.hidden_dims),
        activation_class=nn.Tanh,
        device=device,
    )

    actor_processed_mlp_module = ActorMlpWithProcessedScale(core_actor_mlp, single_agent_action_dim)

    td_module_for_actor = TensorDictModule(
        module=actor_processed_mlp_module,
        in_keys=actor_tdm_in_keys, 
        out_keys=["loc", "scale"], 
    )
    
    actor_network = ProbabilisticActor(
        module=td_module_for_actor, # This module now outputs "loc" and "scale" into the tensordict
        spec=action_spec_for_policy, # The overall action spec this policy is trying to match
        in_keys=["loc", "scale"], # ProbabilisticActor reads these from td_module_for_actor's output TD
        distribution_class=TanhNormal,
        distribution_kwargs={}, # TanhNormal defaults to outputting in ~[-1,1], spec handles final range
        return_log_prob=True,
    )

    # --- Critic Network ---
    critic_input_features_concat = 0
    critic_tdm_in_keys = []
    critic_obs_keys_list = env_cfg.get("obs_keys_for_critic", env_cfg.obs_keys_for_actor) # Default to actor's keys

    for obs_key in critic_obs_keys_list:
        critic_input_features_concat += proof_environment.observation_spec[obs_key].shape[-1]
        critic_tdm_in_keys.append(obs_key)

    # Core MLP for the critic
    critic_mlp_raw = MLP(
        in_features=critic_input_features_concat,
        out_features=1, # Critic outputs a single value
        num_cells=list(algo_cfg.hidden_dims),
        activation_class=nn.Tanh,
        device=device,
    )
    
    # It will take `critic_tdm_in_keys` from input TD, concatenate them,
    # pass to `critic_mlp_raw.forward()`, and use the output as the state value.
    value_network = ValueOperator(
        module=critic_mlp_raw,
        in_keys=critic_tdm_in_keys,
        # Default out_key is "state_value", which is what GAE and ClipPPOLoss expect.
    )

    return actor_network, value_network