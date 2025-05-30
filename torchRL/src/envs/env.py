import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
# Use new spec names, but keep DiscreteTensorSpec for now
from torchrl.data.tensor_specs import (
    UnboundedContinuousTensorSpec as Unbounded,
    CompositeSpec as Composite,
    BoundedTensorSpec as Bounded,
    DiscreteTensorSpec, # Reverted: Keeping DiscreteTensorSpec for done
)
import math

class FormationEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, cfg, device="cpu"):
        super().__init__(device=device, batch_size=torch.Size([cfg.env.num_agents]))

        self.num_agents = cfg.env.num_agents
        self.arena_size = cfg.env.arena_size
        self.max_steps = cfg.env.max_steps
        self.target_radius = cfg.env.target_radius
        self.current_step = 0

        self._make_specs()
        self.actor_obs_keys = cfg.env.obs_keys

    def _make_specs(self) -> None:
        self.observation_spec = Composite(
            {
                "observation_self": Unbounded(
                    shape=(self.num_agents, 2),
                    device=self.device,
                ),
                "observation_target_vector": Unbounded(
                    shape=(self.num_agents, 2),
                    device=self.device,
                ),
            },
            shape=torch.Size([self.num_agents])
        )

        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, 2),
            device=self.device,
            dtype=torch.float32
        )

        self.reward_spec_unbatched = Unbounded(
            shape=(1,),
            device=self.device
        )

        self.done_spec_unbatched = DiscreteTensorSpec( # Reverted to DiscreteTensorSpec
            n=2,
            shape=torch.Size([1]),
            dtype=torch.bool,
            device=self.device
        )

    def _reset(self, tensordict: TensorDictBase = None) -> TensorDictBase:
        self.current_step = 0
        agent_positions = (torch.rand(self.num_agents, 2, device=self.device) - 0.5) * self.arena_size
        target_positions = torch.zeros(self.num_agents, 2, device=self.device)
        for i in range(self.num_agents):
            angle = 2 * math.pi * i / self.num_agents
            target_positions[i, 0] = self.target_radius * math.cos(angle)
            target_positions[i, 1] = self.target_radius * math.sin(angle)
        self.agent_positions = agent_positions
        self.target_positions = target_positions

        obs_self = self.agent_positions.clone()
        obs_target_vector = self.target_positions - self.agent_positions
        done_val_for_each_agent = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device)

        td_out = TensorDict(
            {
                "observation_self": obs_self,
                "observation_target_vector": obs_target_vector,
                "done": done_val_for_each_agent,
            },
            batch_size=torch.Size([self.num_agents]),
            device=self.device
        )
        return td_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.current_step += 1
        actions = tensordict["action"]
        self.agent_positions += actions * 0.1
        self.agent_positions = torch.clamp(self.agent_positions, -self.arena_size / 2, self.arena_size / 2)

        distances_to_target = torch.norm(self.agent_positions - self.target_positions, dim=1, keepdim=True)
        reward_val_for_each_agent = -distances_to_target

        is_episode_done = self.current_step >= self.max_steps
        done_val_for_each_agent = torch.full((self.num_agents, 1), is_episode_done, dtype=torch.bool, device=self.device)

        obs_self = self.agent_positions.clone()
        obs_target_vector = self.target_positions - self.agent_positions

        td_out = TensorDict(
            {
                "observation_self": obs_self,
                "observation_target_vector": obs_target_vector,
                "reward": reward_val_for_each_agent,
                "done": done_val_for_each_agent,
            },
            batch_size=torch.Size([self.num_agents]),
            device=self.device
        )
        return td_out

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)