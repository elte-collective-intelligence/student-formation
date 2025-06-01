import torch
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import CompositeSpec
from torchrl.data.tensor_specs import BoundedTensorSpec

class Env(EnvBase):
    def __init__(self):
        super().__init__()
        self.num_agents = 4
        self.obs_dim = 4  # e.g., (x, y, dx_to_center, nearest_agent_dist)
        self.action_dim = 2  # e.g., (dx, dy)

        self._observation_spec = CompositeSpec({
            "obs": BoundedTensorSpec(
                shape=(self.num_agents, self.obs_dim),
                dtype=torch.float32,
                minimum=-1.0,
                maximum=1.0
            )
        })

        self._action_spec = CompositeSpec({
            "action": BoundedTensorSpec(
                shape=(self.num_agents, self.action_dim),
                dtype=torch.float32,
                minimum=-1.0,
                maximum=1.0
            )
        })

        self._reward_spec = CompositeSpec({
            "reward": BoundedTensorSpec(shape=(1,), dtype=torch.float32, minimum=-10.0, maximum=10.0)
        })

        self._done_spec = CompositeSpec({
            "done": BoundedTensorSpec(shape=(1,), dtype=torch.bool)
        })

        self.positions = None
        self.steps = 0

    def reset(self) -> TensorDict:
        self.positions = torch.rand(self.num_agents, 2) * 2 - 1
        self.steps = 0
        return TensorDict({"obs": self._get_obs()}, batch_size=[])

    def step(self, actions: TensorDict) -> TensorDict:
        action = actions["action"]
        self.positions += action.clamp(-0.1, 0.1)  # movement

        reward = -torch.var(self.positions.norm(dim=1), unbiased=False).unsqueeze(0)
        self.steps += 1
        done = torch.tensor([self.steps > 100])

        return TensorDict({
            "obs": self._get_obs(),
            "reward": reward,
            "done": done
        }, batch_size=[])

    def _get_obs(self):
        center = self.positions.mean(dim=0)
        dists = ((self.positions.unsqueeze(1) - self.positions.unsqueeze(0)).norm(dim=2) + 1e-5)
        nearest = dists.topk(2, largest=False).values[:, 1]
        rel = torch.cat([
            self.positions,
            (center - self.positions),
            nearest.unsqueeze(1)
        ], dim=1)
        return rel

    @property
    def observation_spec(self): return self._observation_spec
    @property
    def action_spec(self): return self._action_spec
    @property
    def reward_spec(self): return self._reward_spec
    @property
    def done_spec(self): return self._done_spec
