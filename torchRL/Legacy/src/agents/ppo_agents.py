import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4, 128),  # 4 agents * obs_dim=4
            nn.ReLU(),
            nn.Linear(128, 4 * 2)   # 4 agents * action_dim=2
        )

    def forward(self, x):
        return self.net(x).view(4, 2)

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

policy = TensorDictModule(
    module=PolicyNetwork(),
    in_keys=["obs"],
    out_keys=["action"]
)

critic = ValueOperator(
    module=CriticNetwork(),
    in_keys=["obs"]
)
