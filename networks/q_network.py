import torch
from environments.environments import Environment
from typing import List


class DuellingDQN(torch.nn.Module):
    def __init__(self, env: Environment, hidden_size: int):
        super().__init__()
        self.feature_map = None
        num_feature_maps = 1 + sum([env.has_goal, env.has_obstacle])
        self.combine_feature_map = torch.nn.Sequential(
            torch.nn.Linear(num_feature_maps * hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.value_map = torch.nn.Linear(hidden_size, 1)
        self.advantage_map = torch.nn.Linear(hidden_size, env.action_space.n)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        input_features = torch.cat([self.feature_map(x) for x in inputs], dim=-1)
        features = self.combine_feature_map(input_features)
        value = self.value_map(features)
        advantage = self.advantage_map(features)
        return value + advantage - advantage.mean(-1)


class QNetwork(DuellingDQN):
    def __init__(self, env: Environment, hidden_size: int):
        super().__init__(env, hidden_size)
        self.feature_map = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space["obs"].shape[0], hidden_size),
            torch.nn.ReLU(),
        )


class ConvQNetwork(DuellingDQN):
    def __init__(self, env: Environment, hidden_size: int):
        super().__init__(env, hidden_size)
        self.feature_map = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.ReLU(), torch.nn.Flatten()
        )  # todo: finish this definition
