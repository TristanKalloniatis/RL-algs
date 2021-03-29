import torch
from environments.environments import Environment
from typing import List, Union


class ActorCritic(torch.nn.Module):
    def __init__(self, env: Environment, hidden_size: int):
        super().__init__()
        self.feature_map = None
        num_inputs = 1 + sum([env.has_goal, env.has_obstacle])
        self.combine_feature_map = torch.nn.Sequential(
            torch.nn.Linear(num_inputs * hidden_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.value_map = torch.nn.Linear(hidden_size, 1)
        self.policy_logit_map = torch.nn.Linear(hidden_size, env.action_space.n)

    def forward(
        self, inputs: List[torch.Tensor]
    ) -> List[Union[torch.Tensor, torch.distributions.Categorical]]:
        input_features = torch.cat([self.feature_map(x) for x in inputs], dim=-1)
        features = self.combine_feature_map(input_features)
        value = self.value_map(features).squeeze()
        logits = self.policy_logit_map(features)
        policy = torch.distributions.Categorical(logits=logits)
        return [policy, value]


class ActorCriticNetwork(ActorCritic):
    def __init__(self, env: Environment, hidden_size: int):
        super().__init__(env, hidden_size)
        self.feature_map = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space["obs"].shape[0], hidden_size),
            torch.nn.ReLU(),
        )


class ConvActorCriticNetwork(ActorCritic):
    def __init__(
        self,
        env: Environment,
        hidden_size: int,
        kernel_size: int = 3,
        num_feature_maps: int = 2,
    ):
        assert kernel_size % 2 == 1, "For simplicity only support odd kernel sizes"
        pad_size = (kernel_size - 1) / 2
        super().__init__(env, hidden_size)
        self.feature_map = torch.nn.Sequential(
            torch.nn.Conv2d(1, num_feature_maps, kernel_size, padding=pad_size),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                num_feature_maps
                * env.observation_space["obs"].shape[0]
                * env.observation_space["obs"].shape[1],
                hidden_size,
            ),
            torch.nn.ReLU(),
        )
