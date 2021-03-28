import torch
from environments.environments import Environment

class ActorCritic(torch.nn.Module):
    def __init__(self, env: Environment, hidden_size: int):
        super().__init__()

    def forward(self, inputs):
        pass