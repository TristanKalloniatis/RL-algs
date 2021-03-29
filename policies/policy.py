import torch
import numpy as np
from typing import Callable


class QPolicySampler:
    def sample(self, q_values: torch.Tensor, num_episodes: int) -> np.ndarray:
        raise NotImplementedError


class GreedySampler(QPolicySampler):
    def sample(self, q_values: torch.Tensor, num_episodes: int) -> np.ndarray:
        return q_values.argmax(-1).cpu().numpy()


class EpsilonGreedySampler(QPolicySampler):
    def __init__(self, epsilon_anneal_fn: Callable[[int], float]):
        self.epsilon_anneal_fn = epsilon_anneal_fn

    def sample(self, q_values: torch.Tensor, num_episodes: int) -> np.ndarray:
        epsilon = self.epsilon_anneal_fn(num_episodes)
        rand = np.random.rand(q_values.shape[0])
        best_action = q_values.argmax(-1).cpu().numpy()
        rand_action = np.random.choice(q_values.shape[1], size=(q_values.shape[0]))
        return np.where(rand < epsilon, rand_action, best_action)


class BoltzmanSampler(QPolicySampler):
    def __init__(self, temperature_anneal_fn: Callable[[int], float]):
        self.temperature_anneal_fn = temperature_anneal_fn

    def sample(self, q_values: torch.Tensor, num_episodes: int) -> np.ndarray:
        temperature = self.temperature_anneal_fn(num_episodes)
        dist = torch.distributions.Categorical(logits=q_values / temperature)
        return dist.sample()
