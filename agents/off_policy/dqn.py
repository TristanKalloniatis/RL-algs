from agents.off_policy.agent import Agent
from typing import Dict, Any
from networks.q_network import QNetwork
import numpy as np


class DQN(Agent):
    def __init__(
        self,
        hidden_size: int,
        device_name: str,
        env_name: str,
        num_envs: int,
        env_fn,
        env_kwargs: Dict[str, Any],
        polyak_weight: float,
        gamma: float,
        capacity: int,
        batch_size: int,
        use_hindsight: bool,
        optim_steps: int,
    ):
        env = env_fn(env_name, **env_kwargs)
        network = QNetwork(env, hidden_size)
        super().__init__(
            network,
            device_name,
            env_name,
            num_envs,
            env_fn,
            env_kwargs,
            polyak_weight,
            gamma,
            capacity,
            batch_size,
            use_hindsight,
            optim_steps,
        )

    def select_action(self, state: Dict[str, np.ndarray]):
        pass

    def learn_on_batch(self) -> Dict[str, float]:
        pass
