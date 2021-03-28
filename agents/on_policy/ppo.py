from agents.on_policy.agent import Agent
import torch
from typing import Dict, Any, List


class ProximalPolicyOptimisation(Agent):
    def __init__(
        self,
        network, # todo: construct network in init
        device_name: str,
        env_name: str,
        num_envs: int,
        env_fn,
        env_kwargs: Dict[str, Any],
        gamma: float,
        target_capacity: int,
        batch_size: int,
        use_hindsight: bool,
        optim_steps: int,
        loss_weight: float,
        loss_epsilon: float,
    ):
        super().__init__(
            network,
            device_name,
            env_name,
            num_envs,
            env_fn,
            env_kwargs,
            gamma,
            target_capacity,
            batch_size,
            use_hindsight,
            optim_steps,
            ["actor_loss", "critic_loss", "actor_entropy"],
            loss_weight,
            loss_epsilon,
        )
        self.loss_function = torch.nn.SmoothL1Loss()
