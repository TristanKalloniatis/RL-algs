from agents.on_policy.agent import Agent
import torch
from typing import Dict, Any, Optional
from log_utils.log_utils import CustomLogger


class ProximalPolicyOptimisation(Agent):
    def __init__(
        self,
        network,  # todo: construct network in init
        device_name: str,
        env_name: str,
        num_envs: int,
        env_fn,
        env_kwargs: Dict[str, Any],
        gamma: float,
        capacity: int,
        batch_size: int,
        use_hindsight: bool,
        optim_steps: int,
        loss_weight: float,
        loss_epsilon: float,
        logger: CustomLogger,
        log_freq: int,
        evaluate_episodes: Optional[int],
    ):
        super().__init__(
            network,
            device_name,
            env_name,
            num_envs,
            env_fn,
            env_kwargs,
            gamma,
            capacity,
            batch_size,
            use_hindsight,
            optim_steps,
            ["actor_loss", "critic_loss", "actor_entropy"],
            loss_weight,
            loss_epsilon,
            logger,
            log_freq,
            evaluate_episodes,
        )
        self.loss_function = torch.nn.SmoothL1Loss()
