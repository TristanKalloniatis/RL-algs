from agents.off_policy.agent import Agent
from typing import Dict, Any


class SoftActorCritic(Agent):
    def __init__(
        self,
        network,
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
            polyak_weight,
            gamma,
            capacity,
            batch_size,
            use_hindsight,
            optim_steps,
            ["value_loss", "q_loss", "policy_loss"],
            loss_weight,
            loss_epsilon,
        )
