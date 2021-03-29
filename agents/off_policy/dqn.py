from agents.off_policy.agent import Agent
from typing import Dict, Any, Callable, Optional
from networks.q_network import QNetwork
import numpy as np
from samplers import samplers
from environments.environments import Environment
import torch
from log_utils.log_utils import CustomLogger


class DeepQualityNetwork(Agent):
    def __init__(
        self,
        hidden_size: int,
        sampler: samplers.QPolicySampler,
        device_name: str,
        env_name: str,
        num_envs: int,
        env_fn: Callable[[str, Dict[str, Any]], Environment],
        env_kwargs: Dict[str, Any],
        polyak_weight: float,
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
            ["q_loss"],
            loss_weight,
            loss_epsilon,
            logger,
            log_freq,
            evaluate_episodes,
        )
        self.sampler = sampler
        self.loss_function = torch.nn.SmoothL1Loss()

    def select_action(self, state: Dict[str, np.ndarray], explore: bool) -> np.ndarray:
        state = self.prepare_state(state)
        q = self.network(state)
        return (
            self.sampler.sample(q, self.buffer.num_episodes)
            if explore
            else samplers.GreedySampler().sample(q, self.buffer.num_episodes)
        )

    def learn_on_batch(self):
        batch = self.sample()
        state = self.prepare_state_from_batch(batch)
        next_state = self.prepare_state_from_batch(batch, use_next=True)
        q = self.network(state, use_online=True)
        with torch.no_grad():
            next_q = self.network(next_state, use_online=False)
        target = batch["reward"] + self.gamma * torch.where(
            batch["done"], torch.zeros_like(batch["reward"]), next_q.max(-1)[0]
        )
        value = q.gather(1, batch["action"].unsqueeze(-1)).squeeze(-1)
        q_loss = self.loss_function(value, target)
        loss = self.loss_manager.loss({"q_loss": q_loss})
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
