from agents.on_policy.agent import Agent
import torch
from networks.policy_network import ActorCriticNetwork
from typing import Dict, Any, Optional
from log_utils.custom_logger import CustomLogger
import numpy as np


class AdvantageActorCritic(Agent):
    def __init__(
        self,
        hidden_size: int,
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
        env = env_fn(env_name, **env_kwargs)
        network = ActorCriticNetwork(env, hidden_size)
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

    def select_action(self, state: Dict[str, np.ndarray], explore: bool):
        state = self.prepare_state(state)
        policy, _ = self.network(state)
        action = policy.sample() if explore else policy.logits.argmax(-1)
        return action.cpu().numpy()

    def train_loop(self):
        self.collect_experience()
        for _ in range(self.optim_steps):
            batch = self.buffer.sample()
            state = self.prepare_state_from_batch(batch, use_next=False)
            policy, value = self.network(state)
            log_prob = policy.log_prob(batch["action"])
            actor_entropy = policy.entropy().mean()
            advantage = batch["episode_return"] - value.detach()
            critic_loss = self.loss_function(value, batch["episode_return"])
            actor_loss = (
                -log_prob
                * advantage
                * torch.pow(
                    torch.tensor(self.gamma, device=self.device), batch["timestep"]
                )
            ).mean()
            loss = self.loss_manager.loss(
                {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "actor_entropy": actor_entropy,
                }
            )
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
