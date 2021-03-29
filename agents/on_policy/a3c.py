from agents.on_policy.agent import Agent
import torch
from networks.policy_network import ActorCriticNetwork
from typing import Dict, Any, List
from log_utils.log_utils import CustomLogger


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
        target_capacity: int,
        batch_size: int,
        use_hindsight: bool,
        optim_steps: int,
        loss_weight: float,
        loss_epsilon: float,
        logger: CustomLogger,
        log_freq: int,
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
            target_capacity,
            batch_size,
            use_hindsight,
            optim_steps,
            ["actor_loss", "critic_loss", "actor_entropy"],
            loss_weight,
            loss_epsilon,
            logger,
            log_freq,
        )
        self.loss_function = torch.nn.SmoothL1Loss()

    def train_loop(self):
        self.collect_experience()
        for _ in range(self.optim_steps):
            batch = self.buffer.sample()
            state = self.prepare_state_from_batch(batch)
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
                ).mean()
            )
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
