from buffers.buffer import MultiEnvTransition
from networks.off_policy import OffPolicyNetworkFactory
import torch
from typing import Optional, Dict, Any, List, Callable
from environments.environments import Environment
from agents import agent
from log_utils.log_utils import CustomLogger


class Agent(agent.Agent):
    def __init__(
        self,
        network: torch.nn.Module,
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
        loss_names: List[str],
        loss_weight: float,
        loss_epsilon: float,
        logger: CustomLogger,
        log_freq: int,
        evaluate_episodes: Optional[int],
    ):
        super().__init__(
            env_name,
            num_envs,
            env_fn,
            env_kwargs,
            device_name,
            logger,
            evaluate_episodes,
            gamma,
            capacity,
            batch_size,
            use_hindsight,
            optim_steps,
            loss_names,
            loss_weight,
            loss_epsilon,
            log_freq,
        )
        self.network = OffPolicyNetworkFactory(network.to(self.device), polyak_weight)
        self.network.synchronise(use_polyak=False)

        self.optim = torch.optim.Adam(self.network.online_network.parameters())

    def sample(self) -> Dict[str, Optional[torch.Tensor]]:
        return self.buffer.sample()

    def train_on_batch(self):
        self.learn_on_batch()
        self.network.synchronise()

    def train(self, episodes: int):
        state = self.envs.reset()
        write_log_flag = True
        while self.buffer.num_episodes < episodes:
            if self.buffer.num_episodes % self.log_freq == 0:
                if write_log_flag:
                    self.logger.write_log(
                        "Seen {0} episodes so far".format(self.buffer.num_episodes)
                    )
                    if self.evaluate_episodes:
                        self.evaluations.append(
                            self.evaluate(self.evaluate_episodes, return_value=True)
                        )
                    write_log_flag = False
            else:
                write_log_flag = True
            with torch.no_grad():
                action = self.select_action(state, explore=True)
            next_state, reward, done, _ = self.envs.step(action)
            transition = MultiEnvTransition(
                state, action, reward, next_state, done, self.gamma
            )
            self.buffer.observe(transition)
            state = next_state
            if self.buffer.ready_to_sample:
                for _ in range(self.optim_steps):
                    self.train_on_batch()
