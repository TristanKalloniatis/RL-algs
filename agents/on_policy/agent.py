import torch
from buffers.buffer import MultiEnvTransition
from typing import Dict, List, Any, Callable, Optional
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
        self.network = network.to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters())

    def collect_experience(self):
        self.buffer.flush()
        state = self.envs.reset()
        while not self.buffer.filled:
            with torch.no_grad():
                action = self.select_action(state, explore=True)
            next_state, reward, done, _ = self.envs.step(action)
            transition = MultiEnvTransition(
                state, action, reward, next_state, done, self.gamma
            )
            self.buffer.observe(transition)
            state = next_state

    def train_loop(self) -> Dict[str, List[float]]:
        raise NotImplementedError

    def train(self, episodes: int):
        write_log_flag = True
        next_eval_episode = 0
        while self.buffer.num_episodes < episodes:
            if self.buffer.num_episodes >= next_eval_episode and self.evaluate_episodes:
                self.evaluations[self.buffer.num_episodes] = self.evaluate(
                    self.evaluate_episodes, return_value=True
                )
                next_eval_episode += self.log_freq
            if self.buffer.num_episodes % self.log_freq == 0:
                if write_log_flag:
                    self.logger.write_log(
                        "Seen {0} episodes so far".format(self.buffer.num_episodes)
                    )
                    write_log_flag = False
            else:
                write_log_flag = True
            self.train_loop()
