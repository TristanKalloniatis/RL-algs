from typing import Dict, List, Optional, Callable, Any, Union
from common.multiprocessing_env import SubprocVecEnv
from environments.environments import Environment
import numpy as np
import torch
from losses.multi_loss import LossManager
from environments.env_constructor import make_envs
from log_utils.log_utils import CustomLogger
from buffers.buffer import MultiEnvBuffer
import matplotlib.pyplot as plt
import os

if not os.path.exists("plots/"):
    os.makedirs("plots/losses")
    os.makedirs("plots/evaluations")


class Agent:
    def __init__(
        self,
        env_name: str,
        num_envs: int,
        env_fn: Callable[[str, Dict[str, Any]], Environment],
        env_kwargs: Dict[str, Any],
        device_name: str,
        logger: CustomLogger,
        evaluate_episodes: Optional[int],
        gamma: float,
        capacity: int,
        batch_size: int,
        use_hindsight: bool,
        optim_steps: int,
        loss_names: List[str],
        loss_weight: float,
        loss_epsilon: float,
        log_freq: int,
    ):

        self.env = env_fn(env_name, **env_kwargs)
        self.envs = SubprocVecEnv(
            [make_envs(env_name, env_fn, env_kwargs) for _ in range(num_envs)]
        )
        self.use_hindsight = use_hindsight if self.env.has_goal else False
        self.buffer = MultiEnvBuffer(
            num_envs,
            capacity,
            batch_size,
            self.use_hindsight,
            self.env.has_goal,
            self.env.has_obstacle,
            gamma,
            device_name,
        )
        self.gamma = gamma
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.evaluations: Dict[int, List[float]] = {}
        self.evaluate_episodes = evaluate_episodes
        self.optim_steps = optim_steps
        self.loss_manager = LossManager(loss_names, loss_weight, loss_epsilon)
        self.log_freq = log_freq

    def prepare_state(
        self, state: Dict[str, np.ndarray], use_next: bool = False
    ) -> List[torch.Tensor]:
        obs = torch.tensor(
            state["next_obs"] if use_next else state["obs"], device=self.device
        ).float()
        if obs.ndim == len(self.env.observation_space["obs"].shape):
            obs = obs.unsqueeze(0)
        ret = [obs]
        if self.env.has_goal:
            goal = torch.tensor(state["goal"], device=self.device).float()
            if goal.ndim == len(self.env.observation_space["goal"].shape):
                goal = goal.unsqueeze(0)
            ret.append(goal)
        if self.env.has_obstacle:
            obstacle = torch.tensor(
                state["next_obstacle"] if use_next else state["obstacle"],
                device=self.device,
            ).float()
            if obstacle.ndim == len(self.env.observation_space["obstacle"].shape):
                obstacle = obstacle.unsqueeze(0)
            ret.append(obstacle)
        return ret

    def prepare_state_from_batch(
        self, batch: Dict[str, torch.Tensor], use_next: bool = False
    ) -> List[torch.Tensor]:
        obs = batch["next_obs"] if use_next else batch["obs"]
        if obs.ndim == len(self.env.observation_space["obs"].shape):
            obs = obs.unsqueeze(0)
        ret = [obs]
        if self.env.has_goal:
            goal = batch["goal"]
            if goal.ndim == len(self.env.observation_space["goal"].shape):
                goal = goal.unsqueeze(0)
            ret.append(goal)
        if self.env.has_obstacle:
            obstacle = batch["next_obstacle"] if use_next else batch["obstacle"]
            if obstacle.ndim == len(self.env.observation_space["obstacle"].shape):
                obstacle = obstacle.unsqueeze(0)
            ret.append(obstacle)
        return ret

    def select_action(self, state: Dict[str, np.ndarray], explore: bool):
        raise NotImplementedError

    def play_episode(self, explore: bool) -> List[Union[bool, Optional[float]]]:
        done = False
        state = self.env.reset()
        while not done:
            with torch.no_grad():
                action = self.select_action(state, explore)[0]
            state, _, done, _ = self.env.step(action)
        return [self.env.solved, self.env.numerical_evaluation]

    def learn_on_batch(self) -> Dict[str, float]:
        raise NotImplementedError

    def evaluate(
        self, episodes: int, return_value: bool = False
    ) -> List[Optional[float]]:
        results = [self.play_episode(explore=False) for _ in range(episodes)]
        success = float(np.mean([result[0] for result in results]))
        performance = (
            float(np.mean([result[1] for result in results]))
            if self.env.has_numerical_evaluation
            else None
        )
        self.logger.write_log(
            "Success rate over {0} episodes is {1}".format(episodes, success)
        )
        if return_value:
            return [success, performance]

    def train(self, episodes: int):
        raise NotImplementedError

    def plot(self):
        for loss_name in self.loss_manager.losses.keys():
            plt.plot(self.loss_manager.losses[loss_name], label=loss_name)
        plt.xlabel("SGD steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/losses/{0}.png".format(self.logger.name))
        plt.cla()
        if self.evaluate_episodes:
            plt.plot(
                self.evaluations.keys(),
                [e[0] for e in self.evaluations.values()],
                label="Success rate",
            )
            if self.env.has_numerical_evaluation:
                plt.plot(
                    self.evaluations.keys(),
                    [e[1] for e in self.evaluations.values()],
                    label="Performance rate",
                )
            plt.xlabel("Train episodes")
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/evaluations/{0}.png".format(self.logger.name))
