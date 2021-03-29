from typing import Dict, List
from environments.environments import Environment
import numpy as np
import torch


class Agent:
    def __init__(self, env: Environment, device_name: str):
        self.env = env
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")

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

    def select_action(self, state: Dict[str, np.ndarray]):
        raise NotImplementedError

    def play_episode(self) -> bool:
        done = False
        state = self.env.reset()
        while not done:
            with torch.no_grad():
                action = self.select_action(state)[0]
            state, _, done, _ = self.env.step(action)
        return self.env.solved

    def learn_on_batch(self) -> Dict[str, float]:
        raise NotImplementedError

    def evaluate(self, episodes: int) -> float:
        return float(np.mean([self.play_episode() for _ in range(episodes)]))

    def train(self, episodes: int):
        raise NotImplementedError
