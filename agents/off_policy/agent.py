from buffers.buffer import MultiEnvTransition, MultiEnvBuffer
from networks.off_policy import OffPolicyNetworkFactory
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from common.multiprocessing_env import SubprocVecEnv


def make_envs(env_name: str, env_fn, env_kwargs):
    def _thunk():
        env = env_fn(env_name, **env_kwargs)
        return env

    return _thunk


class Agent:
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
    ):
        self.network = OffPolicyNetworkFactory(network, polyak_weight)
        self.network.synchronise(use_polyak=False)
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
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
        self.optim = torch.optim.Adam(self.network.online_network.parameters())
        self.optim_steps = optim_steps
        self.losses_by_type: Dict[str, List[float]] = {}

    @property
    def num_episodes_trained(self) -> int:
        return self.buffer.num_episodes

    def prepare_state(self, state: Dict[str, np.ndarray]) -> List[torch.Tensor]:
        obs = torch.tensor(state["obs"], device=self.device).float()
        if obs.ndim == len(self.env.observation_space["obs"].shape):
            obs = obs.unsqueeze(0)
        ret = [obs]
        if self.env.has_goal:
            goal = torch.tensor(state["goal"], device=self.device).float()
            if goal.ndim == len(self.env.observation_space["goal"].shape):
                goal = goal.unsqueeze(0)
            ret.append(goal)
        if self.env.has_obstacle:
            obstacle = torch.tensor(state["obstacle"], device=self.device).float()
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
                action = self.select_action(state)
            state, _, done, _ = self.env.step(action)
        return self.env.solved

    def evaluate(self, episodes: int) -> float:
        return float(np.mean([self.play_episode() for _ in range(episodes)]))

    def sample(self) -> Dict[str, Optional[torch.Tensor]]:
        return self.buffer.sample()

    def learn_on_batch(self) -> Dict[str, float]:
        raise NotImplementedError

    def train_on_batch(self):
        losses = self.learn_on_batch()
        for loss_name in losses:
            self.losses_by_type[loss_name].append(losses[loss_name])
        self.network.synchronise()

    def train(self, episodes: int):
        state = self.envs.reset()
        while self.num_episodes_trained < episodes:
            with torch.no_grad():
                action = self.select_action(state)
            next_state, reward, done, _ = self.envs.step(action)
            transition = MultiEnvTransition(
                state, action, reward, next_state, done, self.gamma
            )
            self.buffer.observe(transition)
            state = next_state
            if self.buffer.ready_to_sample:
                self.train_on_batch()
