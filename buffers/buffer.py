from __future__ import annotations

import random
import numpy as np
from torch import tensor, Tensor, device
from torch.cuda import is_available
from typing import Dict, List, Optional


class SingleEnvTransition:
    def __init__(
        self,
        state: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_state: Dict[str, np.ndarray],
        done: bool,
        gamma: float,
        episode_return: Optional[float] = None,
        # Episode return represents the discounted sum of rewards from here until the end
        # ie r1 + gamma * r2 + ... + gamma^(T-1) * rT
        timestep: Optional[int] = None,
        # Timestep represents the number of steps from the start of the episode until here
        horizon: Optional[int] = None,
    ):
        self.has_goal = "goal" in state.keys()
        self.has_obstacle = "obstacle" in state.keys()
        self.obs: np.ndarray = state["obs"].copy()
        self.goal: Optional[np.ndarray] = (
            state["goal"].copy() if self.has_goal else None
        )
        self.action = action
        self.reward = reward
        self.next_obs: np.ndarray = next_state["obs"].copy()
        self.done = done
        self.obstacle: Optional[np.ndarray] = (
            state["obstacle"].copy() if self.has_obstacle else None
        )
        self.next_obstacle: Optional[np.ndarray] = (
            next_state["obstacle"] if self.has_obstacle else None
        )
        self._gamma = gamma
        self.episode_return = episode_return
        self.timestep = timestep
        self.horizon = horizon

    def set_episode_return(self, episode_return: float):
        self.episode_return = episode_return

    def set_timestep(self, timestep: int):
        self.timestep = timestep

    def set_horizon(self, horizon: int):
        self.horizon = horizon

    def copy(self) -> SingleEnvTransition:
        state = {"obs": self.obs.copy()}
        next_state = {"obs": self.next_obs.copy()}
        if self.has_goal:
            state["goal"] = self.goal.copy()
            next_state["goal"] = self.goal.copy()
        if self.has_obstacle:
            state["obstacle"] = self.obstacle.copy()
            state["next_obstacle"] = self.next_obstacle.copy()
        return SingleEnvTransition(
            state,
            self.action,
            self.reward,
            next_state,
            self.done,
            self._gamma,
            self.episode_return,
            self.timestep,
            self.horizon,
        )


class SingleEnvEpisodeBuffer:
    def __init__(
        self,
        use_hindsight: bool,
        has_goal: bool,
        has_obstacle: bool,
        gamma: float,
    ):
        assert has_goal or not use_hindsight, "HER only valid with goals"
        self._use_hindsight = use_hindsight
        self._has_goal = has_goal
        self._has_obstacle = has_obstacle
        self._gamma = gamma

        self.timestep = 0
        self.history: List[SingleEnvTransition] = []
        self._final_obs: Optional[np.ndarray] = None
        self.filled = False

    def __len__(self):
        return len(self.history)

    def observe(self, transition: SingleEnvTransition):
        assert not self.filled, "Episode buffer is full"
        assert (self._has_goal == transition.has_goal) and (
            self._has_obstacle == transition.has_obstacle
        ), "Malformed transition!"
        transition.set_timestep(self.timestep)
        self.history.append(transition)
        self._final_obs = transition.next_obs.copy()
        if transition.done:
            self.set_episode_returns_and_horizons(self.history)
            if self._use_hindsight:
                self.append_hindsight()
        self.timestep += 1
        self.filled = transition.done

    def set_episode_returns_and_horizons(self, history: List[SingleEnvTransition]):
        discounted_reward: float = 0.0
        horizon: int = 1
        for transition in reversed(history):
            horizon += 1
            discounted_reward = transition.reward + self._gamma * discounted_reward
            transition.set_episode_return(discounted_reward)
            transition.set_horizon(horizon)

    def append_hindsight(self):
        counterfactual_transitions: List[SingleEnvTransition] = []
        for transition in self.history:
            counterfactual_transition = self.generate_counterfactual_transition(
                transition
            )
            counterfactual_transitions.append(counterfactual_transition)
        self.set_episode_returns_and_horizons(counterfactual_transitions)
        self.history += counterfactual_transitions

    def generate_counterfactual_transition(
        self, transition: SingleEnvTransition
    ) -> SingleEnvTransition:
        counterfactual_transition = transition.copy()
        counterfactual_transition.goal = self._final_obs.copy()
        counterfactual_transition.reward = (
            0.0 if transition.timestep == len(self) - 1 else -1.0
        )
        return counterfactual_transition

    def flush(self):
        self.history = []
        self._final_obs = None
        self.filled = False
        self.timestep = 0


class SingleEnvBuffer:
    """Use an episode buffer to store processed trajectories FIFO"""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        use_hindsight: bool,
        has_goal: bool,
        has_obstacle: bool,
        gamma: float,
        device_name: str,
    ):
        assert has_goal or not use_hindsight, "HER only valid with goals"
        self._capacity = capacity
        self._batch_size = batch_size
        self._has_goal = has_goal
        self._has_obstacle = has_obstacle
        self._episode_buffer = SingleEnvEpisodeBuffer(
            use_hindsight, has_goal, has_obstacle, gamma
        )
        self.buffer: List[SingleEnvTransition] = []
        self._buffer_write_position: int = 0
        self.num_episodes: int = 0
        self.device_name = device(device_name if is_available() else "cpu")

    def __len__(self):
        return len(self.buffer)

    @property
    def filled(self) -> bool:
        return len(self) > self._capacity

    @property
    def ready_to_sample(self) -> bool:
        return len(self) > self._batch_size

    def add_transition(self, transition: SingleEnvTransition):
        if self.filled:
            self.buffer[self._buffer_write_position] = transition
        else:
            self.buffer.append(transition)
        self._buffer_write_position = (self._buffer_write_position + 1) % self._capacity

    def observe(self, transition: SingleEnvTransition):
        self._episode_buffer.observe(transition)
        if self._episode_buffer.filled:
            for transition in self._episode_buffer.history:
                self.add_transition(transition)
            self.num_episodes += 1
            self._episode_buffer.flush()

    def sample(self) -> Dict[str, Optional[Tensor]]:
        samples = random.sample(self.buffer, self._batch_size)
        obs = tensor(
            np.stack([x.obs for x in samples]), device=self.device_name
        ).float()
        goal = (
            tensor(np.stack([x.goal for x in samples]), device=self.device_name).float()
            if self._has_goal
            else None
        )
        action = tensor(
            np.stack([x.action for x in samples]), device=self.device_name
        ).long()
        reward = tensor(
            np.stack([x.reward for x in samples]), device=self.device_name
        ).float()
        episode_return = tensor(
            np.stack([x.episode_return for x in samples]), device=self.device_name
        ).float()
        timestep = tensor(
            np.stack([x.timestep for x in samples]), device=self.device_name
        ).long()
        horizon = tensor(
            np.stack([x.horizon for x in samples]), device=self.device_name
        ).long()
        next_obs = tensor(
            np.stack([x.next_obs for x in samples]), device=self.device_name
        ).float()
        done = tensor(
            np.stack([x.done for x in samples]), device=self.device_name
        ).bool()
        obstacle = (
            tensor(
                np.stack([x.obstacle for x in samples]), device=self.device_name
            ).float()
            if self._has_obstacle
            else None
        )
        next_obstacle = (
            tensor(
                np.stack([x.next_obstacle for x in samples]), device=self.device_name
            ).float()
            if self._has_obstacle
            else None
        )
        return {
            "obs": obs,
            "goal": goal,
            "action": action,
            "reward": reward,
            "episode_return": episode_return,
            "timestep": timestep,
            "horizon": horizon,
            "next_obs": next_obs,
            "done": done,
            "obstacle": obstacle,
            "next_obstacle": next_obstacle,
        }

    def flush(self):
        self.buffer = []


class MultiEnvTransition:
    def __init__(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        next_state: Dict[str, np.ndarray],
        done: np.ndarray,
        gamma: float,
        episode_return: Optional[np.ndarray] = None,
        # Episode return represents the discounted sum of rewards from here until the end
        # ie r1 + gamma * r2 + ... + gamma^(T-1) * rT
        timestep: Optional[np.ndarray] = None,
        # Timestep represents the number of steps from the beginning of the episode until here
        horizon: Optional[np.ndarray] = None,
    ):
        self.has_goal = "goal" in state.keys()
        self.has_obstacle = "obstacle" in state.keys()
        self._num_envs = len(done)
        all_obs = state["obs"]
        if self.has_goal:
            all_goal = state["goal"]
            self.goal: Optional[Dict[int, np.ndarray]] = {
                env: all_goal[env].copy() for env in range(self._num_envs)
            }
        else:
            self.goal: Optional[Dict[int, np.ndarray]] = None
        all_next_obs = next_state["obs"]
        self.obs: Dict[int, np.ndarray] = {
            env: all_obs[env].copy() for env in range(self._num_envs)
        }
        self.next_obs: Dict[int, np.ndarray] = {
            env: all_next_obs[env].copy() for env in range(self._num_envs)
        }
        self.action: Dict[int, int] = {
            env: action[env] for env in range(self._num_envs)
        }
        self.reward: Dict[int, float] = {
            env: reward[env] for env in range(self._num_envs)
        }
        self.done: Dict[int, bool] = {env: done[env] for env in range(self._num_envs)}
        if self.has_obstacle:
            all_obstacle = state["obstacle"]
            all_next_obstacle = next_state["obstacle"]
            self.obstacle: Optional[Dict[int, np.ndarray]] = {
                env: all_obstacle[env].copy() for env in range(self._num_envs)
            }
            self.next_obstacle: Optional[Dict[int, np.ndarray]] = {
                env: all_next_obstacle[env].copy() for env in range(self._num_envs)
            }
        else:
            self.obstacle: Optional[Dict[int, np.ndarray]] = None
            self.next_obstacle: Optional[Dict[int, np.ndarray]] = None
        self._gamma = gamma
        self.episode_return = episode_return
        self.timestep = timestep
        self.horizon = horizon

    def __getitem__(self, env):
        obs: np.ndarray = self.obs[env].copy()
        action = self.action[env]
        reward = self.reward[env]
        next_obs: np.ndarray = self.next_obs[env].copy()
        done = self.done[env]
        state: Dict[str, np.ndarray] = {"obs": obs}
        next_state: Dict[str, np.ndarray] = {"obs": next_obs}
        if self.has_goal:
            goal: np.ndarray = self.goal[env].copy()
            state["goal"] = goal
            next_state["goal"] = goal
        if self.has_obstacle:
            obstacle: np.ndarray = self.obstacle[env].copy()
            next_obstacle: np.ndarray = self.next_obstacle[env].copy()
            state["obstacle"] = obstacle
            next_state["obstacle"] = next_obstacle
        episode_return = self.episode_return[env] if self.episode_return else None
        timestep = self.timestep[env] if self.timestep else None
        horizon = self.horizon[env] if self.horizon else None
        return SingleEnvTransition(
            state,
            action,
            reward,
            next_state,
            done,
            self._gamma,
            episode_return,
            timestep,
            horizon,
        )


class MultiEnvEpisodeBuffer:
    def __init__(
        self,
        num_envs: int,
        use_hindsight: bool,
        has_goal: bool,
        has_obstacle: bool,
        gamma: float,
    ):
        assert has_goal or not use_hindsight, "HER only valid with goals"
        self._num_envs = num_envs
        self._use_hindsight = use_hindsight
        self._has_goal = has_goal
        self._has_obstacle = has_obstacle
        self._gamma = gamma

        self.buffers: Dict[int, SingleEnvEpisodeBuffer] = {
            env: SingleEnvEpisodeBuffer(use_hindsight, has_goal, has_obstacle, gamma)
            for env in range(num_envs)
        }

    def __getitem__(self, env):
        return self.buffers[env]

    @property
    def filled(self) -> List[bool]:
        return [self.buffers[env].filled for env in self.buffers]

    def observe(self, transition: MultiEnvTransition):
        assert (self._has_goal == transition.has_goal) and (
            self._has_obstacle == transition.has_obstacle
        ), "Malformed transition!"
        for env in self.buffers:
            self.buffers[env].observe(transition[env])

    def flush(self, env: int):
        self.buffers[env].flush()


class MultiEnvBuffer:
    def __init__(
        self,
        num_envs: int,
        capacity: int,
        batch_size: int,
        use_hindsight: bool,
        has_goal: bool,
        has_obstacle: bool,
        gamma: float,
        device_name: str,
    ):
        assert has_goal or not use_hindsight, "HER only valid with goals"
        self._capacity = capacity
        self._batch_size = batch_size
        self._has_goal = has_goal
        self._has_obstacle = has_obstacle
        self._episode_buffer = MultiEnvEpisodeBuffer(
            num_envs, use_hindsight, has_goal, has_obstacle, gamma
        )
        self.buffer: List[SingleEnvTransition] = []
        self._buffer_write_position: int = 0
        self.num_episodes = 0
        self.device_name = device(device_name if is_available() else "cpu")

    def __len__(self):
        return len(self.buffer)

    @property
    def filled(self) -> bool:
        return len(self) > self._capacity

    @property
    def ready_to_sample(self) -> bool:
        return len(self) > self._batch_size

    def add_transition(self, transition: SingleEnvTransition):
        if self.filled:
            self.buffer[self._buffer_write_position] = transition
        else:
            self.buffer.append(transition)
        self._buffer_write_position = (self._buffer_write_position + 1) % self._capacity

    def observe(self, transition: MultiEnvTransition):
        self._episode_buffer.observe(transition)
        for env, done in enumerate(self._episode_buffer.filled):
            if done:
                for transition in self._episode_buffer[env].history:
                    self.add_transition(transition)
                self.num_episodes += 1
                self._episode_buffer[env].flush()

    def sample(self) -> Dict[str, Optional[Tensor]]:
        samples = random.sample(self.buffer, self._batch_size)
        obs = tensor(
            np.stack([x.obs for x in samples]), device=self.device_name
        ).float()
        goal = (
            tensor(np.stack([x.goal for x in samples]), device=self.device_name).float()
            if self._has_goal
            else None
        )
        action = tensor(
            np.stack([x.action for x in samples]), device=self.device_name
        ).long()
        reward = tensor(
            np.stack([x.reward for x in samples]), device=self.device_name
        ).float()
        episode_return = tensor(
            np.stack([x.episode_return for x in samples]), device=self.device_name
        ).float()
        timestep = tensor(
            np.stack([x.timestep for x in samples]), device=self.device_name
        ).long()
        horizon = tensor(
            np.stack([x.horizon for x in samples]), device=self.device_name
        ).long()
        next_obs = tensor(
            np.stack([x.next_obs for x in samples]), device=self.device_name
        ).float()
        done = tensor(
            np.stack([x.done for x in samples]), device=self.device_name
        ).bool()
        obstacle = (
            tensor(
                np.stack([x.obstacle for x in samples]), device=self.device_name
            ).float()
            if self._has_obstacle
            else None
        )
        next_obstacle = (
            tensor(
                np.stack([x.next_obstacle for x in samples]), device=self.device_name
            ).float()
            if self._has_obstacle
            else None
        )
        return {
            "obs": obs,
            "goal": goal,
            "action": action,
            "reward": reward,
            "episode_return": episode_return,
            "timestep": timestep,
            "horizon": horizon,
            "next_obs": next_obs,
            "done": done,
            "obstacle": obstacle,
            "next_obstacle": next_obstacle,
        }

    def flush(self):
        self.buffer = []
