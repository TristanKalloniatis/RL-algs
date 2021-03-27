import gym
import numpy as np
from typing import List, Dict

gym.envs.register(id="Gridworld-v0", entry_point="environments.environments:Gridworld")
gym.envs.register(
    id="SimpleMaze-v0", entry_point="environments.environments:SimpleMaze"
)
gym.envs.register(
    id="CartPoleDictionary-v0",
    entry_point="environments.environments:CartPoleDictionary",
)


class Environment(gym.Env):
    def __init__(self, has_goal: bool, has_obstacle: bool):
        super().__init__()
        self.has_goal = has_goal
        self.has_obstacle = has_obstacle
        self.episode_length = 0

    @property
    def solved(self) -> bool:
        raise NotImplementedError


class Gridworld(Environment):
    def __init__(self, grid_size: int):
        super().__init__(has_goal=True, has_obstacle=False)
        self._grid_size = grid_size

        self._episode_length = 0
        self._state = np.random.randint(low=0, high=grid_size, size=2)
        self._goal_state = np.random.randint(low=0, high=grid_size, size=2)

        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(0.0, 1.0, (1, grid_size, grid_size), np.float32),
                "goal": gym.spaces.Box(0.0, 1.0, (1, grid_size, grid_size), np.float32),
            }
        )

        self._non_trivial_state()

    def _equal(self, state: np.ndarray, goal_state: np.ndarray) -> bool:
        return (state == goal_state).all()

    @property
    def solved(self) -> bool:
        return self._equal(self._state, self._goal_state)

    def _non_trivial_state(self):
        while self.solved:  # Make sure I have something to do
            self._state = np.random.randint(low=0, high=self._grid_size, size=2)
            self._goal_state = np.random.randint(low=0, high=self._grid_size, size=2)

    def _compute_reward(self, state: np.ndarray, goal_state: np.ndarray):
        return 0.0 if self._equal(state, goal_state) else -1.0

    @property
    def reward(self):
        return self._compute_reward(self._state, self._goal_state)

    def _compute_observation(self, state: np.ndarray):
        observation = np.zeros((1, self._grid_size, self._grid_size), dtype=np.float32)
        observation[0, state[0], state[1]] = 1
        return observation

    @property
    def observation(self):
        return self._compute_observation(self._state)

    @property
    def goal(self):
        return self._compute_observation(self._goal_state)

    @property
    def total_observation(self):
        return {"obs": self.observation, "goal": self.goal}

    def step(self, action: int):
        assert action in range(4), "Provided invalid action {0}".format(action)
        if action == 0:
            self._state[0] = max(self._state[0] - 1, 0)
        elif action == 1:
            self._state[1] = max(self._state[1] - 1, 0)
        elif action == 2:
            self._state[0] = min(self._state[0] + 1, self._grid_size - 1)
        elif action == 3:
            self._state[1] = min(self._state[1] + 1, self._grid_size - 1)
        self._episode_length += 1
        done = self.solved or self._episode_length == 2 * self._grid_size
        return self.total_observation, self.reward, done, {}

    def up(self):
        return self.step(0)

    def left(self):
        return self.step(1)

    def down(self):
        return self.step(2)

    def right(self):
        return self.step(3)

    def reset(self):
        self.__init__(self._grid_size)
        return self.total_observation


class SimpleMaze(Environment):
    def __init__(self, grid_size: int):
        super().__init__(has_goal=True, has_obstacle=True)
        self._grid_size = grid_size

        self.episode_length = 0
        self.state = np.random.randint(low=0, high=grid_size, size=2)
        self.goal_state = np.random.randint(low=0, high=grid_size, size=2)
        self.obstacle_state = [np.random.randint(low=0, high=grid_size, size=2)]

        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(0.0, 1.0, (grid_size, grid_size, 1), np.float32),
                "goal": gym.spaces.Box(0.0, 1.0, (grid_size, grid_size, 1), np.float32),
                "obstacle": gym.spaces.Box(
                    0.0, 1.0, (grid_size, grid_size, 1), np.float32
                ),
            }
        )

        self.non_trivial_state()

    def equal(self, state: np.ndarray, goal_state: np.ndarray) -> bool:
        return (state == goal_state).all()

    @property
    def solved(self):
        return self.equal(self.state, self.goal_state)

    @property
    def obstacle_clash(self):
        return self.equal(self.state, self.obstacle_state[0]) or self.equal(
            self.goal_state, self.obstacle_state[0]
        )

    def non_trivial_state(self):
        while self.solved or self.obstacle_clash:  # Make sure I have something to do
            self.state = np.random.randint(low=0, high=self._grid_size, size=2)
            self.goal_state = np.random.randint(low=0, high=self._grid_size, size=2)
            self.obstacle_state = [
                np.random.randint(low=0, high=self._grid_size, size=2)
            ]

    def compute_reward(self, state: np.ndarray, goal_state: np.ndarray):
        return 0.0 if self.equal(state, goal_state) else -1.0

    @property
    def reward(self):
        return self.compute_reward(self.state, self.goal_state)

    def compute_observation(self, state: np.ndarray):
        observation = np.zeros((self._grid_size, self._grid_size, 1), dtype=np.float32)
        observation[state[0], state[1], 0] = 1
        return observation

    def compute_state(self, observation: np.ndarray):
        # Invert the above, needed for computing counterfactual rewards
        # There's probably a better way to do this also...
        return np.array(
            [
                np.sum(np.argmax(observation, axis=0)),
                np.sum(np.argmax(observation, axis=1)),
            ]
        )

    def compute_reward_from_observation(
        self, observation: np.ndarray, goal: np.ndarray
    ):
        return self.compute_reward(
            self.compute_state(observation), self.compute_state(goal)
        )

    def compute_obstacle(self, obstacle_state: List[np.ndarray]):
        observation = np.zeros((self._grid_size, self._grid_size, 1), dtype=np.float32)
        for state in obstacle_state:
            observation[state[0], state[1], 0] = 1
        return observation

    @property
    def obstacle(self):
        return self.compute_obstacle(self.obstacle_state)

    @property
    def observation(self):
        return self.compute_observation(self.state)

    @property
    def goal(self):
        return self.compute_observation(self.goal_state)

    @property
    def total_observation(self):
        return {"obs": self.observation, "goal": self.goal, "obstacle": self.obstacle}

    def step(self, action: int):
        assert action in range(4), "Provided invalid action {0}".format(action)
        prev_state = self.state.copy()
        next_state = self.state.copy()
        if action == 0:
            next_state[0] = max(self.state[0] - 1, 0)
        elif action == 1:
            next_state[1] = max(self.state[1] - 1, 0)
        elif action == 2:
            next_state[0] = min(self.state[0] + 1, self._grid_size - 1)
        elif action == 3:
            next_state[1] = min(self.state[1] + 1, self._grid_size - 1)
        blocked = False
        for state in self.obstacle_state:
            if self.equal(state, next_state):
                blocked = True
                break
        if not blocked:
            self.state = next_state
        self.episode_length += 1
        done = self.solved or self.episode_length == 2 * self._grid_size
        if not self.equal(prev_state, self.state):
            self.obstacle_state.append(prev_state)
        return self.total_observation, self.reward, done, {}

    def up(self):
        return self.step(0)

    def left(self):
        return self.step(1)

    def down(self):
        return self.step(2)

    def right(self):
        return self.step(3)

    def reset(self):
        self.__init__(self._grid_size)
        return self.total_observation


class DictionaryWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"obs": env.observation_space})

    def observation(self, observation: np.ndarray):
        return {"obs": observation}


class CartPoleDictionary(Environment):
    def __init__(self, steps_considered_solved: int = 100):
        super().__init__(has_goal=False, has_obstacle=False)
        cart_pole = gym.make("CartPole-v0")
        self.env = DictionaryWrapper(cart_pole)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.steps_considered_solved = steps_considered_solved
        self.episode_length = 0

    @property
    def solved(self):
        return self.episode_length >= self.steps_considered_solved

    def set_steps_considered_solved(self, step_considered_solved: int):
        self.steps_considered_solved = step_considered_solved

    def step(self, action: int):
        self.episode_length += 1
        return self.env.step(action)

    def reset(self) -> Dict[str, np.ndarray]:
        self.episode_length = 0
        return self.env.reset()
