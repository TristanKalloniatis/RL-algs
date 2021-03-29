from buffers.buffer import MultiEnvTransition, MultiEnvBuffer
from networks.off_policy import OffPolicyNetworkFactory
import torch
from typing import Optional, Dict, Any, List, Callable
from common.multiprocessing_env import SubprocVecEnv
from losses.multi_loss import LossManager
from environments.env_constructor import make_envs
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
    ):
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.network = OffPolicyNetworkFactory(network.to(self.device), polyak_weight)
        self.network.synchronise(use_polyak=False)
        self.env = env_fn(env_name, **env_kwargs)
        super().__init__(self.env, device_name, logger)
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
        self.loss_manager = LossManager(loss_names, loss_weight, loss_epsilon)
        self.log_freq = log_freq

    def sample(self) -> Dict[str, Optional[torch.Tensor]]:
        return self.buffer.sample()

    def train_on_batch(self):
        self.learn_on_batch()
        self.network.synchronise()

    def train(self, episodes: int):
        state = self.envs.reset()
        while self.buffer.num_episodes < episodes:
            if self.buffer.num_episodes % self.log_freq == 0:
                self.logger.write_log(
                    "Seen {0} episodes so far".format(self.buffer.num_episodes)
                )
            with torch.no_grad():
                action = self.select_action(state)
            next_state, reward, done, _ = self.envs.step(action)
            transition = MultiEnvTransition(
                state, action, reward, next_state, done, self.gamma
            )
            self.buffer.observe(transition)
            state = next_state
            if self.buffer.ready_to_sample:
                for _ in range(self.optim_steps):
                    self.train_on_batch()
