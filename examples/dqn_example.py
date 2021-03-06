from agents.off_policy.dqn import DeepQualityNetwork
from samplers.samplers import EpsilonGreedySampler
from environments import environments
from log_utils.custom_logger import CustomLogger

logger = CustomLogger("dqn_cartpole")
agent = DeepQualityNetwork(
    hidden_size=64,
    sampler=EpsilonGreedySampler(lambda x: 0.05),
    device_name="cpu",
    env_name="CartPoleDictionary-v0",
    num_envs=2,
    env_fn=environments.gym.make,
    env_kwargs={},
    polyak_weight=0.01,
    gamma=0.95,
    capacity=5000,
    batch_size=64,
    use_hindsight=False,
    optim_steps=2,
    loss_weight=0.05,
    loss_epsilon=1e-6,
    logger=logger,
    log_freq=10,
    evaluate_episodes=10,
)

agent.train(200)
agent.plot()
