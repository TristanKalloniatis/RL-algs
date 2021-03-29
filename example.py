from agents.off_policy.dqn import DeepQualityNetwork
from policies.policy import EpsilonGreedySampler
from environments import environments
from log_utils.log_utils import CustomLogger

logger = CustomLogger("example")
agent = DeepQualityNetwork(
    hidden_size=64,
    sampler=EpsilonGreedySampler(lambda x: 0.05),
    device_name="cpu",
    env_name="CartPoleDictionary-v0",
    num_envs=1,
    env_fn=environments.gym.make,
    env_kwargs={},
    polyak_weight=0.01,
    gamma=0.95,
    capacity=5000,
    batch_size=64,
    use_hindsight=False,
    optim_steps=1,
    loss_weight=0.05,
    loss_epsilon=1e-6,
    logger=logger,
    log_freq=10,
    evaluate_episodes=10,
)

agent.train(500)
agent.plot()
