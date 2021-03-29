from agents.on_policy.a3c import AdvantageActorCritic
from environments import environments
from log_utils.log_utils import CustomLogger

logger = CustomLogger("a3c_cartpole")
agent = AdvantageActorCritic(
    hidden_size=64,
    device_name="cpu",
    env_name="CartPoleDictionary-v0",
    num_envs=2,
    env_fn=environments.gym.make,
    env_kwargs={},
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
