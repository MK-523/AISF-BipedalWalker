from stable_baselines3 import PPO
from wrappers import make_research_env

env = make_research_env(normalize=True, shaped=True)
model = PPO(
    "MlpPolicy", 
    env, 
    policy_kwargs=dict(net_arch=[256, 256]), 
    learning_rate=3e-4, 
    verbose=1,
    tensorboard_log="./ppo_logs/" 
)
model.learn(total_timesteps=500000)
model.save("ppo_bipedal_final")
env.save("vec_normalize.pkl")
