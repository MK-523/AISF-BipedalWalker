from stable_baselines3 import PPO
from wrappers import make_research_env

env = make_research_env(normalize=True, shaped=True)
model = PPO("MlpPolicy", env, n_steps=4096, verbose=1)
model.learn(total_timesteps=100000)
model.save("exp_b_model")