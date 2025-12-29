from stable_baselines3 import PPO
from wrappers import make_research_env

env = make_research_env(normalize=True, shaped=False)
model = PPO("MlpPolicy", env, ent_coef=0.05, verbose=1)
model.learn(total_timesteps=100000)
model.save("exp_a_model")