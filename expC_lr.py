from stable_baselines3 import PPO
from wrappers import make_research_env

env = make_research_env(normalize=True, shaped=True)
model = PPO("MlpPolicy", env, learning_rate=1e-4, ent_coef=0.01, verbose=1)
model.learn(total_timesteps=200000)
model.save("exp_c_model")