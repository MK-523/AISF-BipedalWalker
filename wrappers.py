import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

class ResearchRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        angle = self.env.unwrapped.hull.angle
        vel_x = self.env.unwrapped.hull.linearVelocity[0]
        
        if abs(angle) > 0.15:
            reward -= 0.1
        if vel_x > 1.0:
            reward += 0.05
        return reward

def make_research_env(num_envs=8, normalize=True, shaped=True):
    def env_factory():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(env)
        if shaped:
            env = ResearchRewardWrapper(env)
        return env
    
    venv = DummyVecEnv([env_factory for _ in range(num_envs)])
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)
    return venv