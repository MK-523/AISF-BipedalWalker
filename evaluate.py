import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import imageio

def record(model_path="ppo_bipedal_final", stats_path="vec_normalize.pkl"):
    env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", render_mode="rgb_array")])
    env = VecNormalize.load(stats_path, env)
    env.training = False 
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    obs = env.reset()
    frames = []
    
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done:
            break

    imageio.mimsave("walker_performance.mp4", frames, fps=30, codec="libx264", macro_block_size=1)

if __name__ == "__main__":
    record()