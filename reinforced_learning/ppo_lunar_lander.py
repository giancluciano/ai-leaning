import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure


logger = configure('tmp', ["stdout", "tensorboard"])

vec_env = make_vec_env("LunarLander-v2", n_envs=16, vec_env_cls=SubprocVecEnv)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.set_logger(logger)
print("training")
model.learn(total_timesteps=200000)
model.save("ppo_lunarlander")

del model

model = PPO.load("ppo_lunarlander", env=gym.make("LunarLander-v2", render_mode="human"))
env = model.get_env()
obs = env.reset()
model.set_env(env)
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)


env.close()

