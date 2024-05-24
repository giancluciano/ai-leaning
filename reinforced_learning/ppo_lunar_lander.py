import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure




if __name__ == "__main__":
    train_model = False
    model = None
    if train_model:
        logger = configure('tmp', ["stdout", "tensorboard"])

        n_steps = 1024  # default 2048
        n_envs = 12
        vec_env = make_vec_env("LunarLander-v2", n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps = n_steps,
            n_epochs = 10,
            batch_size = 64, # ? actions taken in a single training default 64
            gamma = 0.99,
        )
        # 24576
        model.set_logger(logger)
        print("training")
        model.learn(total_timesteps=n_steps * n_envs * 50)
        model.save("ppo_lunarlander")
        # n_steps = 2048
        # ep_rew_mean          | 109
        # time_elapsed         | 435


        #del model

    model = PPO.load("ppo_lunarlander", env=gym.make("LunarLander-v2"))
    env = model.get_env()
    obs = env.reset()

    for _ in range(1024):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if rewards[0]  == 100:
            print(rewards)
            obs = env.reset()

    env.close() 