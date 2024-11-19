import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import torch
import os

models_dir = "best models"
loaded_model = "ppo_car_racing_10.zip"

continuous = False

BUFFER_SIZE = 100000

def show_progress(model):
    #env_test = gym.make("CarRacing-v2", render_mode="human")
    env_test = make_vec_env("CarRacing-v2", n_envs=1,
        env_kwargs={
        "render_mode": "human",
        "lap_complete_percent": 0.95,
        "domain_randomize": False,
        "continuous": continuous})
    obs = env_test.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        step_result = env_test.step(action)
        obs, _, terminated, truncated = step_result
        done = terminated[0] or truncated[0]["TimeLimit.truncated"]
        env_test.render()
    env_test.close()

# Initialise the environment
#env = gym.make("CarRacing-v2")
env = make_vec_env("CarRacing-v2", n_envs=1,
    env_kwargs={
    "lap_complete_percent": 0.95,
    "domain_randomize": False,
    "continuous": continuous})

# model = PPO("CnnPolicy", env, verbose=1)

""" model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=BUFFER_SIZE,
        learning_rate=3e-4,
        train_freq=4,
        gradient_steps=1,
        learning_starts=1000,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
    ) """

# Cargar el modelo
model = PPO.load(os.path.join(models_dir, loaded_model))

# Mostrar el progreso
show_progress(model)

env.close()