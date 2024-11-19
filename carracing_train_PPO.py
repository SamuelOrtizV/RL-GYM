import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import torch
import os

models_dir = "models/Carracing_PPO"
logdir = "logs"
continuous = False
os.makedirs(models_dir, exist_ok=True) # Ensure the models directory exists
os.makedirs(logdir, exist_ok=True) # Ensure the logs directory exists

# Hyperparameters
TIMESTEPS = 10000
EPISODES = 3000
BUFFER_SIZE = 100000

# Initialise the environment
#env = gym.make("CarRacing-v2")
env = make_vec_env("CarRacing-v2", n_envs=1,
    env_kwargs={
    "lap_complete_percent": 0.95,
    "domain_randomize": False,
    "continuous": continuous})

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir)

start_time = time.time()

for i in range(1,EPISODES):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"ppo_cnn_car_racing")
    model.save(f"{models_dir}/ppo_cnn_car_racing_{i}")


print("Model trained.")
# Guardar el modelo
model.save("ppo_cnn_car_racing")

print(time.time() - start_time)

env.close()