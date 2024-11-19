import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import time
import torch
import os

# Directorios para modelos y logs
models_dir = "models/Carracing_SAC"
logdir = "logs"
continuous = True

# Crear directorios si no existen
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Hiperparámetros
TIMESTEPS = 1000
EPISODES = 30
BUFFER_SIZE = 100000

# Crear y configurar el ambiente
env = make_vec_env(
    "CarRacing-v2", 
    n_envs=1,
    env_kwargs={
        "lap_complete_percent": 0.95,
        "domain_randomize": False,
        "continuous": continuous,
        "render_mode": None  # Añadido para evitar problemas de renderizado
    }
)

# Configurar el modelo SAC con policy CNN
model = SAC(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    buffer_size=BUFFER_SIZE,
    learning_rate=0.0003,  # Ajustado para mejor estabilidad
    batch_size=256,        # Aumentado para mejor aprendizaje
    train_freq=1,          # Frecuencia de entrenamiento
    gradient_steps=1,      # Pasos de gradiente por actualización
    learning_starts=1000   # Muestras antes de empezar el entrenamiento
)

start_time = time.time()

try:
    for i in range(1, EPISODES):
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=f"sac_car_racing"
        )
        # Guardar el modelo después de cada episodio
        model.save(f"{models_dir}/sac_car_racing_{i}")
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
finally:
    print("Finalizando entrenamiento...")
    # Guardar el modelo final
    model.save("sac_car_racing1")
    print(f"Tiempo total de entrenamiento: {time.time() - start_time} segundos")
    env.close()