import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import os
import psutil
import multiprocessing
from multiprocessing import freeze_support
import time

def main():
    # Configuración de torch para optimizar rendimiento
    torch.backends.cudnn.benchmark = True
    
    # Detectar recursos disponibles
    num_cpu = multiprocessing.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_cores = psutil.cpu_count(logical=False)  # Cores físicos
    recommended_envs = min(cpu_cores, 8)  # Limitar a 8 ambientes máximo

    print(f"Dispositivo: {device}")
    print(f"CPUs físicas: {cpu_cores}")
    print(f"Ambientes paralelos: {recommended_envs}")

    # Directorios y configuración básica
    models_dir = "models/Carracing_SAC"
    logdir = "logs"
    continuous = True

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Hiperparámetros
    TIMESTEPS = 10000
    EPISODES = 3000
    BUFFER_SIZE = 300000

    # Función para crear un ambiente individual
    def make_env(rank):
        def _init():
            env = gym.make(
                "CarRacing-v2",
                continuous=continuous,
                lap_complete_percent=0.95,
                domain_randomize=False,
                render_mode=None
            )
            return env
        return _init

    # Crear ambientes vectorizados usando SubprocVecEnv
    env = SubprocVecEnv([make_env(i) for i in range(recommended_envs)])

    # Configurar el modelo SAC
    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        buffer_size=BUFFER_SIZE,
        learning_rate=3e-4,
        batch_size=min(256 * recommended_envs, 2048),  # Batch size adaptativo
        train_freq=4,
        gradient_steps=1,
        learning_starts=1000,
        tau=0.005,
        gamma=0.99,
        device=device,
        ent_coef="auto",
        target_entropy="auto",
    )

    # Función para monitorear recursos
    def print_resource_usage():
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # En GB
            print(f"GPU Memory: {gpu_memory:.2f}GB", end=" ")
        print(f"CPU: {cpu_percent}% Memory: {memory.percent}%")

    start_time = time.time()

    try:
        for i in range(1, EPISODES):
            print(f"\nEpisodio {i}/{EPISODES-1}")
            model.learn(
                total_timesteps=TIMESTEPS * recommended_envs,
                reset_num_timesteps=False,
                tb_log_name=f"sac_car_racing",
                progress_bar=True
            )
            
            # Monitorear recursos y guardar modelo
            print_resource_usage()
            model.save(f"{models_dir}/SAC_car_racing_{i}")
            
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
    finally:
        print("\nFinalizando entrenamiento...")
        print(f"Tiempo total de entrenamiento: {time.time() - start_time:.2f} segundos")
        env.close()

if __name__ == '__main__':
    freeze_support()
    main()