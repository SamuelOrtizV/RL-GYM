import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

class CustomCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CustomCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.fc(x)
        return x


# Crear un extractor personalizado de características usando tu CNN
class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        # La CNN personalizada
        self.cnn = CustomCNN(observation_space.shape, features_dim)

    def forward(self, observations):
        return self.cnn(observations)


def custom_reward_function(state, action, reward, done):
    # Penalizar si el coche sale de la pista
    if state["off_track"]:
        reward -= 10
    return reward

def plot_rewards(reward_history):
    plt.plot(reward_history)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Rewards over time")
    plt.show()

# Crear una política personalizada que usa la CNN como extractor de características
policy_kwargs = dict(
    features_extractor_class=CustomCNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=512)
)

# Cargar el entorno (con vectores para paralelismo)
env = make_vec_env('CarRacing-v0', n_envs=1)

# Definir el modelo PPO usando la política personalizada
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

model = PPO("CnnPolicy", env, 
            policy_kwargs=policy_kwargs, 
            learning_rate=3e-4,  # Tasa de aprendizaje
            n_steps=2048,        # Tamaño del buffer de pasos antes de actualizar la red
            batch_size=64,        # Tamaño del lote de entrenamiento
            n_epochs=10,          # Número de veces que se entrena la red por actualización
            gamma=0.99,           # Factor de descuento para recompensas futuras
            gae_lambda=0.95,      # Parámetro de suavizado de las ventajas
            clip_range=0.2,       # Rango de clipping para PPO
            verbose=1)

# Para guardar las recompensas
reward_history = []

for i in range(100):  # 100 episodios
    done = False
    obs = env.reset()
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    reward_history.append(total_reward)

plot_rewards(reward_history)

# Entrenar el modelo
model.learn(total_timesteps=500000)  # Modifica según el tiempo y los recursos disponibles

# Guardar el modelo
model.save("ppo_carracing_custom_cnn")

# Cargar y evaluar el modelo
model = PPO.load("ppo_carracing_custom_cnn")

# Evaluar el agente entrenado
obs = env.reset()
for _ in range(1000):  # Probar 1000 pasos
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # Ver la simulación
env.close()
