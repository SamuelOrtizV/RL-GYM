import gymnasium as gym

# Initialise the environment
env = gym.make("CarRacing-v2", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

observation = env.reset()

for _ in range(1000):  # 1000 pasos
    env.render()  # Muestra la simulación
    action = env.action_space.sample()  # Toma una acción aleatoria
    observation, reward, terminated, truncated, info = env.step(action)  # Realiza la acción y recibe resultados
    if terminated:
        observation = env.reset()  # Reinicia el entorno si el episodio terminó
env.close()