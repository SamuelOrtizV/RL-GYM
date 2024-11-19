import gym
import numpy as np

env = gym.make('MountainCar-v0')

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000

SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

def show_progress(q_table):
    env_test = gym.make('MountainCar-v0', render_mode='human')
    discrete_state = get_discrete_state(env_test.reset()[0])
    done2 = False
    while not done2:
        action2 = np.argmax(q_table[discrete_state])
        next_state2, _, terminated2, truncated2, _ = env_test.step(action2)
        done2 = terminated2 or truncated2
        discrete_state = get_discrete_state(next_state2)
        #env_test.render()
    env_test.close()    

for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        print("episode: ", episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset()[0])

    done = False

    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        #action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)

        if render:
            show_progress(q_table)
            render = False

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()