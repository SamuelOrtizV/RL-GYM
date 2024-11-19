import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot") # setting the style of the plot

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.2
EPS_DECAY = 0.9998

SHOW_EVERY = 3000
start_q_table = "qtable-1716439870.pickle" # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

class Blob:
    def __init__(self): # initialize the blob
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self): # print the blobs location for debugging
        return f"{self.x}, {self.y}"

    def __sub__(self, other): # subtract the blobs location from another blob
        return (self.x-other.x, self.y-other.y)

    def action(self, choice): # move the blob based on the choice made by the Q table
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        

    def move(self, x=False, y=False): # move the blob in the x and y direction
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

if start_q_table is None: # if the Q table is not provided, create a new one
    q_table = {}
    for x1 in range(-SIZE+1, SIZE): # create the Q table with all possible states and actions for the blobs to take in the environment 
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(8)] # initialize the Q table with random values

else: # if the Q table is provided, load it
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = [] # list to store the rewards for each episode

for episode in range(HM_EPISODES): # loop through each episode
    player = Blob() # create the player blob
    food = Blob() # create the food blob
    enemy = Blob() # create the enemy blob

    if episode % SHOW_EVERY == 0: # print the episode number every SHOW_EVERY episodes
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0 # initialize the episode reward
    for i in range(200): # loop through each step in the episode
        obs = (player-food, player-enemy) # get the observation of the player blob
        if np.random.random() > epsilon: # choose the best action based on the Q table
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 8)

        player.action(action) # move the player blob based on the action chosen

        enemy.move() # move the enemy blob
        food.move() # move the food blob

        if player.x == enemy.x and player.y == enemy.y: # if the player blob and enemy blob collide, subtract the enemy penalty from the reward
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y: # if the player blob and food blob collide, add the food reward to the reward
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food, player-enemy) # get the new observation of the player blob
        max_future_q = np.max(q_table[new_obs]) # get the maximum Q value for the new observation
        current_q = q_table[obs][action] # get the current Q value for the current observation and action
        if reward == FOOD_REWARD: # if the player blob collides with the food blob, set the target Q value to the reward
            new_q = FOOD_REWARD
        else: # if the player blob does not collide with the food blob, calculate the target Q value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q # update the Q table with the new Q value

        if show: # show the environment every SHOW_EVERY episodes
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]
            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("Title", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break

        episode_reward += reward # add the reward to the episode reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward) # add the episode reward to the list of episode rewards
    epsilon *= EPS_DECAY # decay the epsilon value

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid") # calculate the moving average of the episode rewards

plt.plot([i for i in range(len(moving_avg))], moving_avg) # plot the moving average
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f: # save the Q table
    pickle.dump(q_table, f)