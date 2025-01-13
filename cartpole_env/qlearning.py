"""
Note - Qlearning takes around 600 episodes to solve cartpole problem
reference for qlearning: https://medium.com/@nancyjemi/level-up-understanding-q-learning-cf739867eb1d
reference for parameters: https://en.wikipedia.org/wiki/Q-learning
"""

import os

# Limit the number of threads for various libraries
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["BLIS_NUM_THREADS"] = "1"  # BLIS
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate/vecLib
os.environ["TBB_NUM_THREADS"] = "1"  # Intel TBB

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


env = gym.make("CartPole-v1", render_mode="human")

rewards = []

# Q-learning parameters
num_episodes = 1000  
learning_rate = 0.1 #important not to set high, then it only considers recent info more
gamma = 0.99  # discount 
epsilon = 1.0  # epsilon greedy exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01  
bins = [10, 10, 10, 10]  # discretization

def discretize_state(state, state_bounds, bins):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    discrete_state = [int(np.clip(ratio * bins[i], 0, bins[i] - 1)) for i, ratio in enumerate(ratios)]
    return tuple(discrete_state)

# Q-table
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]  #cart velocity
state_bounds[3] = [-50, 50]  # pole velocity
q_table = np.zeros(bins + [env.action_space.n])  

# Q-learning loop
t0 = time.time()
for episode in tqdm(range(num_episodes)):
    state = discretize_state(env.reset()[0], state_bounds, bins)
    total_reward = 0

    done = False
    while not done:
        # epsilon-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(q_table[state])  

        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state, state_bounds, bins)
        total_reward += reward

        best_future_q = np.max(q_table[next_state])
        q_table[state][action] += learning_rate * (
            reward + gamma * best_future_q - q_table[state][action]
        )

        state = next_state

    # decay epsilon
    rewards.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# print("Training complete. Testing the agent...")


plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title("Rewards over episodes")
plt.show()

# state = discretize_state(env.reset()[0], state_bounds, bins)
# done = False
# total_reward = 0
# while not done:
#     env.render()
#     action = np.argmax(q_table[state])  # exploiting learned policy
#     next_state, reward, done, _, _ = env.step(action)
#     state = discretize_state(next_state, state_bounds, bins)
#     total_reward += reward

# print(f"Test Reward: {total_reward}")
# env.close()

# plt.savefig('cartpole_env/results/qlearning_plot.png')


