"""
Note - Qlearning takes around 600 episodes to solve cartpole problem
reference for qlearning: https://medium.com/@nancyjemi/level-up-understanding-q-learning-cf739867eb1d
reference for parameters: https://en.wikipedia.org/wiki/Q-learning
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


env = gym.make("CartPole-v1", render_mode="human")

rewards = []

# Q-learning parameters
num_episodes = 1000  
learning_rate = 0.1 
gamma = 0.99  # discount factor 
epsilon = 1.0  # Exploration rate (epsilon-greedy)
epsilon_decay = 0.995  # Decay rate for epsilon
min_epsilon = 0.01  # Minimum exploration rate
bins = [10, 10, 10, 10]  # Discretization bins for state space

def discretize_state(state, state_bounds, bins):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    discrete_state = [int(np.clip(ratio * bins[i], 0, bins[i] - 1)) for i, ratio in enumerate(ratios)]
    return tuple(discrete_state)

# Q-table
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]  # Clip for cart velocity
state_bounds[3] = [-50, 50]  # Clip for pole velocity
q_table = np.zeros(bins + [env.action_space.n])  # Discretized states × actions

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

print("Training complete. Testing the agent...")


plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title("Rewards over episodes")
plt.show()

state = discretize_state(env.reset()[0], state_bounds, bins)
done = False
total_reward = 0
while not done:
    env.render()
    action = np.argmax(q_table[state])  # exploiting learned policy
    next_state, reward, done, _, _ = env.step(action)
    state = discretize_state(next_state, state_bounds, bins)
    total_reward += reward

print(f"Test Reward: {total_reward}")
env.close()


